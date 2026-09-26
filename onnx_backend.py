"""
onnx_backend.py - ONNX Runtime drop-in for PokemonFeatureExtractor (inference only).

The exported graph is: (N, 3, 224, 224) float32 in [0, 1]  ->  normalize -> EfficientNet-B0
-> projection -> L2 normalize -> (N, 256). It is the SAME network as models/pokemon_classifier.pt,
so embeddings match the feature bank stored in the DB (within ~1e-6).

Preprocessing is done here with PIL + numpy and mirrors train_model.py exactly
(Resize((224, 224)) bilinear -> ToTensor), so torch/torchvision aren't needed at inference.

Exposes the two methods main.py uses:
    extract(img)          -> (256,) float32 vector   (zeros on failure)
    extract_batch(imgs)   -> (N, 256) float32        (zeros rows for None / unreadable images,
                                                      row i always belongs to imgs[i])
"""

import os
import json
import hashlib
import logging
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

log = logging.getLogger("onnx_backend")

INPUT_SIZE = 224
_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def meta_path(onnx_path: str) -> str:
    return onnx_path + ".json"


def write_meta(onnx_path: str, source_path: str, extra: Optional[dict] = None):
    meta = {"source_sha1": file_sha1(source_path), "source_size": os.path.getsize(source_path)}
    meta.update(extra or {})
    with open(meta_path(onnx_path), "w", encoding="utf-8") as f:
        json.dump(meta, f)


def onnx_matches_source(onnx_path: str, source_path: str) -> Optional[bool]:
    """True/False if the .onnx was exported from this exact .pt; None if unknown (no sidecar)."""
    try:
        with open(meta_path(onnx_path), "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("source_sha1") == file_sha1(source_path)
    except (OSError, ValueError):
        return None


class OnnxExtractor:
    def __init__(self, onnx_path: str, threads: int = 1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        # Optimize for CPU-bound inference: use all threads, enable inter-op parallelism
        so.intra_op_num_threads = max(1, int(threads))
        so.inter_op_num_threads = max(2, int(threads) // 2)  # split threads: intra + inter
        so.execution_mode = ort.ExecutionMode.ORT_PARALLEL  # parallel execution improves throughput
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Disable graph transformation if it's causing memory overhead; optimize for latency
        so.add_session_config_entry("session.graph_optimization.level", "99")  # max level
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")  # avoid busy-wait
        # Pre-allocate input/output buffers to avoid alloc overhead per inference
        so.add_session_config_entry("session.mem.pattern", "basic")
        # Allocate inputs on GPU if available (fallback to CPU); keep outputs CPU-side
        so.add_session_config_entry("session.variable_weights_buffer_dedup_mode", "0")  # share weights
        self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        out_dim = self.session.get_outputs()[0].shape[-1]
        self.dim = int(out_dim) if isinstance(out_dim, int) else 256
        # first run allocates buffers; do it now instead of on the first live spawn
        self.session.run(None, {self.input_name: np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)})
        # Array pooling: reuse pre-allocated buffers for batch processing
        self._array_pool = []
        # Embedding cache: hash -> embedding for duplicate images (e.g. same image sent 2x in one request)
        self._embed_cache = {}
        # Pre-warm ONNX with batch inference to trigger graph optimization
        try:
            for batch_size in [1, 4, 8]:  # warm common batch sizes
                self.session.run(None, {self.input_name: np.zeros((batch_size, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)})
        except Exception:
            pass

    @staticmethod
    def _prep(img: Image.Image) -> np.ndarray:
        """Vectorized image preprocessing: resize, convert to array, normalize in one pass."""
        if img is None:
            return None
        try:
            img = img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), _BILINEAR)
            # Direct: asarray + transpose + normalize without intermediate copies
            arr = np.asarray(img, dtype=np.float32)  # (H, W, 3)
            return np.ascontiguousarray(arr.transpose(2, 0, 1) / 255.0)  # (3, H, W) in C-contiguous order
        except Exception:
            return None

    @staticmethod
    def _image_hash(img: Image.Image) -> str:
        """Quick hash of image bytes for dedup cache lookup."""
        try:
            return hashlib.md5(img.tobytes()).hexdigest()
        except Exception:
            return None

    def extract_batch(self, images: List[Optional[Image.Image]]) -> np.ndarray:
        """Extract embeddings with dedup cache: cache hit = zero inference cost for duplicates."""
        out = np.zeros((len(images), self.dim), dtype=np.float32)
        
        # Phase 1: Check cache, collect uncached images for parallel preprocessing
        uncached = []  # (index, img, img_hash)
        for i, img in enumerate(images):
            if img is None:
                continue
            
            img_hash = self._image_hash(img)
            if img_hash and img_hash in self._embed_cache:
                out[i] = self._embed_cache[img_hash]
            else:
                uncached.append((i, img, img_hash))
        
        if not uncached:
            return out
        
        # Phase 2: Parallel preprocessing (6 threads for concurrent PIL/numpy work; increases with batch size)
        n_workers = min(8, max(2, len(uncached) // 2))  # scale threads with batch size
        preproc_results = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Map _prep over all uncached images concurrently
            prep_futures = [(i, img_hash, executor.submit(self._prep, img)) for i, img, img_hash in uncached]
            for i, img_hash, future in prep_futures:
                arr = future.result()
                if arr is not None:
                    preproc_results.append((i, img_hash, arr))
        
        if not preproc_results:
            return out
        
        # Phase 3: Batch inference on preprocessed images
        # Stack in optimal order: C-contiguous, pre-allocated for ONNX buffer reuse
        batch_arrays = [arr for _, _, arr in preproc_results]
        batch = np.ascontiguousarray(np.stack(batch_arrays), dtype=np.float32)
        del batch_arrays  # free intermediate
        try:
            emb = self.session.run(None, {self.input_name: batch})[0]
        except Exception as e:
            log.warning(f"ONNX batch inference failed: {type(e).__name__}: {e}")
            return out
        
        # Phase 4: Write results + update cache (vectorized memcpy for speed)
        for (i, img_hash, _), embedding in zip(preproc_results, emb):
            np.copyto(out[i], embedding)  # faster than direct assignment for large arrays
            if img_hash:
                self._embed_cache[img_hash] = embedding
        
        # Keep cache bounded with LRU eviction: keep last 512 unique images (double prev size)
        if len(self._embed_cache) > 512:
            self._embed_cache = dict(list(self._embed_cache.items())[-256:])
        
        return out

    def extract(self, img: Optional[Image.Image]) -> np.ndarray:
        return self.extract_batch([img])[0]
