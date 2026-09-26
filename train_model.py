"""
train_model.py - Stream Processing AI Pokémon Trainer
Trains in batches, clears memory after each batch, writes the feature bank
straight to disk (base_features.npy / base_species.npy / base_meta.json).

There is no database. The feature bank directory is the source of truth; the
API server (server.py, via disk_bank.py) reads exactly those files.
"""

import os
import sys
import json
import logging
import time
import zipfile
import shutil
import subprocess
import gc
import resource
import threading
import queue
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple, Any, Iterator
from io import BytesIO
import re

# ============ NUCLEAR LOG SUPPRESSION ============
logging.root.handlers = []
logging.basicConfig = lambda *args, **kwargs: None

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DATASETS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["GRPC_VERBOSITY"] = "ERROR"

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

for name in logging.root.manager.loggerDict.keys():
    logging.getLogger(name).disabled = True
    logging.getLogger(name).setLevel(logging.CRITICAL)

sys.stderr = open(os.devnull, 'w') if not os.getenv("DEBUG") else sys.stderr

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

# ============ SILENT LOGGER ============
class SilentLogger:
    def info(self, msg, *args, **kwargs):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} INFO {msg}")
    def warning(self, msg, *args, **kwargs):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} WARNING {msg}")
    def error(self, msg, *args, **kwargs):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} ERROR {msg}")
    def debug(self, msg, *args, **kwargs):
        pass

log = SilentLogger()


def log_memory(tag: str = ""):
    try:
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        current_mb = None
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        current_mb = int(line.split()[1]) / 1024
                        break
        except Exception:
            pass
        if current_mb is not None:
            log.info(f"   🧠 Memory{f' ({tag})' if tag else ''}: "
                     f"{current_mb:.0f} MB current RSS, {peak_mb:.0f} MB peak RSS")
        else:
            log.info(f"   🧠 Memory{f' ({tag})' if tag else ''}: {peak_mb:.0f} MB peak RSS")
    except Exception:
        pass


def _trim_memory():
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class PrefetchIterator:
    _SENTINEL = object()

    def __init__(self, source_iterable, maxsize: int = 4):
        self._source = source_iterable
        self._queue: "queue.Queue" = queue.Queue(maxsize=maxsize)
        self._error: Optional[BaseException] = None
        self._thread: Optional[threading.Thread] = None
        try:
            self._thread = threading.Thread(target=self._produce, daemon=True)
            self._thread.start()
        except Exception as e:
            log.warning(f"   ⚠️ Prefetch thread failed to start ({e}), "
                        f"falling back to non-prefetched loading")
            self._thread = None

    def _produce(self):
        try:
            for item in self._source:
                self._queue.put(item)
        except BaseException as e:
            self._error = e
        finally:
            self._queue.put(self._SENTINEL)

    def __iter__(self):
        if self._thread is None:
            yield from self._source
            return
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                if self._error is not None:
                    raise self._error
                return
            yield item


# ============ CONFIGURATION ============

HF_TOKEN = os.getenv("HF_TOKEN")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
STREAM_BATCH_SIZE = int(os.getenv("STREAM_BATCH_SIZE", "64"))
HEAD_EPOCHS = int(os.getenv("HEAD_EPOCHS", "80"))
HEAD_LR = float(os.getenv("HEAD_LR", "1e-3"))
HEAD_BATCH = int(os.getenv("HEAD_BATCH", "512"))
HEAD_WEIGHT_DECAY = float(os.getenv("HEAD_WEIGHT_DECAY", "1e-2"))
HEAD_FEATURE_DROPOUT = float(os.getenv("HEAD_FEATURE_DROPOUT", "0.2"))
HEAD_PATIENCE = int(os.getenv("HEAD_PATIENCE", "15"))
DATASET_NAME = os.getenv("DATASET_NAME", "SpreadSheets/Poketwo-Spawn-Images")
MODEL_OUTPUT = os.getenv("MODEL_OUTPUT", "models/pokemon_classifier.pt")
BANK_DIR = Path(os.getenv("BANK_DIR", "bank"))
BANK_DTYPE = os.getenv("BANK_DTYPE", "fp16").lower()   # fp16 or fp32
AUTO_EXTRACT_ARCHIVES = os.getenv("AUTO_EXTRACT_ARCHIVES", "true").lower() == "true"
MAX_SPECIES = int(os.getenv("MAX_SPECIES", "100"))
MAX_IMAGES_PER_SPECIES = int(os.getenv("MAX_IMAGES_PER_SPECIES", "10"))
VAL_IMAGES_PER_SPECIES = int(os.getenv("VAL_IMAGES_PER_SPECIES", "2"))
MAX_CACHE_IMAGES = int(os.getenv("MAX_CACHE_IMAGES", "0"))
IMAGE_WORKERS = int(os.getenv("IMAGE_WORKERS", str(min(16, (os.cpu_count() or 2) * 2))))
HF_SHARDS = int(os.getenv("HF_SHARDS", "4"))
FEATURE_BANK_DIR = Path(os.getenv("FEATURE_BANK_DIR", "feature_bank_tmp"))
BACKBONE_DIM = 1280

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    log.info(f"🔑 HF_TOKEN configured")

DEVICE = torch.device("cpu")
torch.set_num_threads(os.cpu_count() or 4)


# ============ ARCHIVE EXTRACTION ============

def extract_archive_files():
    if not AUTO_EXTRACT_ARCHIVES:
        return

    archives = []
    for pattern in ["*.zip", "*.ZIP", "*.rar", "*.RAR", "*.7z", "*.7Z"]:
        archives.extend(Path(".").glob(pattern))

    if not archives:
        log.info("📦 No archive files found.")
        return

    log.info(f"📦 Found {len(archives)} archive file(s), extracting...")

    extra_dir = Path("Extra pokemons")
    extra_dir.mkdir(exist_ok=True)

    extracted_count = 0

    for archive_path in archives:
        try:
            ext = archive_path.suffix.lower()
            temp_dir = Path(f"temp_extract_{archive_path.stem}")
            temp_dir.mkdir(exist_ok=True)

            if ext in ['.zip']:
                log.info(f"   📂 Extracting ZIP: {archive_path.name}")
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
            elif ext in ['.rar']:
                log.info(f"   📂 Extracting RAR: {archive_path.name}")
                try:
                    import rarfile
                    with rarfile.RarFile(archive_path) as rf:
                        rf.extractall(temp_dir)
                except Exception:
                    subprocess.run(['unrar', 'x', '-y', str(archive_path), str(temp_dir)],
                                   capture_output=True, check=False)
            elif ext in ['.7z']:
                log.info(f"   📂 Extracting 7z: {archive_path.name}")
                try:
                    import py7zr
                    with py7zr.SevenZipFile(archive_path, 'r') as sz:
                        sz.extractall(temp_dir)
                except Exception:
                    subprocess.run(['7z', 'x', '-y', str(archive_path), f'-o{temp_dir}'],
                                   capture_output=True, check=False)

            extracted = process_extracted_files(temp_dir, extra_dir)
            extracted_count += extracted

            shutil.rmtree(temp_dir)
            archive_path.unlink()
            log.info(f"   ✅ Extracted: {archive_path.name} ({extracted} images)")

        except Exception as e:
            log.error(f"   ❌ Failed to extract {archive_path}: {e}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    if extra_dir.exists():
        species = [f for f in extra_dir.iterdir() if f.is_dir()]
        if species:
            log.info(f"   📁 Extracted {len(species)} species, {extracted_count} images total")


def process_extracted_files(temp_dir: Path, extra_dir: Path) -> int:
    extracted_count = 0
    valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.PNG', '.JPG', '.JPEG', '.WEBP'}

    for root, dirs, files in os.walk(temp_dir):
        root_path = Path(root)
        images = [f for f in files if Path(f).suffix in valid_extensions]

        if images:
            rel_path = root_path.relative_to(temp_dir)
            species_name = rel_path.parts[-1].replace("_", " ").strip() if rel_path.parts else "unknown"

            dest_dir = extra_dir / species_name.replace(" ", "_")
            dest_dir.mkdir(exist_ok=True)

            for img_name in images:
                src = root_path / img_name
                dest = dest_dir / img_name
                if dest.exists():
                    counter = 1
                    stem = Path(img_name).stem
                    suffix = Path(img_name).suffix
                    while dest.exists():
                        dest = dest_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                shutil.move(str(src), str(dest))
                extracted_count += 1

    return extracted_count


# ============ DISK-BACKED FEATURE BANK (training temp + final output) ============

class FeatureBank:
    """
    Memory-mapped, on-disk store of frozen-backbone features for one dataset
    split (train or val), plus the labels. Used as scratch space during training.
    """

    def __init__(self, name: str, base_dir: Path, dim: int = BACKBONE_DIM,
                 capacity: int = 100_000):
        self.name = name
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.features_path = self.base_dir / f"{name}.npy"
        self.labels_path = self.base_dir / f"{name}_labels.pt"
        self.idx_path = self.base_dir / f"{name}.idx"

        self._capacity = capacity
        if not self.features_path.exists():
            mm = np.lib.format.open_memmap(
                str(self.features_path), mode="w+",
                dtype=np.float16, shape=(capacity, dim),
            )
            del mm

        self._features = np.lib.format.open_memmap(
            str(self.features_path), mode="r+", dtype=np.float16,
        )
        if self.labels_path.exists():
            self._labels: List[int] = torch.load(self.labels_path, weights_only=False)
        else:
            self._labels = []
        self._n = len(self._labels)

    def append_batch(self, feats: torch.Tensor, labels: torch.Tensor):
        if feats.size(0) == 0:
            return
        need = self._n + feats.size(0)
        if need > self._features.shape[0]:
            raise RuntimeError(
                f"FeatureBank '{self.name}' capacity {self._features.shape[0]} "
                f"exceeded ({need} rows). Raise the `capacity=` argument."
            )
        self._features[self._n:need] = feats.to(torch.float16).cpu().numpy()
        self._labels.extend(labels.tolist())
        self._n = need

    def flush(self):
        try:
            self._features.flush()
        except Exception:
            pass
        torch.save(self._labels, self.labels_path)
        self.idx_path.write_text(json.dumps({"n": self._n, "dim": self.dim}))

    def __len__(self):
        return self._n

    def iter_batches(self, batch_size: int, shuffle: bool = False, seed: int = 0):
        n = self._n
        if n == 0:
            return
        if shuffle:
            g = torch.Generator().manual_seed(seed)
            order = torch.randperm(n, generator=g)
        else:
            order = torch.arange(n)
        all_labels = torch.tensor(self._labels, dtype=torch.long)
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            xb_np = np.asarray(self._features[idx.numpy()])
            yield torch.from_numpy(xb_np).float(), idx, all_labels[idx]

    def labels_tensor(self) -> torch.Tensor:
        return torch.tensor(self._labels, dtype=torch.long)

    def close(self):
        try:
            self.flush()
        except Exception:
            pass
        try:
            del self._features
        except Exception:
            pass


# ============ AI MODEL ============

class PokemonFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        classifier = self.backbone.classifier
        if isinstance(classifier, nn.Sequential):
            linear_layers = [layer for layer in classifier if isinstance(layer, nn.Linear)]
            if not linear_layers:
                raise RuntimeError(f"Could not find a Linear layer in EfficientNet classifier: {classifier!r}")
            backbone_dim = linear_layers[-1].in_features
        elif isinstance(classifier, nn.Linear):
            backbone_dim = classifier.in_features
        else:
            raise RuntimeError(f"Unsupported EfficientNet classifier type: {type(classifier).__name__}")
        self.backbone.classifier = nn.Identity()

        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.to(DEVICE)
        self.eval()

    @torch.no_grad()
    def extract(self, img: Image.Image) -> np.ndarray:
        if img is None:
            return np.zeros(256)
        try:
            img_tensor = self.transform(img).unsqueeze(0).to(DEVICE)
            img_tensor = self.normalize(img_tensor)
            features = self.backbone(img_tensor)
            projected = self.projection(features)
            projected = F.normalize(projected, p=2, dim=1)
            return projected.cpu().numpy().flatten()
        except Exception as e:
            log.warning(f"   ⚠️ Feature extraction failed: {e}")
            return np.zeros(256)

    def extract_batch(self, images: List[Image.Image], grad: bool = False) -> np.ndarray:
        valid_images = []
        for img in images:
            if img is not None and isinstance(img, Image.Image):
                try:
                    valid_images.append(self.transform(img).unsqueeze(0))
                except Exception:
                    continue

        if not valid_images:
            return np.zeros((len(images), 256))

        try:
            batch_tensor = torch.cat(valid_images, dim=0).to(DEVICE)
            batch_tensor = self.normalize(batch_tensor)

            with torch.no_grad():
                features = self.backbone(batch_tensor)

            if grad:
                projected = self.projection(features)
            else:
                with torch.no_grad():
                    projected = self.projection(features)
            projected = F.normalize(projected, p=2, dim=1)

            if grad:
                result = projected
                if len(valid_images) < len(images):
                    pad = torch.zeros(len(images) - len(valid_images), result.shape[1], device=result.device)
                    result = torch.cat([result, pad], dim=0)
                return result

            result = projected.cpu().numpy()
            if len(valid_images) < len(images):
                padded = np.zeros((len(images), result.shape[1]))
                padded[:len(valid_images)] = result
                return padded
            return result
        except Exception as e:
            log.warning(f"   ⚠️ Batch feature extraction failed: {e}")
            if grad:
                return torch.zeros(len(images), 256, device=DEVICE, requires_grad=True)
            return np.zeros((len(images), 256))


class CosineHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, scale: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * F.linear(F.normalize(x, dim=1), F.normalize(self.weight, dim=1))


class PokemonClassifier(nn.Module):
    def __init__(self, num_species: int):
        super().__init__()
        self.feature_extractor = PokemonFeatureExtractor()
        self.classifier = CosineHead(256, num_species)

    def forward_batch(self, images: List[Image.Image], grad: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor.extract_batch(images, grad=grad)
        if grad:
            features_tensor = features.to(DEVICE)
        else:
            features_tensor = torch.tensor(features, dtype=torch.float32).to(DEVICE)
        logits = self.classifier(features_tensor)
        return features_tensor, logits


# ============ STREAMING DATASET ============

class StreamingPokemonDataset(IterableDataset):
    def __init__(self, extra_dir: str = "Extra pokemons"):
        self.extra_dir = extra_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.PILToTensor(),
        ])

        self._cache: List[Tuple[torch.Tensor, int]] = []
        self._cache_ready = False

        self.disk_cache_enabled = os.getenv("DISK_CACHE", "true").lower() == "true"
        self.disk_cache_dir = Path(os.getenv("DISK_CACHE_DIR", "image_cache"))
        self._disk_chunk_files: List[Path] = []
        self._resume_partial_chunks: List[Path] = []
        self._complete_marker = self.disk_cache_dir / "_complete.marker"
        if self.disk_cache_enabled:
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
            existing = sorted(self.disk_cache_dir.glob("chunk_*.pt"))
            if existing and self._complete_marker.exists():
                self._disk_chunk_files = existing
                self._cache_ready = True
                log.info(f"   💽 Found {len(existing)} cached chunk files on disk "
                         f"from a previous COMPLETED run - will replay from disk, no HF re-stream")
            elif existing:
                self._resume_partial_chunks = existing
                log.info(f"   💽 Found {len(existing)} PARTIAL cached chunk files "
                         f"(no completion marker - a previous run was cut off) - "
                         f"will replay those, then resume streaming from Hugging "
                         f"Face to fill the rest")

        self._val_cache: List[Tuple[torch.Tensor, int]] = []
        self.val_cache_path = self.disk_cache_dir / "val_cache.pt"
        if self.disk_cache_enabled and self.val_cache_path.exists():
            try:
                self._val_cache = torch.load(self.val_cache_path, weights_only=False)
                log.info(f"   💽 Loaded {len(self._val_cache)} held-out validation images from disk")
            except Exception as e:
                log.warning(f"   ⚠️ Failed to load validation cache: {e}")

        self._build_species_mapping()

        log.info(f"📊 Species mapping built: {len(self.species_to_idx)} species")

    def _build_species_mapping(self):
        local_species = set()
        extra_path = Path(self.extra_dir)
        if extra_path.exists():
            for folder in extra_path.iterdir():
                if folder.is_dir():
                    species = folder.name.replace("_", " ").strip().lower()
                    if species:
                        local_species.add(species)

        hf_species = set()
        try:
            from datasets import load_dataset

            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = devnull
                sys.stderr = devnull
                try:
                    ds = load_dataset(DATASET_NAME, split="train", streaming=True)
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr

            features = ds.features
            label_col = None
            for col in ["label", "text", "name", "species", "pokemon"]:
                if col in features:
                    label_col = col
                    break

            if label_col:
                col_feature = features[label_col]
                names = getattr(col_feature, "names", None)
                if names:
                    hf_species = {str(n).strip().lower() for n in names}
                    log.info(f"   📋 Got {len(hf_species)} species directly from dataset schema")
                else:
                    count = 0
                    for row in ds:
                        raw_label = row[label_col]
                        if isinstance(raw_label, int):
                            raw_label = features[label_col].int2str(raw_label)
                        species = str(raw_label).strip().lower()
                        hf_species.add(species)
                        count += 1
                        if count % 2000 == 0:
                            log.info(f"   🔎 Scanned {count} rows, found {len(hf_species)} species so far")
                        if count > 20000:
                            break
        except Exception as e:
            log.warning(f"Could not get species from Hugging Face: {e}")

        all_species = sorted(local_species | hf_species)

        if not all_species:
            log.error("❌ No species found!")
            self.species_to_idx = {}
            self.idx_to_species = {}
            return

        self.species_to_idx = {s: i for i, s in enumerate(all_species)}
        self.idx_to_species = {i: s for s, i in self.species_to_idx.items()}

        if MAX_SPECIES > 0 and len(all_species) > MAX_SPECIES:
            local_list = sorted(local_species)
            hf_list = sorted(s for s in all_species if s not in local_species)
            selected = local_list[:MAX_SPECIES]
            if len(selected) < MAX_SPECIES:
                remaining = MAX_SPECIES - len(selected)
                selected += hf_list[:remaining]
            self.species_to_idx = {s: i for i, s in enumerate(selected)}
            self.idx_to_species = {i: s for s, i in self.species_to_idx.items()}
            log.info(f"   Limited to {len(selected)} species (MAX_SPECIES={MAX_SPECIES})")

    def __iter__(self) -> Iterator:
        if self._disk_chunk_files:
            log.info(f"   💽 Replaying {len(self._disk_chunk_files)} chunk files from disk "
                     f"(no HF re-stream, no network)")
            for chunk_path in self._disk_chunk_files:
                try:
                    chunk = torch.load(chunk_path, weights_only=False)
                    for item in chunk:
                        yield item
                    del chunk
                except Exception as e:
                    log.warning(f"   ⚠️ Failed to load cache chunk {chunk_path}: {e}")
            gc.collect()
            _trim_memory()
            return

        if self._cache_ready:
            log.info(f"   ♻️  Replaying {len(self._cache)} cached images (no HF re-stream)")
            for item in self._cache:
                yield item
            return

        target_total = len(self.species_to_idx) * MAX_IMAGES_PER_SPECIES
        species_counts: Dict[str, int] = {}
        collected = 0
        cache_capped_warned = False
        t_start = time.time()

        DISK_CHUNK_SIZE = 200
        disk_buffer: List[Tuple[torch.Tensor, int]] = []
        chunk_idx = 0

        if self._resume_partial_chunks:
            chunk_idx = len(self._resume_partial_chunks)
            replayed = 0
            for chunk_path in self._resume_partial_chunks:
                try:
                    chunk = torch.load(chunk_path, weights_only=False)
                    for tensor, label in chunk:
                        species = self.idx_to_species.get(label)
                        if species:
                            species_counts[species] = species_counts.get(species, 0) + 1
                            collected += 1
                            replayed += 1
                        yield tensor, label
                    del chunk
                except Exception as e:
                    log.warning(f"   ⚠️ Failed to replay partial chunk {chunk_path}: {e}")
            gc.collect()
            _trim_memory()
            log.info(f"   💽 Replayed {replayed} images from partial cache "
                     f"({collected}/{target_total}) - resuming HF stream for the rest")

        def _flush_disk_chunk():
            nonlocal disk_buffer, chunk_idx
            if not self.disk_cache_enabled or not disk_buffer:
                return
            chunk_path = self.disk_cache_dir / f"chunk_{chunk_idx:05d}.pt"
            try:
                torch.save(disk_buffer, chunk_path)
                self._disk_chunk_files.append(chunk_path)
                chunk_idx += 1
                if self._val_cache:
                    torch.save(self._val_cache, self.val_cache_path)
            except Exception as e:
                log.warning(f"   ⚠️ Failed to write cache chunk to disk: {e}")
            disk_buffer = []
            gc.collect()
            _trim_memory()

        def _emit(tensor, label, species):
            nonlocal collected, cache_capped_warned
            species_idx = species_counts.get(species, 0)
            species_counts[species] = species_idx + 1
            collected += 1

            if species_idx < VAL_IMAGES_PER_SPECIES:
                self._val_cache.append((tensor, label))
                return None

            if self.disk_cache_enabled:
                pass
            elif MAX_CACHE_IMAGES > 0 and len(self._cache) < MAX_CACHE_IMAGES:
                self._cache.append((tensor, label))
            elif MAX_CACHE_IMAGES > 0 and not cache_capped_warned:
                cache_capped_warned = True
                log.warning(f"   ⚠️ Cache cap ({MAX_CACHE_IMAGES} images) reached")

            if self.disk_cache_enabled:
                disk_buffer.append((tensor, label))
                if len(disk_buffer) >= DISK_CHUNK_SIZE:
                    _flush_disk_chunk()
            return tensor, label

        extra_path = Path(self.extra_dir)
        if extra_path.exists():
            for folder in extra_path.iterdir():
                if not folder.is_dir():
                    continue

                species = folder.name.replace("_", " ").strip().lower()
                if species not in self.species_to_idx:
                    continue

                label = self.species_to_idx[species]
                valid_extensions = {".png", ".jpg", ".jpeg", ".webp"}

                for img_path in folder.iterdir():
                    if species_counts.get(species, 0) >= MAX_IMAGES_PER_SPECIES:
                        break
                    if img_path.suffix.lower() not in valid_extensions:
                        continue
                    try:
                        img = Image.open(img_path).convert("RGB")
                        if img.size[0] > 10 and img.size[1] > 10:
                            item = _emit(self.transform(img), label, species)
                            if item is not None:
                                yield item
                    except Exception:
                        continue

        log.info(f"   📂 Local images collected: {collected}/{target_total}")

        if collected >= target_total:
            _flush_disk_chunk()
            self._cache_ready = True
            self._write_completion_marker(collected, target_total)
            log.info(f"   ✅ Quota already met from local images, skipping HF stream")
            return

        stream_error = False
        try:
            from datasets import load_dataset

            ds = load_dataset(DATASET_NAME, split="train", streaming=True)

            features = ds.features
            label_col = None
            image_col = None

            for col in ["label", "text", "name", "species", "pokemon"]:
                if col in features:
                    label_col = col
                    break
            for col in ["image", "img", "picture"]:
                if col in features:
                    image_col = col
                    break

            if not label_col or not image_col:
                label_col = list(features.keys())[0]
                image_col = list(features.keys())[1] if len(features) > 1 else list(features.keys())[0]

            rows_scanned = 0
            in_flight: "deque" = deque()

            def _drain_one():
                fut, lbl, spc = in_flight.popleft()
                tensor = fut.result()
                return _emit(tensor, lbl, spc)

            row_queue: "queue.Queue" = queue.Queue(maxsize=max(4, HF_SHARDS * 4))
            stop_event = threading.Event()
            _SHARD_DONE = object()

            def _shard_worker(shard_ds, shard_idx):
                try:
                    for row in shard_ds:
                        if stop_event.is_set():
                            return
                        try:
                            raw_label = row[label_col]
                            if isinstance(raw_label, int):
                                raw_label = features[label_col].int2str(raw_label)
                            species = str(raw_label).strip().lower()
                            if species not in self.species_to_idx:
                                continue
                            if species_counts.get(species, 0) >= MAX_IMAGES_PER_SPECIES:
                                continue
                            img = row[image_col]
                            if img is None:
                                continue
                            if not isinstance(img, Image.Image):
                                img = Image.open(BytesIO(img))
                            if img.size[0] < 10 or img.size[1] < 10:
                                continue
                            label = self.species_to_idx[species]
                            row_queue.put((img, label, species))
                        except Exception:
                            continue
                except Exception as e:
                    log.warning(f"   ⚠️ HF shard {shard_idx} reader stopped early: {e}")
                finally:
                    row_queue.put(_SHARD_DONE)

            n_readers = max(1, min(HF_SHARDS, ds.num_shards))
            if n_readers < HF_SHARDS:
                log.warning(f"   ⚠️ Dataset only exposes {ds.num_shards} underlying "
                            f"file(s) - HF_SHARDS={HF_SHARDS} clamped to {n_readers}.")

            shard_threads = []
            for i in range(n_readers):
                shard_ds = ds.shard(num_shards=n_readers, index=i, contiguous=True) if n_readers > 1 else ds
                t = threading.Thread(target=_shard_worker, args=(shard_ds, i), daemon=True)
                shard_threads.append(t)

            for t in shard_threads:
                t.start()
            active_shards = len(shard_threads)
            log.info(f"   🧵 Streaming with {active_shards} parallel HF reader(s), "
                     f"{IMAGE_WORKERS} decode worker(s)")

            with ThreadPoolExecutor(max_workers=IMAGE_WORKERS) as executor:
                while active_shards > 0:
                    entry = row_queue.get()
                    if entry is _SHARD_DONE:
                        active_shards -= 1
                        continue

                    img, label, species = entry
                    rows_scanned += 1

                    if rows_scanned % 200 == 0:
                        elapsed = time.time() - t_start
                        log.info(f"   🔎 Scanned {rows_scanned} HF rows, "
                                 f"kept {collected}/{target_total} images "
                                 f"({elapsed:.0f}s elapsed)")

                    in_flight.append((executor.submit(self.transform, img), label, species))
                    if len(in_flight) >= IMAGE_WORKERS:
                        item = _drain_one()
                        if item is not None:
                            yield item

                    if collected >= target_total:
                        log.info(f"   ✅ Quota met ({collected}/{target_total}) "
                                 f"after scanning {rows_scanned} HF rows")
                        stop_event.set()
                        break

                while in_flight:
                    try:
                        item = _drain_one()
                        if item is not None:
                            yield item
                    except Exception:
                        continue

            stop_event.set()
            for t in shard_threads:
                t.join(timeout=5)

        except Exception as e:
            log.warning(f"Error streaming from Hugging Face: {e}")
            stream_error = True

        if collected < target_total:
            log.warning(f"   ⚠️ Only found {collected}/{target_total} images "
                        f"before HF dataset was exhausted")

        _flush_disk_chunk()
        self._cache_ready = True
        if not stream_error:
            self._write_completion_marker(collected, target_total)

    def _write_completion_marker(self, collected: int, target_total: int):
        if not self.disk_cache_enabled:
            return
        try:
            self._complete_marker.write_text(json.dumps({
                "collected": collected,
                "target_total": target_total,
                "species": len(self.species_to_idx),
                "completed_at": time.time(),
            }))
        except Exception as e:
            log.warning(f"   ⚠️ Failed to write cache-completion marker: {e}")

    def get_val_set(self) -> List[Tuple[torch.Tensor, int]]:
        return self._val_cache

    def get_num_species(self) -> int:
        return len(self.species_to_idx)


# ============ STREAMING TRAINER ============

@torch.no_grad()
def _backbone_features(fe: "PokemonFeatureExtractor", batch_u8: torch.Tensor) -> torch.Tensor:
    x = fe.normalize(batch_u8.float().div_(255.0))
    return fe.backbone(x)


@torch.no_grad()
def build_feature_bank(items, fe: "PokemonFeatureExtractor", bank: "FeatureBank",
                       batch_size: int, total_hint: int = 0, tag: str = "train") -> int:
    buf_i: List[torch.Tensor] = []
    buf_l: List[int] = []
    n = 0
    n_flush = 0
    t0 = time.time()
    n_at_last_log = 0
    t_at_last_log = t0

    def flush():
        nonlocal n, n_flush, n_at_last_log, t_at_last_log
        if not buf_i:
            return
        feats = _backbone_features(fe, torch.stack(buf_i))
        labels = torch.tensor(buf_l, dtype=torch.long)
        bank.append_batch(feats, labels)
        n += len(buf_i)
        n_flush += 1
        buf_i.clear()
        buf_l.clear()
        if n_flush % 25 == 0:
            of = f"/{total_hint}" if total_hint else ""
            now = time.time()
            window_rate = (n - n_at_last_log) / max(now - t_at_last_log, 1e-6)
            lifetime_rate = n / max(now - t0, 1e-6)
            log.info(f"   🧊 [{tag}] {n}{of} images embedded "
                     f"({window_rate:.1f} img/s recent, {lifetime_rate:.1f} img/s avg)")
            n_at_last_log = n
            t_at_last_log = now
        if n_flush % 100 == 0:
            bank.flush()
            gc.collect()
            _trim_memory()
            log_memory(f"{tag} feature pass")

    for img, label in items:
        buf_i.append(img)
        buf_l.append(int(label))
        if len(buf_i) >= batch_size:
            flush()
    flush()
    bank.flush()
    return n


@torch.no_grad()
def _embed_bank(fe: "PokemonFeatureExtractor", bank: "FeatureBank", bs: int = 2048) -> Tuple[torch.Tensor, torch.Tensor]:
    E_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []
    for xb, idx, yb in bank.iter_batches(bs, shuffle=False):
        E_list.append(F.normalize(fe.projection(xb), p=2, dim=1))
        y_list.append(yb)
    if not E_list:
        return torch.empty(0, 256), torch.empty(0, dtype=torch.long)
    return torch.cat(E_list), torch.cat(y_list)


@torch.no_grad()
def _evaluate(model: "PokemonClassifier", Etr, ytr, Ev, yv) -> Tuple[float, float]:
    if Ev.size(0) == 0:
        return 0.0, 0.0
    cls_acc = (model.classifier(Ev).argmax(1) == yv).float().mean().item() * 100
    correct = 0
    for i in range(0, Ev.size(0), 512):
        sims = Ev[i:i + 512] @ Etr.T
        correct += (ytr[sims.argmax(1)] == yv[i:i + 512]).sum().item()
    return cls_acc, 100 * correct / Ev.size(0)


def train_head(model: "PokemonClassifier", train_bank: "FeatureBank",
               val_bank: "FeatureBank") -> float:
    fe = model.feature_extractor
    params = list(fe.projection.parameters()) + list(model.classifier.parameters())
    optimizer = optim.AdamW(params, lr=HEAD_LR, weight_decay=HEAD_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(HEAD_EPOCHS, 1))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    N = len(train_bank)
    has_val = len(val_bank) > 0
    if not has_val:
        log.warning("   ⚠️ No held-out validation set - training all epochs and keeping the last one")

    if has_val:
        Ev, yv = _embed_bank(fe, val_bank, bs=2048)
    else:
        Ev, yv = torch.empty(0, 256), torch.empty(0, dtype=torch.long)

    best_acc, best_epoch, best_state, stale = -1.0, 0, None, 0

    for epoch in range(HEAD_EPOCHS):
        t0 = time.time()
        fe.projection.train()
        model.classifier.train()

        loss_sum, correct = 0.0, 0
        for xb, idx, yb in train_bank.iter_batches(HEAD_BATCH, shuffle=True, seed=epoch):
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            xb = F.dropout(xb, p=HEAD_FEATURE_DROPOUT, training=True)
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                emb = F.normalize(fe.projection(xb), p=2, dim=1)
                logits = model.classifier(emb)
                loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * yb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
        scheduler.step()

        fe.projection.eval()
        model.classifier.eval()
        if has_val:
            Etr, ytr = _embed_bank(fe, train_bank, bs=2048)
            cls_acc, ret_acc = _evaluate(model, Etr, ytr, Ev, yv)
            del Etr, ytr
        else:
            cls_acc, ret_acc = 0.0, 0.0
        log.info(f"   📊 Epoch {epoch + 1}/{HEAD_EPOCHS}: loss {loss_sum / max(N, 1):.3f}, "
                 f"train acc {100 * correct / max(N, 1):.1f}%, val acc {cls_acc:.1f}%, "
                 f"val retrieval {ret_acc:.1f}% ({time.time() - t0:.1f}s)")

        score = ret_acc if has_val else float(epoch)
        if score > best_acc:
            best_acc, best_epoch, stale = score, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if has_val and stale >= HEAD_PATIENCE:
                log.info(f"   ⏸️ No val improvement in {HEAD_PATIENCE} epochs "
                         f"(best: {best_acc:.1f}% at epoch {best_epoch}), stopping")
                break

        gc.collect()
        _trim_memory()

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return best_acc if has_val else 0.0


# ============ WRITE FINAL BANK TO DISK ============

def _sha1_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_bank_to_disk(model: "PokemonClassifier", train_bank: "FeatureBank",
                       dataset, out_dir: Path):
    """
    Re-embeds every training image with the FINAL (best) model and writes the
    inference-time feature bank directly to disk, in exactly the format
    disk_bank.DiskFeatureBank reads:

        out_dir/base_features.npy    (N, 256) fp16 or fp32
        out_dir/base_species.npy     (N,) unicode strings
        out_dir/base_meta.json       rows, dim, dtype, model sha1, exported_at

    This replaces the old "write to Postgres, then export_to_disk.py" workflow.
    The learned.jsonl in out_dir is left alone - s!learn / s!forget additions
    live there independently.
    """
    log.info(f"\n💾 Writing feature bank to disk ({out_dir})...")
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    N = len(train_bank)
    if N == 0:
        log.error("   ⚠️ No training features to write - skipping")
        return

    np_dtype = np.float16 if BANK_DTYPE == "fp16" else np.float32

    features_path = out_dir / "base_features.npy"
    species_path = out_dir / "base_species.npy"
    meta_path = out_dir / "base_meta.json"

    # Re-open or create the destination memmap
    mm = np.lib.format.open_memmap(
        str(features_path), mode="w+", dtype=np_dtype, shape=(N, 256),
    )

    species_arr = np.empty(N, dtype="U64")
    fe = model.feature_extractor
    fe.projection.eval()

    i = 0
    for xb, idx, yb in train_bank.iter_batches(2048, shuffle=False):
        with torch.no_grad():
            emb = F.normalize(fe.projection(xb), p=2, dim=1).cpu().numpy()
        # Convert to the target dtype in place; also re-normalize defensively
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        emb = emb / norms
        for vec, lbl in zip(emb, yb.tolist()):
            species = dataset.idx_to_species.get(lbl, "unknown")
            mm[i] = vec.astype(np_dtype, copy=False)
            species_arr[i] = species
            i += 1
        gc.collect()

    mm.flush()
    del mm

    if i != N:
        # Trim (shouldn't happen, but be safe)
        features = np.load(features_path, mmap_mode="r")
        trimmed = np.array(features[:i])
        np.save(features_path, trimmed)
        species_arr = species_arr[:i]
        N = i

    np.save(species_path, species_arr, allow_pickle=False)

    model_sha = _sha1_file(Path(MODEL_OUTPUT))
    meta = {
        "rows": int(N),
        "dim": 256,
        "dtype": BANK_DTYPE,
        "model_path": str(MODEL_OUTPUT),
        "model_sha1": model_sha,
        "written_at": int(time.time()),
        "source": "train_model",
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    log.info(f"   ✅ Wrote {N:,} features ({features_path.stat().st_size / 1e6:.1f} MB) "
             f"to {features_path}, {species_path.name}, {meta_path.name} "
             f"in {time.time() - t0:.0f}s")
    log.info(f"   ℹ️  learned.jsonl is untouched - taught examples stay in effect.")


def _detect_container_memory_limit_mb() -> Optional[float]:
    candidates = [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ]
    for path in candidates:
        try:
            with open(path) as f:
                raw = f.read().strip()
            if raw == "max":
                continue
            limit_bytes = int(raw)
            if limit_bytes <= 0 or limit_bytes > (1 << 52):
                continue
            return limit_bytes / (1024 * 1024)
        except Exception:
            continue
    return None


def _get_current_rss_mb() -> Optional[float]:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return None


def stream_train():
    global STREAM_BATCH_SIZE

    log.info("📥 Pre-loading AI model...")
    try:
        from torchvision import models as _m
        _ = _m.efficientnet_b0(weights=_m.EfficientNet_B0_Weights.DEFAULT)
        log.info("✅ Model loaded and cached!")
    except Exception as e:
        log.warning(f"⚠️ Model pre-load failed: {e}")

    log_memory("baseline, right after model load")
    baseline_rss_mb = _get_current_rss_mb()
    container_limit_mb = _detect_container_memory_limit_mb()

    if container_limit_mb is not None:
        log.info(f"   📦 Detected container memory limit: {container_limit_mb:.0f} MB")
        if baseline_rss_mb is not None:
            safety_budget_mb = container_limit_mb * 0.75 - baseline_rss_mb
            safe_batch_size = max(4, int(safety_budget_mb / 15))
            if STREAM_BATCH_SIZE > safe_batch_size:
                log.warning(f"   ⚠️ STREAM_BATCH_SIZE={STREAM_BATCH_SIZE} looks risky for a "
                            f"{container_limit_mb:.0f}MB container with a {baseline_rss_mb:.0f}MB "
                            f"baseline - auto-lowering to {safe_batch_size} to avoid OOM.")
                STREAM_BATCH_SIZE = safe_batch_size
    else:
        log.info(f"   📦 Could not detect a container memory limit")

    log.info("")
    log.info("🚀 Pokémon AI Trainer - STREAMING MODE (disk-backed, no DB)")
    log.info("=" * 60)
    log.info("   ✅ Streams/caches images once, embeds with the frozen backbone")
    log.info("   ✅ Trains the head on shuffled cached features")
    log.info("   ✅ Writes the final feature bank to disk for the API server")
    log.info("=" * 60)

    if HF_TOKEN:
        log.info(f"🔑 HF_TOKEN: ✅ Set")
    else:
        log.warning(f"🔑 HF_TOKEN: ❌ Not set")

    log.info("\n📦 Checking for archive files...")
    extract_archive_files()

    log.info("\n📂 Initializing streaming dataset...")
    dataset = StreamingPokemonDataset(extra_dir="Extra pokemons")
    num_species = dataset.get_num_species()

    if num_species == 0:
        log.error("❌ No species found! Exiting.")
        return

    log.info(f"   📊 Total species: {num_species}")

    log.info("\n🧠 Initializing model...")
    model = PokemonClassifier(num_species=num_species)
    model.to(DEVICE)
    model.eval()
    for p in model.feature_extractor.backbone.parameters():
        p.requires_grad_(False)

    # ============ PHASE 1 ============
    log.info("\n🧊 Phase 1/3: embedding every image once with the frozen backbone...")
    log.info(f"   💾 On-disk feature banks go to: {FEATURE_BANK_DIR}/")
    t0 = time.time()
    fe = model.feature_extractor
    train_capacity = max(200_000, num_species * MAX_IMAGES_PER_SPECIES + 10_000)
    val_capacity = max(20_000, num_species * max(VAL_IMAGES_PER_SPECIES, 1) + 5_000)
    train_bank = FeatureBank("train", FEATURE_BANK_DIR, dim=BACKBONE_DIM, capacity=train_capacity)
    val_bank = FeatureBank("val", FEATURE_BANK_DIR, dim=BACKBONE_DIM, capacity=val_capacity)
    total_hint = num_species * max(MAX_IMAGES_PER_SPECIES - VAL_IMAGES_PER_SPECIES, 0)
    n_train = build_feature_bank(
        PrefetchIterator(dataset, maxsize=STREAM_BATCH_SIZE * 2),
        fe, train_bank, STREAM_BATCH_SIZE, total_hint=total_hint, tag="train",
    )
    n_val = build_feature_bank(
        dataset.get_val_set(), fe, val_bank, STREAM_BATCH_SIZE, tag="val",
    )
    gc.collect()
    _trim_memory()
    log.info(f"   ✅ {n_train} train + {n_val} val images embedded in {time.time() - t0:.0f}s")
    log_memory("after feature pass")

    if n_train == 0:
        log.error("❌ No training images were collected! Exiting.")
        return

    ytr = train_bank.labels_tensor()
    seen = int(ytr.unique().numel())
    del ytr
    if seen < num_species:
        log.warning(f"   ⚠️ Only {seen}/{num_species} species have training images")

    # ============ PHASE 2 ============
    log.info(f"\n🎯 Phase 2/3: training head ({n_train} samples, {num_species} species, "
             f"{HEAD_EPOCHS} epochs max, lr {HEAD_LR}, batch {HEAD_BATCH})")
    log.info("-" * 60)
    t0 = time.time()
    best_acc = train_head(model, train_bank, val_bank)
    log.info("-" * 60)
    log.info(f"   ✅ Head trained in {time.time() - t0:.0f}s (best val retrieval acc: {best_acc:.1f}%)")

    os.makedirs(os.path.dirname(MODEL_OUTPUT) or ".", exist_ok=True)
    torch.save(model.feature_extractor.state_dict(), MODEL_OUTPUT)
    log.info(f"   ✅ Saved model to {MODEL_OUTPUT}")

    # ============ PHASE 3: write bank to disk ============
    log.info("\n🗄️ Phase 3/3: writing feature bank to disk")
    write_bank_to_disk(model, train_bank, dataset, BANK_DIR)

    log.info("-" * 60)
    log.info(f"✅ Training complete!")
    log.info(f"   Best validation retrieval accuracy: {best_acc:.1f}%")
    log.info(f"   Model saved to: {MODEL_OUTPUT}")
    log.info(f"   Bank written to: {BANK_DIR}/")

    log.info("\n" + "=" * 60)
    log.info("✅ All done! Restart or POST /admin/reload on the API server.")
    log.info("=" * 60)

    try:
        train_bank.close()
        val_bank.close()
        for f in FEATURE_BANK_DIR.glob("*"):
            try:
                f.unlink()
            except Exception:
                pass
        FEATURE_BANK_DIR.rmdir()
    except Exception:
        pass


if __name__ == "__main__":
    stream_train()