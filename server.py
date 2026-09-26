"""
server.py - AI_Model: the Pokémon identifier as an HTTP API.

    POST /v1/predict          one image  -> species, score, top neighbours
    POST /v1/predict/batch    N images   -> one result per image
    POST /v1/predict/urls     N urls     -> one result per image (server fetches)
    POST /v1/learn            teach a new example
    POST /v1/learn/batch      teach many examples in one request
    POST /v1/forget           delete learned examples
    GET  /v1/stats            bank size + latency numbers
    GET  /health              readiness
    POST /admin/reload        re-read model + feature bank
    POST /admin/train         start a training run in a subprocess
    GET  /admin/train/status  progress / log tail
    POST /admin/train/stop    abort the run

Feature bank: purely file-backed (bank/base_features.npy + base_species.npy).
Learned examples live in bank/learned.jsonl. There is no database anywhere.

Inference runs on ONNX Runtime, so this process never imports torch. torch is
only used by the subprocesses that train or (re-)export the model.
"""

import io
import os
import math
import hashlib
import urllib.error
import urllib.parse
import urllib.request
import re
import sys
import time
import json
import asyncio
import logging
import secrets
import threading
import subprocess
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Optional
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

try:  # optional: keep-alive connections for fetching images; falls back to urllib in a thread without it
    import aiohttp
except ImportError:
    aiohttp = None

# Pin BLAS/OpenMP to ONE thread in this process. The cosine match (`matrix @ vec`) is a BLAS call, and by
# default every concurrent request spins up (and busy-waits) a thread per host core: it burns CPU and
# slows the real work. Parallelism here comes from INFER_CONCURRENCY slots instead. Child processes
# (training / ONNX export) get the original values back, see child_env().
_BLAS_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
_BLAS_ORIG = {k: os.environ.get(k) for k in _BLAS_VARS}
for _k in _BLAS_VARS:
    os.environ[_k] = "1"

import numpy as np
from PIL import Image
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from disk_bank import DiskFeatureBank, new_species_name, species_key  # noqa: F401

BANK_DIR = os.getenv("BANK_DIR", "bank")
APP_DIR = Path(__file__).resolve().parent

log = logging.getLogger("ai_model")
if not log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False


# ============================================================ settings
def _env(name: str, default: str = "") -> str:
    """
    os.getenv minus any trailing "  # comment". python-dotenv strips those itself, but some panels and
    launchers load .env without doing so, which used to turn `ONNX_THREADS=1   # note` into a crash.
    """
    v = os.getenv(name)
    if v is None:
        return default
    return re.split(r"\s+#", v, maxsplit=1)[0].strip()


def _env_int(name: str, default: int) -> int:
    v = _env(name, str(default))
    try:
        return int(v)
    except ValueError:
        raise ValueError(f"{name} must be a whole number, got {v!r}") from None


def _env_float(name: str, default: float) -> float:
    v = _env(name, str(default))
    try:
        return float(v)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {v!r}") from None


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in ("1", "true", "yes", "on")


_CHILD_TUNABLES = (
    "BATCH_SIZE", "STREAM_BATCH_SIZE", "MAX_SPECIES", "MAX_IMAGES_PER_SPECIES", "VAL_IMAGES_PER_SPECIES",
    "MAX_CACHE_IMAGES", "HEAD_EPOCHS", "HEAD_LR", "HEAD_BATCH", "HEAD_WEIGHT_DECAY", "HEAD_FEATURE_DROPOUT",
    "HEAD_PATIENCE", "DISK_CACHE", "DISK_CACHE_DIR", "AUTO_EXTRACT_ARCHIVES",
    "DATASET_NAME", "MODEL_OUTPUT", "ONNX_MODEL_PATH", "HF_TOKEN",
    "BANK_DIR", "BANK_DTYPE", "FEATURE_BANK_DIR",
)


def child_env(**extra) -> dict:
    env = dict(os.environ)
    for k in _CHILD_TUNABLES:
        if k in env:
            env[k] = _env(k)
    for k, v in _BLAS_ORIG.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    env.update(extra)
    return env


def _detect_cores() -> int:
    """CPU cores this container may actually use (cgroup quota first: the host's core count is misleading)."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:  # cgroup v2: "<quota> <period>" or "max <period>"
            quota, period = f.read().split()[:2]
        if quota != "max":
            return max(1, math.ceil(int(quota) / int(period)))
    except Exception:
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:  # cgroup v1
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        if quota > 0:
            return max(1, math.ceil(quota / period))
    except Exception:
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)


CORES = min(_detect_cores(), 16)


@dataclass
class Settings:
    api_key: str = field(default_factory=lambda: _env("API_KEY"))
    model_path: str = field(default_factory=lambda: _env("MODEL_OUTPUT", "models/pokemon_classifier.pt"))
    onnx_path: str = field(default_factory=lambda: _env("ONNX_MODEL_PATH"))
    auto_export_onnx: bool = field(default_factory=lambda: _env_bool("AUTO_EXPORT_ONNX", True))
    onnx_threads: int = field(default_factory=lambda: _env_int("ONNX_THREADS", CORES))
    infer_concurrency: int = field(default_factory=lambda: _env_int("INFER_CONCURRENCY", CORES * 16))  # doubled: higher parallelism
    max_pending: int = CORES * 256  # doubled: batch accumulation depth
    infer_chunk: int = 8  # doubled from 2: larger batch windows without blocking
    top_k: int = field(default_factory=lambda: max(1, _env_int("TOP_K", 5)))
    near_exact_sim: float = field(default_factory=lambda: _env_float("NEAR_EXACT_SIM", 0.95))
    learn_dup_sim: float = field(default_factory=lambda: _env_float("LEARN_DUP_SIM", 0.995))
    confidence_threshold: float = field(default_factory=lambda: _env_float("CONFIDENCE_THRESHOLD", 0.5))
    max_upload_mb: float = 10.0
    max_batch: int = 32
    export_timeout_s: int = field(default_factory=lambda: _env_int("EXPORT_TIMEOUT_S", 900))
    trainer_cmd: List[str] = field(default_factory=lambda: [sys.executable, "-u", "run_training.py"])
    train_log: str = field(default_factory=lambda: _env("TRAIN_LOG", "train.log"))

    def __post_init__(self):
        if not self.onnx_path:
            self.onnx_path = os.path.splitext(self.model_path)[0] + ".onnx"

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


# ============================================================ engine
class Engine:
    """Owns the ONNX extractor and the feature bank, and swaps them together on reload."""

    def __init__(self, settings: Settings, bank: Optional[DiskFeatureBank] = None):
        self.s = settings
        self.bank: Optional[DiskFeatureBank] = bank
        self.extractor = None
        self.ready = False
        self.error: Optional[str] = "starting"
        self.last_reload_error: Optional[str] = None
        self.backend = "ONNX Runtime"
        self.model_id: Optional[str] = None
        self.loaded_at: Optional[float] = None
        self.started_at = time.time()
        self._load_lock = threading.Lock()
        self._fp_lock = threading.Lock()
        self.m_embed: Deque[float] = deque(maxlen=500)
        self.m_match: Deque[float] = deque(maxlen=500)
        self.n_predictions = 0
        self.n_bad_images = 0

    def load(self) -> dict:
        """(Re)load model + bank. On failure, a previously working state keeps serving."""
        with self._load_lock:
            try:
                new_ex, model_id, note = self._load_extractor()
                if self.bank is None:
                    self.bank = DiskFeatureBank(base_dir=BANK_DIR)

                def validate(matrix: np.ndarray):
                    if matrix.shape[0] and matrix.shape[1] != new_ex.dim:
                        raise RuntimeError(
                            f"model outputs {new_ex.dim}-d vectors but the feature bank holds "
                            f"{matrix.shape[1]}-d vectors - model and bank are out of sync")

                def swap():
                    self.extractor = new_ex
                    self.model_id = model_id

                n_vec, n_species = self.bank.reload(validate=validate, on_swap=swap)
                self.ready, self.error, self.last_reload_error = True, None, None
                self.loaded_at = time.time()
                if n_vec == 0:
                    note = (note + "; " if note else "") + "feature bank is empty - train first"
                log.info("loaded model %s, bank: %d vectors / %d species%s", model_id, n_vec, n_species,
                         f" ({note})" if note else "")
                return {"vectors": n_vec, "species": n_species, "model": model_id, "note": note or None}
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                log.error("load failed: %s", msg)
                self.last_reload_error = msg
                if not self.ready:
                    self.error = msg
                raise

    def _load_extractor(self):
        from onnx_backend import OnnxExtractor, file_sha1, onnx_matches_source

        model_path, onnx_path = self.s.model_path, self.s.onnx_path
        pt_exists, onnx_exists = os.path.exists(model_path), os.path.exists(onnx_path)
        note = ""

        if not pt_exists and not onnx_exists:
            raise FileNotFoundError(
                f"no model found ({model_path} / {onnx_path}). Train one or copy the files into place.")

        need_export = False
        if pt_exists:
            state = onnx_matches_source(onnx_path, model_path) if onnx_exists else None
            if state is not True:
                if self.s.auto_export_onnx:
                    need_export = True
                elif onnx_exists and state is None:
                    note = "ONNX has no sidecar, can't verify it matches the .pt"
                else:
                    raise RuntimeError("the .onnx is missing or was exported from a different .pt")
        else:
            note = "no .pt on disk - serving the .onnx unverified"

        if need_export:
            self._export_onnx()

        model_id = None
        if pt_exists:
            try:
                model_id = file_sha1(model_path)[:12]
            except OSError:
                pass
        model_id = model_id or f"onnx-{os.path.getsize(onnx_path)}"
        extractor = OnnxExtractor(onnx_path, threads=self.s.onnx_threads)
        return extractor, model_id, note

    def _export_onnx(self):
        log.info("exporting ONNX from %s (one-time, needs torch) ...", self.s.model_path)
        cmd = [sys.executable, "export_onnx.py", "--model", self.s.model_path, "--out", self.s.onnx_path]
        env = child_env(DEBUG="1")
        try:
            r = subprocess.run(cmd, cwd=APP_DIR, env=env, capture_output=True, text=True,
                               timeout=self.s.export_timeout_s)
        except subprocess.TimeoutExpired:
            raise RuntimeError("ONNX export timed out")
        if r.returncode != 0:
            tail = "\n".join((r.stdout + "\n" + r.stderr).strip().splitlines()[-8:])
            raise RuntimeError(f"ONNX export failed:\n{tail}")

    def snapshot(self):
        """(extractor, species_list, matrix, bank_version) - consistent with each other."""
        bank = self.bank
        with bank.lock:
            return self.extractor, bank._species, bank._matrix, bank.version

    # Below this many images a plain matrix-vector product is as fast as a matrix-matrix one;
    # from here on one BLAS matmul streams the (huge) bank from RAM once for the whole group
    # instead of once per image (measured at 7 lakh x 256: 53 ms/img alone, ~28 at 4, ~15 at 8).
    _GEMM_MIN = 2  # lowered from 3: batch smaller images too for better cache reuse
    _GEMM_MAX = 32  # increased from 16: larger batches to amortize bank streaming cost

    def _finish_match(self, sims: np.ndarray, species_list, k: int):
        """Top-k of one similarity row + majority vote (the rules the old _match used)."""
        order = np.argpartition(-sims, min(k - 1, len(sims) - 1))[:k]
        order = order[np.argsort(-sims[order])]
        neighbors = [(species_list[i], float(sims[i])) for i in order]
        votes = {}
        for sp, _ in neighbors:
            votes[sp] = votes.get(sp, 0) + 1
        winner = max(votes, key=lambda sp: (votes[sp], max(s for n, s in neighbors if n == sp)))
        score = max(s for n, s in neighbors if n == winner)
        top_species, top_sim = neighbors[0]
        if top_species != winner and top_sim >= self.s.near_exact_sim:
            winner, score = top_species, top_sim
        return winner, score, neighbors

    def _match(self, vec: np.ndarray, species_list, matrix):
        """Cosine top-k (vectors are L2-normalised so it's a dot product) + majority vote."""
        k = min(self.s.top_k, matrix.shape[0])
        return self._finish_match(matrix @ vec, species_list, k)

    def _match_many(self, vecs: List[np.ndarray], species_list, matrix):
        k = min(self.s.top_k, matrix.shape[0])
        if len(vecs) < self._GEMM_MIN:
            return [self._finish_match(matrix @ v, species_list, k) for v in vecs]
        out = []
        for a in range(0, len(vecs), self._GEMM_MAX):
            V = np.ascontiguousarray(np.stack(vecs[a:a + self._GEMM_MAX]), dtype=np.float32)
            S = np.ascontiguousarray((matrix @ V.T).T)  # (b, N): one pass over the bank for b images
            out.extend(self._finish_match(S[j], species_list, k) for j in range(S.shape[0]))
        return out

    @staticmethod
    def _decode_image(b: bytes) -> Optional[Image.Image]:
        """Decode and convert one image to RGB. Thread-safe."""
        try:
            im = Image.open(io.BytesIO(b))
            return im.convert("RGB")
        except Exception:
            return None

    def embed_many(self, images: List[bytes], snap=None):
        """Decode + embed all images in ONE model call (no bank access).
        Returns (vecs, embed_ms): vecs[i] is None for an unreadable image; embed_ms[i] is per-image ms."""
        extractor = (snap or self.snapshot())[0]
        if extractor is None:
            raise RuntimeError("model not loaded")

        # Decode images in parallel (PIL is thread-safe for different Image objects)
        pil: List[Optional[Image.Image]] = []
        n_imgs = len(images)
        if n_imgs <= 1:
            pil = [self._decode_image(b) for b in images]
        else:
            n_workers = min(4, n_imgs)  # cap at 4 to avoid context thrashing on 2-core containers
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                pil = list(ex.map(self._decode_image, images))

        n_valid = sum(1 for im in pil if im is not None)
        t0 = time.perf_counter()
        arr = extractor.extract_batch(pil) if n_valid else np.zeros((len(pil), extractor.dim), np.float32)
        embed_ms = (time.perf_counter() - t0) * 1000 / max(1, n_valid)

        vecs: List[Optional[np.ndarray]] = []
        # Vectorize L2-norm check: compute all norms at once, then filter
        norms = np.linalg.norm(arr, axis=1)
        for i, (im, vec, norm) in enumerate(zip(pil, arr, norms)):
            if im is None or norm < 1e-6:
                self.n_bad_images += 1
                vecs.append(None)
            else:
                vecs.append(vec)
        return vecs, [embed_ms] * len(pil)

    def match_many(self, snap, vecs: List[Optional[np.ndarray]], embed_ms: List[float]) -> List[Optional[dict]]:
        """Match embedded images against the bank snapshot they were embedded for (one batched matmul)."""
        _, species_list, matrix, version = snap
        valid = [i for i, v in enumerate(vecs) if v is not None]
        out: List[Optional[dict]] = [None] * len(vecs)
        if not valid:
            return out
        if matrix.shape[0] == 0:
            raise EmptyBank()
        t1 = time.perf_counter()
        matches = self._match_many([vecs[i] for i in valid], species_list, matrix)
        match_ms = (time.perf_counter() - t1) * 1000 / len(valid)
        for i, (species, score, neighbors) in zip(valid, matches):
            self.m_embed.append(embed_ms[i])
            self.m_match.append(match_ms)
            self.n_predictions += 1
            out[i] = {"species": species, "score": score, "neighbors": neighbors, "vec": vecs[i],
                      "embed_ms": embed_ms[i], "match_ms": match_ms, "bank_version": version}
        return out

    def identify(self, images: List[bytes]) -> List[Optional[dict]]:
        """Decode + embed all images in ONE model call, then match them together. None = unreadable image."""
        snap = self.snapshot()
        vecs, embed_ms = self.embed_many(images, snap)
        return self.match_many(snap, vecs, embed_ms)

    def embed_one(self, image: bytes) -> Optional[np.ndarray]:
        extractor = self.extractor
        if extractor is None:
            raise RuntimeError("model not loaded")
        try:
            im = Image.open(io.BytesIO(image)).convert("RGB")
        except Exception:
            return None
        vec = extractor.extract(im)
        return vec if float(np.linalg.norm(vec)) >= 1e-6 else None

    @staticmethod
    def _pcts(values) -> dict:
        v = list(values)
        if not v:
            return {"n": 0, "p50": None, "p95": None}
        return {"n": len(v), "p50": round(float(np.percentile(v, 50)), 2),
                "p95": round(float(np.percentile(v, 95)), 2)}

    _FP_BLOCK = 4096

    @staticmethod
    def _fp_block(h, species, matrix, a: int, b: int):
        h.update(("\n".join(species[a:b]) + "\n").encode())
        h.update(memoryview(matrix[a:b]))   # no bytes copy of the block

    def _bank_fingerprint(self, species, matrix, version, generation: int = 0) -> str:
        """Content hash of model + feature bank, so a client can tell if its saved results are still valid
        (unlike bank_version, it survives server restarts). Learning only appends rows, so the hash of all
        full 4096-row blocks is kept and only the new rows are hashed: cheap after each learn."""
        with self._fp_lock:
            cached = getattr(self, "_fp_cache", None)
            if cached and cached[0] == version:
                return cached[1]
            n, B = int(matrix.shape[0]), self._FP_BLOCK
            st = getattr(self, "_fp_state", None)   # (generation, rows_hashed, running sha1)
            if st is None or st[0] != generation or st[1] > n:
                h = hashlib.sha1()
                h.update(str(self.model_id).encode())
                done = 0
            else:
                _, done, h = st
            full = (n // B) * B
            for a in range(done, full, B):
                self._fp_block(h, species, matrix, a, a + B)
            self._fp_state = (generation, full, h)
            t = h.copy()
            if full < n:
                self._fp_block(t, species, matrix, full, n)
            fp = t.hexdigest()[:16]
            self._fp_cache = (version, fp)
            return fp

    def health(self) -> dict:
        species, matrix, version, generation = ([], np.zeros((0, 0)), 0, 0)
        if self.bank is not None:
            species, matrix, version, generation = self.bank.snapshot_gen()
        return {
            "status": "ok" if self.ready else "not_ready",
            "ready": self.ready,
            "error": None if self.ready else self.error,
            "last_reload_error": self.last_reload_error,
            "backend": self.backend,
            "model": self.model_id,
            "vectors": int(matrix.shape[0]),
            "species": self.bank.species_count if self.bank is not None else 0,
            "bank_version": version,
            "bank_fingerprint": self._bank_fingerprint(species, matrix, version, generation) if matrix.shape[0] else None,
            "bank_backend": self.bank.backend if self.bank else None,
            "bank_dir": BANK_DIR,
            "uptime_s": int(time.time() - self.started_at),
        }


class EmptyBank(Exception):
    pass


# ============================================================ trainer (subprocess)
class Trainer:
    """Runs run_training.py in a child process so its heavy torch memory never lives in the API process."""

    def __init__(self, settings: Settings, engine: Engine):
        self.s = settings
        self.engine = engine
        self.proc: Optional[subprocess.Popen] = None
        self.state = "idle"  # idle | running | reloading | succeeded | failed | stopped
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.returncode: Optional[int] = None
        self.reload_result: Optional[dict] = None
        self.reload_error: Optional[str] = None
        self._lock = threading.Lock()
        self._stop_requested = False

    @property
    def log_path(self) -> Path:
        p = Path(self.s.train_log)
        return p if p.is_absolute() else APP_DIR / p

    def warnings(self) -> List[str]:
        w = []
        if not (APP_DIR / "Extra pokemons.zip").exists() and not (APP_DIR / "Extra pokemons").is_dir():
            w.append("No 'Extra pokemons.zip' or 'Extra pokemons/' found.")
        return w

    def start(self) -> dict:
        with self._lock:
            if self.state in ("running", "reloading"):
                raise RuntimeError(f"a training run is already {self.state}")
            env = child_env(DEBUG="1", PYTHONUNBUFFERED="1")
            self.proc = subprocess.Popen(self.s.trainer_cmd, cwd=APP_DIR, env=env,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         bufsize=1, text=True)
            self.state, self._stop_requested = "running", False
            self.started_at, self.finished_at, self.returncode = time.time(), None, None
            self.reload_result = self.reload_error = None
            threading.Thread(target=self._pump_output, args=(self.proc,), daemon=True).start()
            threading.Thread(target=self._watch, args=(self.proc,), daemon=True).start()
            log.info("training started (pid %d)", self.proc.pid)
            return {"pid": self.proc.pid, "warnings": self.warnings()}

    def _pump_output(self, proc: subprocess.Popen):
        """Read the trainer's stdout line by line and write it to BOTH train.log and this
        server's own stdout, so you can watch progress live in the terminal server.py runs
        in, not just via GET /admin/train/status."""
        try:
            with open(self.log_path, "w", encoding="utf-8", errors="replace") as logf:
                for line in proc.stdout:
                    sys.stdout.write(f"[train] {line}")
                    sys.stdout.flush()
                    logf.write(line)
                    logf.flush()
        except Exception as e:
            log.error("training output pump crashed: %s", e)
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass

    def _watch(self, proc: subprocess.Popen):
        rc = proc.wait()
        with self._lock:
            self.returncode = rc
            if self._stop_requested:
                self.state, self.finished_at = "stopped", time.time()
                return
            if rc != 0:
                self.state, self.finished_at = "failed", time.time()
                log.error("training failed (exit %s), see %s", rc, self.log_path)
                return
            self.state = "reloading"
        try:
            self.reload_result = self.engine.load()
            self.reload_error = None
        except Exception as e:
            self.reload_error = f"{type(e).__name__}: {e}"
        with self._lock:
            self.state = "succeeded" if self.reload_error is None else "failed"
            self.finished_at = time.time()
        log.info("training run finished: %s", self.state)

    def stop(self) -> bool:
        with self._lock:
            if self.state != "running" or self.proc is None:
                return False
            self._stop_requested = True
            self.proc.terminate()
        return True

    def status(self, tail_lines: int = 40) -> dict:
        return {
            "state": self.state,
            "pid": self.proc.pid if self.proc else None,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "returncode": self.returncode,
            "reload": self.reload_result,
            "reload_error": self.reload_error,
            "log_tail": self._tail(tail_lines),
        }

    def _tail(self, n: int) -> List[str]:
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 16384))
                text = f.read().decode("utf-8", "replace")
        except OSError:
            return []
        parts = [p.strip() for p in text.replace("\r", "\n").split("\n")]
        return [p for p in parts if p][-n:]


# ============================================================ request models
class ForgetBody(BaseModel):
    species: Optional[str] = None
    last: bool = False


class UrlsBody(BaseModel):
    urls: List[str]


# ============================================================ server-side image fetching
# The bot can send image URLs instead of image bytes; the server downloads them itself (much shorter
# trip than bot -> download -> upload). Only these hosts are fetched, so the endpoint can't be used to
# make this server request arbitrary addresses. Add more with EXTRA_IMAGE_HOSTS=host1,host2 in .env.
_IMAGE_HOST_SUFFIXES = (".discordapp.com", ".discordapp.net", ".discord.com", ".poketwo.net")
_EXTRA_IMAGE_HOSTS = tuple(h.strip().lower() for h in _env("EXTRA_IMAGE_HOSTS").split(",") if h.strip())
_FETCH_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AI_Model/1.0)", "Accept": "image/*,*/*;q=0.8"}
_FETCH_TIMEOUT_S = 8.0


class FetchError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _check_image_url(url: str) -> None:
    try:
        parts = urllib.parse.urlsplit(url)
        host = (parts.hostname or "").lower()
    except ValueError:
        raise FetchError("bad_url", "not a valid URL")
    if parts.scheme not in ("https", "http") or not host or parts.username or parts.password:
        raise FetchError("bad_url", "only plain http(s) URLs are accepted")
    ok = host in _EXTRA_IMAGE_HOSTS or any(host == suf[1:] or host.endswith(suf) for suf in _IMAGE_HOST_SUFFIXES)
    if not ok:
        raise FetchError("host_not_allowed", f"host not allowed: {host}")


_fetch_session = None


async def _get_fetch_session():
    global _fetch_session
    if _fetch_session is None or _fetch_session.closed:
        _fetch_session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ttl_dns_cache=300, keepalive_timeout=60, limit=64),
            headers=_FETCH_HEADERS)
    return _fetch_session


async def _close_fetch_session():
    global _fetch_session
    if _fetch_session is not None and not _fetch_session.closed:
        await _fetch_session.close()
    _fetch_session = None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _fetch_sync(url: str, limit: int) -> bytes:
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, headers=_FETCH_HEADERS)
    try:
        with opener.open(req, timeout=_FETCH_TIMEOUT_S) as r:
            data = r.read(limit + 1)
    except urllib.error.HTTPError as e:
        raise FetchError("fetch_failed", f"HTTP {e.code}")
    except Exception as e:
        raise FetchError("fetch_failed", f"{type(e).__name__}: {e}")
    if len(data) > limit:
        raise FetchError("too_large", "image too large")
    return data


async def fetch_image(url: str, limit: int) -> bytes:
    """Download one allowed image URL (no redirects, size-capped). Raises FetchError."""
    _check_image_url(url)
    if aiohttp is None:
        return await run_in_threadpool(_fetch_sync, url, limit)
    try:
        sess = await _get_fetch_session()
        timeout = aiohttp.ClientTimeout(total=_FETCH_TIMEOUT_S, connect=4)
        async with sess.get(url, allow_redirects=False, timeout=timeout) as r:
            if r.status != 200:
                raise FetchError("fetch_failed", f"HTTP {r.status}")
            if r.content_length and r.content_length > limit:
                raise FetchError("too_large", "image too large")
            buf = bytearray()
            async for chunk in r.content.iter_chunked(65536):
                buf.extend(chunk)
                if len(buf) > limit:
                    raise FetchError("too_large", "image too large")
            return bytes(buf)
    except FetchError:
        raise
    except Exception as e:
        raise FetchError("fetch_failed", f"{type(e).__name__}: {e}")


# ============================================================ app factory
def create_app(settings: Optional[Settings] = None, bank: Optional[DiskFeatureBank] = None) -> FastAPI:
    s = settings or Settings()
    engine = Engine(s, bank=bank)
    trainer = Trainer(s, engine)
    infer_sem = asyncio.Semaphore(s.infer_concurrency)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info(f"CPU cores={CORES}, onnx_threads={s.onnx_threads}, infer_slots={s.infer_concurrency}, "
                 f"max_pending={s.max_pending}, bank_dir={BANK_DIR}")
        if not s.api_key:
            log.warning("API_KEY is not set: the API is OPEN to anyone who can reach it.")

        async def _initial_load():
            try:
                await run_in_threadpool(engine.load)
            except Exception:
                pass  # already logged + recorded in engine.error; /health explains

        task = asyncio.create_task(_initial_load())
        yield
        task.cancel()
        await _close_fetch_session()
        trainer.stop()
        if engine.bank is not None:
            engine.bank.close()

    app = FastAPI(title="AI_Model", version="1.0.0", lifespan=lifespan)
    app.state.engine, app.state.trainer, app.state.settings = engine, trainer, s

    # ------------------------------------------------------ auth + helpers
    def _supplied_key(x_api_key: Optional[str], authorization: Optional[str]) -> str:
        if x_api_key:
            return x_api_key
        if authorization and authorization.lower().startswith("bearer "):
            return authorization[7:].strip()
        return ""

    async def require_key(x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
        if not s.api_key:
            return
        supplied = _supplied_key(x_api_key, authorization)
        if not secrets.compare_digest(supplied.encode(), s.api_key.encode()):
            raise HTTPException(401, "invalid or missing API key (send X-API-Key or Authorization: Bearer)")

    async def require_admin(x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
        if not s.api_key:
            raise HTTPException(403, "admin endpoints are disabled until API_KEY is set")
        await require_key(x_api_key, authorization)

    def require_ready():
        if not engine.ready:
            raise HTTPException(503, engine.error or "not ready")

    async def read_upload(up: UploadFile) -> bytes:
        data = await up.read(s.max_upload_bytes + 1)
        if len(data) > s.max_upload_bytes:
            raise HTTPException(413, f"image larger than {s.max_upload_mb:g} MB")
        if not data:
            raise HTTPException(400, "empty file")
        return data

    pending = {"n": 0}

    @asynccontextmanager
    async def admitted(n: int):
        """Load shedding: refuse new work instead of queueing without limit (RAM / latency blow-up)."""
        if pending["n"] + n > s.max_pending:
            raise HTTPException(503, "busy - too many images queued, retry shortly")
        pending["n"] += n
        try:
            yield
        finally:
            pending["n"] -= n

    async def infer(fn, *args):
        async with infer_sem:
            return await run_in_threadpool(fn, *args)

    async def identify_chunks(chunks: List[List[bytes]]) -> List[Optional[dict]]:
        """Embed the chunks in parallel (one per free inference slot), then match ALL images in one
        batched pass over the bank. One snapshot keeps model and bank consistent across a reload."""
        snap = engine.snapshot()
        parts = await asyncio.gather(*(infer(engine.embed_many, c, snap) for c in chunks))
        vecs = [v for p in parts for v in p[0]]
        ms = [m for p in parts for m in p[1]]
        return await infer(engine.match_many, snap, vecs, ms)

    def public(result: dict, threshold: float, include_embedding: bool) -> dict:
        body = {
            "species": result["species"],
            "score": round(result["score"], 6),
            "confident": result["score"] >= threshold,
            "threshold": threshold,
            "neighbors": [{"species": sp, "score": round(sim, 6)} for sp, sim in result["neighbors"]],
            "bank_version": result["bank_version"],
            "timing_ms": {"embed": round(result["embed_ms"], 2), "match": round(result["match_ms"], 2)},
        }
        if include_embedding:
            body["embedding"] = [round(float(x), 6) for x in result["vec"]]
        return body

    # ------------------------------------------------------ routes
    @app.get("/health")
    async def health():
        body = await run_in_threadpool(engine.health)   # may hash new bank rows: keep it off the event loop
        return JSONResponse(body, status_code=200 if engine.ready else 503)

    @app.post("/v1/predict", dependencies=[Depends(require_key)])
    async def predict(file: UploadFile = File(...),
                      threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
                      include_embedding: bool = Query(False)):
        require_ready()
        data = await read_upload(file)
        try:
            async with admitted(1):
                results = await infer(engine.identify, [data])
        except EmptyBank:
            raise HTTPException(503, "the feature bank is empty - train first")
        if results[0] is None:
            raise HTTPException(422, "could not read that image")
        th = s.confidence_threshold if threshold is None else threshold
        return public(results[0], th, include_embedding)

    @app.post("/v1/predict/batch", dependencies=[Depends(require_key)])
    async def predict_batch(files: List[UploadFile] = File(...),
                            threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
                            include_embedding: bool = Query(False)):
        require_ready()
        if len(files) > s.max_batch:
            raise HTTPException(413, f"at most {s.max_batch} images per batch")
        blobs = list(await asyncio.gather(*(read_upload(f) for f in files)))
        # Fan the batch out over all inference slots instead of running it on one thread:
        # chunks of INFER_CHUNK images run in parallel, results stay in input order.
        chunks = [blobs[i:i + s.infer_chunk] for i in range(0, len(blobs), s.infer_chunk)]
        try:
            async with admitted(len(blobs)):
                results = await identify_chunks(chunks)
        except EmptyBank:
            raise HTTPException(503, "the feature bank is empty - train first")
        th = s.confidence_threshold if threshold is None else threshold
        return {"results": [
            {"ok": False, "error": "could not read that image"} if r is None
            else {"ok": True, **public(r, th, include_embedding)}
            for r in results
        ]}

    @app.post("/v1/predict/urls", dependencies=[Depends(require_key)])
    async def predict_urls(body: UrlsBody,
                           threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
                           include_embedding: bool = Query(False)):
        """Same as /v1/predict/batch, but the server downloads the images from the given URLs."""
        require_ready()
        urls = body.urls
        if not urls:
            raise HTTPException(422, "urls must not be empty")
        if len(urls) > s.max_batch:
            raise HTTPException(413, f"at most {s.max_batch} urls per request")
        th = s.confidence_threshold if threshold is None else threshold

        async def fetch_one(u: str):
            t = time.perf_counter()
            try:
                data = await fetch_image(u, s.max_upload_bytes)
                return data, None, (time.perf_counter() - t) * 1000
            except FetchError as e:
                return None, e, (time.perf_counter() - t) * 1000

        out: List[Optional[dict]] = [None] * len(urls)
        async with admitted(len(urls)):
            fetched = await asyncio.gather(*(fetch_one(u) for u in urls))
            good = []  # (index, bytes, fetch_ms)
            for i, (data, err, ms) in enumerate(fetched):
                if err is not None:
                    out[i] = {"ok": False, "code": err.code, "error": err.message}
                else:
                    good.append((i, data, ms))
            if good:
                blobs = [g[1] for g in good]
                chunks = [blobs[i:i + s.infer_chunk] for i in range(0, len(blobs), s.infer_chunk)]
                try:
                    results = await identify_chunks(chunks)
                except EmptyBank:
                    raise HTTPException(503, "the feature bank is empty - train first")
                for (i, data, ms), r in zip(good, results):
                    if r is None:
                        out[i] = {"ok": False, "code": "unreadable", "error": "could not read that image"}
                    else:
                        out[i] = {"ok": True, **public(r, th, include_embedding),
                                  "sha1": hashlib.sha1(data).hexdigest(), "fetch_ms": round(ms, 1)}
        return {"results": out}

    @app.post("/v1/learn", dependencies=[Depends(require_key)])
    async def learn(species: str = Form(...),
                    allow_new: bool = Form(False),
                    file: Optional[UploadFile] = File(None),
                    embedding: Optional[str] = Form(None)):
        require_ready()
        name = species.strip()
        if not name or len(name) > 64 or not species_key(name):
            raise HTTPException(422, "species must be a non-empty name (max 64 chars)")
        if (file is None) == (embedding is None):
            raise HTTPException(422, "send exactly one of: file (image) or embedding (JSON array)")

        if file is not None:
            data = await read_upload(file)
            vec = await infer(engine.embed_one, data)
            if vec is None:
                return {"status": "bad_image", "species": None, "examples": None, "suggestions": []}
        else:
            try:
                vec = np.asarray(json.loads(embedding), dtype=np.float32).flatten()
            except Exception:
                raise HTTPException(422, "embedding must be a JSON array of numbers")
            if engine.extractor is not None and vec.shape[0] != engine.extractor.dim:
                raise HTTPException(422, f"embedding must have {engine.extractor.dim} values")

        status, sp, extra = await run_in_threadpool(engine.bank.learn, name, vec, allow_new, s.learn_dup_sim)
        return {
            "status": status,
            "species": sp,
            "examples": extra if status == "learned" else None,
            "suggestions": extra if status == "unknown" else [],
            "bank_version": engine.bank.version,
        }

    @app.post("/v1/learn/batch", dependencies=[Depends(require_key)])
    async def learn_batch(
        species: List[str] = Form(...),
        files: List[UploadFile] = File(...),
        allow_new: bool = Form(True),
    ):
        """
        Teach many examples in one request. Used by the bot's backfill mode.
        Each (species[i], files[i]) pair becomes one learned example.

        Returns a list of per-item results in the same order:
            {"status": "learned"|"duplicate"|"unknown"|"bad_image", "species": str, "examples": int|None}
        """
        require_ready()
        if len(species) != len(files):
            raise HTTPException(422, f"species ({len(species)}) and files ({len(files)}) must be the same length")
        if len(files) == 0:
            raise HTTPException(422, "no items in batch")
        if len(files) > 64:
            raise HTTPException(413, "at most 64 items per batch")

        # Read all uploads in parallel, then decode + embed in ONE ONNX call
        blobs = list(await asyncio.gather(*(read_upload(f) for f in files)))
        extractor = engine.extractor
        if extractor is None:
            raise HTTPException(503, "model not loaded")

        # Decode PIL images here so the whole batch goes through one extract_batch.
        pil: List[Optional[Image.Image]] = []
        for b in blobs:
            try:
                pil.append(Image.open(io.BytesIO(b)).convert("RGB"))
            except Exception:
                pil.append(None)

        # extract_batch returns zeros for unreadable images, so filter those out up-front.
        valid_idx = [i for i, im in enumerate(pil) if im is not None]
        vectors: List[Optional[np.ndarray]] = [None] * len(pil)
        if valid_idx:
            async with infer_sem:
                vecs = await run_in_threadpool(extractor.extract_batch, [pil[i] for i in valid_idx])
            for out_i, in_i in enumerate(valid_idx):
                v = vecs[out_i]
                if v is not None and float(np.linalg.norm(v)) >= 1e-6:
                    vectors[in_i] = v

        # Now apply the same learn() logic as the single endpoint, one item at a time,
        # under the bank's own lock. Batched at the ONNX level, sequential at the file level.
        results = []
        for sp_name, vec in zip(species, vectors):
            name = sp_name.strip()
            if not name or len(name) > 64 or not species_key(name):
                results.append({"status": "bad_image", "species": None, "examples": None})
                continue
            if vec is None:
                results.append({"status": "bad_image", "species": None, "examples": None})
                continue
            try:
                status, sp, extra = await run_in_threadpool(
                    engine.bank.learn, name, vec, allow_new, s.learn_dup_sim
                )
            except Exception as e:
                log.warning("learn_batch: item %r failed: %s", name, e)
                status, sp, extra = "bad_image", None, None
            results.append({"status": status, "species": sp, "examples": extra})

        return {"results": results, "bank_version": engine.bank.version}

    @app.post("/v1/forget", dependencies=[Depends(require_key)])
    async def forget(body: ForgetBody):
        require_ready()
        if not body.species and not body.last:
            raise HTTPException(422, 'send {"species": "..."} or {"last": true}')
        target = None
        if body.species:
            species_list, _ = engine.bank.snapshot()
            target = engine.bank.find_species(species_list, body.species)
            if target is None:
                raise HTTPException(404, {"error": "unknown species", "suggestions": engine.bank.suggestions(body.species)})
        removed = await run_in_threadpool(engine.bank.delete_learned, target, body.last and not body.species)
        return {"removed": len(removed), "species": sorted(set(removed)), "bank_version": engine.bank.version}

    @app.get("/v1/stats", dependencies=[Depends(require_key)])
    async def stats():
        body = engine.health()
        if engine.ready:
            body["learned_vectors"] = await run_in_threadpool(engine.bank.count_learned)
        body["settings"] = {"top_k": s.top_k, "confidence_threshold": s.confidence_threshold,
                            "near_exact_sim": s.near_exact_sim, "learn_dup_sim": s.learn_dup_sim,
                            "onnx_threads": s.onnx_threads, "infer_concurrency": s.infer_concurrency,
                            "infer_chunk": s.infer_chunk, "cpu_cores": CORES, "pending_images": pending["n"],
                            "max_pending": s.max_pending, "bank_dir": BANK_DIR}
        body["counters"] = {"predictions": engine.n_predictions, "unreadable_images": engine.n_bad_images}
        body["latency_ms"] = {"embed_per_image": engine._pcts(engine.m_embed), "match": engine._pcts(engine.m_match)}
        return body

    # ------------------------------------------------------ admin
    @app.post("/admin/reload", dependencies=[Depends(require_admin)])
    async def reload_all():
        try:
            return await run_in_threadpool(engine.load)
        except Exception as e:
            raise HTTPException(500, f"{type(e).__name__}: {e}")

    @app.post("/admin/train", dependencies=[Depends(require_admin)], status_code=202)
    async def train_start():
        try:
            return await run_in_threadpool(trainer.start)
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    @app.get("/admin/train/status", dependencies=[Depends(require_admin)])
    async def train_status(lines: int = Query(40, ge=0, le=500)):
        return trainer.status(lines)

    @app.post("/admin/train/stop", dependencies=[Depends(require_admin)])
    async def train_stop():
        return {"stopped": trainer.stop()}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    # uvloop (libuv-backed event loop) cuts the per-request Python overhead of accepting
    # connections and parsing HTTP - a real win since everything except the ONNX call itself
    # runs on this one event loop. Falls back to the default loop if unavailable (e.g. Windows).
    loop_impl = "auto"
    try:
        import uvloop  # noqa: F401
    except ImportError:
        loop_impl = "asyncio"
        log.info("uvloop not installed - using the default asyncio loop (pip install uvloop for a bit more headroom)")
    access_log = _env_bool("UVICORN_ACCESS_LOG", False)
    uvicorn.run(app, host=_env("HOST", "0.0.0.0"), port=_env_int("PORT", 8000), log_level="info",
                timeout_keep_alive=75, loop=loop_impl, access_log=access_log)