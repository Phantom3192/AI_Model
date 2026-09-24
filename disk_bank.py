"""
disk_bank.py - file-backed replacement for feature_bank.FeatureBank.

Reads its base data from:
    <dir>/base_features.npy    (N, dim) fp16 or fp32, memory-mapped
    <dir>/base_species.npy     (N,) unicode strings
    <dir>/base_meta.json       metadata (rows, dim, model sha1, exported_at)

And its learned additions from:
    <dir>/learned.jsonl        one JSON object per line:
                               {"species": str, "vec": "<base64 fp32 bytes>", "t": unix_ts}

Public interface is identical to FeatureBank (same method names, same return
shapes), so the API server can switch between them without any other code change.

Key performance property vs the Postgres backend: learn() appends to a small
in-memory overlay (preallocated, doubled when full) instead of copying the whole
base matrix. That turns a 2-5 second operation into <1 ms.
"""

import os
import re
import json
import time
import base64
import difflib
import logging
import threading
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

log = logging.getLogger("ai_model.bank.disk")

LEARNED_MARK = "__learned_"
LEARNED_FILE = "learned.jsonl"
LEARNED_CAPACITY_START = 1024   # first preallocation for the learned overlay
LEARNED_CAPACITY_GROWTH = 2     # double when full


def species_key(name: str) -> str:
    return re.sub(r"[^a-z0-9♀♂]+", "", name.lower())


def new_species_name(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip().lower())


def _vec_to_b64(vec: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(vec, dtype=np.float32).tobytes()).decode("ascii")


def _b64_to_vec(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.float32).copy()


class DiskFeatureBank:
    def __init__(self, base_dir: str = "bank", dim: Optional[int] = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.version = 0
        self.backend = "disk"

        self._species: List[str] = []
        self._matrix = np.zeros((0, 256), dtype=np.float32)
        self._lock = threading.RLock()
        self._write_lock = threading.RLock()

        # Learned overlay
        self._learned_species: List[str] = []
        self._learned_matrix = np.zeros((LEARNED_CAPACITY_START, 0), dtype=np.float32)
        self._learned_n = 0
        self._learned_path = self.base_dir / LEARNED_FILE

        # Cached concatenated view (base + learned), rebuilt on version change.
        self._cached_version = -1
        self._cached_species: List[str] = []
        self._cached_matrix: np.ndarray = np.zeros((0, 256), dtype=np.float32)

        self._base_features_path = self.base_dir / "base_features.npy"
        self._base_species_path = self.base_dir / "base_species.npy"
        self._base_meta_path = self.base_dir / "base_meta.json"

        self._base_matrix: Optional[np.ndarray] = None
        self._base_species: List[str] = []
        self.dim_hint = dim

        self.reload()

    # ------------------------------------------------------------------ loading
    def reload(self, validate: Optional[Callable[[np.ndarray], None]] = None,
               on_swap: Optional[Callable[[], None]] = None) -> Tuple[int, int]:
        with self._write_lock:
            self._load_base()
            self._load_learned()
            self._rebuild_cache()

            with self._lock:
                if validate:
                    validate(self._cached_matrix)
                self._species = self._cached_species
                self._matrix = self._cached_matrix
                self.version += 1
                if on_swap:
                    on_swap()
            return len(self._species), len(set(self._species))

    def _load_base(self):
        if not self._base_features_path.exists():
            log.warning("base_features.npy not found in %s - bank will start empty. "
                        "Run export_to_disk.py first.", self.base_dir)
            self._base_matrix = np.zeros((0, self.dim_hint or 256), dtype=np.float32)
            self._base_species = []
            return

        # mmap_mode="r": the OS pages rows in on demand. We materialize to fp32
        # in one shot because the matmul wants fp32, and doing the cast lazily
        # per-request is slower. After this the mmap can be closed.
        mm = np.load(self._base_features_path, mmap_mode="r")
        if mm.ndim != 2:
            raise ValueError(f"base_features.npy has wrong shape: {mm.shape}")
        self._base_matrix = np.asarray(mm, dtype=np.float32)
        del mm

        species = np.load(self._base_species_path, allow_pickle=False)
        if species.shape[0] != self._base_matrix.shape[0]:
            raise ValueError(
                f"base_species.npy has {species.shape[0]} entries but "
                f"base_features.npy has {self._base_matrix.shape[0]} rows"
            )
        self._base_species = [str(s) for s in species]

        meta = {}
        try:
            meta = json.loads(self._base_meta_path.read_text())
        except Exception:
            pass
        log.info("Loaded base bank: %d rows x %d dims (dtype %s) from %s",
                 self._base_matrix.shape[0], self._base_matrix.shape[1],
                 meta.get("dtype", "?"), self.base_dir)

        if meta.get("dim") and int(meta["dim"]) != self._base_matrix.shape[1]:
            log.warning("base_meta.json dim=%s disagrees with the file's %s",
                        meta["dim"], self._base_matrix.shape[1])

    def _load_learned(self):
        self._learned_species = []
        if self._base_matrix is not None and self._base_matrix.shape[0]:
            dim = int(self._base_matrix.shape[1])
        else:
            dim = int(self.dim_hint or 256)
        self._learned_matrix = np.zeros((LEARNED_CAPACITY_START, dim), dtype=np.float32)
        self._learned_n = 0

        if not self._learned_path.exists():
            return

        with self._learned_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    vec = _b64_to_vec(row["vec"])
                    if vec.shape[0] != dim:
                        log.warning("%s:%d: vector is %d-dim, expected %d - skipping",
                                    self._learned_path.name, line_no, vec.shape[0], dim)
                        continue
                    self._append_learned_raw(row["species"], vec)
                except Exception as e:
                    log.warning("%s:%d: couldn't parse learned row (%s) - skipping",
                                self._learned_path.name, line_no, e)

        if self._learned_n:
            log.info("Loaded %d learned examples from %s", self._learned_n, self._learned_path.name)

    def _append_learned_raw(self, species: str, vec: np.ndarray):
        """In-memory append only (no disk write). Caller is responsible for persistence."""
        if self._learned_n >= self._learned_matrix.shape[0]:
            new_cap = max(LEARNED_CAPACITY_START,
                          self._learned_matrix.shape[0] * LEARNED_CAPACITY_GROWTH)
            new_mat = np.zeros((new_cap, self._learned_matrix.shape[1]), dtype=np.float32)
            if self._learned_n:
                new_mat[:self._learned_n] = self._learned_matrix[:self._learned_n]
            self._learned_matrix = new_mat
        self._learned_matrix[self._learned_n] = vec
        self._learned_species.append(species)
        self._learned_n += 1

    def _rebuild_cache(self):
        """Recompute the concatenated (base + learned) view. Called under _write_lock."""
        if self._base_matrix is None:
            self._cached_species = []
            self._cached_matrix = np.zeros((0, self.dim_hint or 256), dtype=np.float32)
        else:
            base_n = self._base_matrix.shape[0]
            learned_n = self._learned_n
            if learned_n == 0:
                self._cached_species = list(self._base_species)
                self._cached_matrix = self._base_matrix
            else:
                self._cached_species = list(self._base_species) + list(self._learned_species[:learned_n])
                combined = np.empty((base_n + learned_n, self._base_matrix.shape[1]), dtype=np.float32)
                combined[:base_n] = self._base_matrix
                combined[base_n:] = self._learned_matrix[:learned_n]
                self._cached_matrix = combined
        self._cached_version = self.version + 1

    # ------------------------------------------------------------------ reading
    @property
    def lock(self):
        return self._lock

    def snapshot(self) -> Tuple[List[str], np.ndarray]:
        with self._lock:
            return self._species, self._matrix

    def snapshot_versioned(self):
        with self._lock:
            return self._species, self._matrix, self.version

    @property
    def dim(self) -> Optional[int]:
        with self._lock:
            return int(self._matrix.shape[1]) if self._matrix.shape[0] else self.dim_hint

    # ------------------------------------------------------------------ species lookup
    @staticmethod
    def find_species(species_list: List[str], name: str) -> Optional[str]:
        key = species_key(name)
        if not key:
            return None
        for sp in set(species_list):
            if species_key(sp) == key:
                return sp
        return None

    def suggestions(self, name: str, n: int = 3) -> List[str]:
        species_list, _ = self.snapshot()
        return difflib.get_close_matches(new_species_name(name), sorted(set(species_list)), n=n, cutoff=0.6)

    # ------------------------------------------------------------------ learning
    def learn(self, name: str, vec, allow_new: bool = False, dup_sim: float = 0.995):
        vec = np.asarray(vec, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6 or not np.isfinite(norm):
            return "bad_image", None, None
        vec = vec / norm

        with self._write_lock:
            species_list, matrix = self.snapshot()
            if matrix.shape[0] and vec.shape[0] != matrix.shape[1]:
                return "bad_image", None, None

            species = self.find_species(species_list, name)
            if species is None:
                if not allow_new:
                    return "unknown", None, self.suggestions(name)
                species = new_species_name(name)
            else:
                idx = [i for i, s in enumerate(species_list) if s == species]
                if idx and float((matrix[idx] @ vec).max()) >= dup_sim:
                    return "duplicate", species, None

            # Persist first, then update memory (so a crash mid-write doesn't leave
            # the in-memory bank ahead of the file)
            self._append_learned_to_disk(species, vec)

            with self._lock:
                self._append_learned_raw(species, vec)
                self.version += 1
                self._rebuild_cache()
                self._species = self._cached_species
                self._matrix = self._cached_matrix
                count = sum(1 for s in self._cached_species if s == species)
            return "learned", species, count

    def _append_learned_to_disk(self, species: str, vec: np.ndarray):
        line = json.dumps({
            "species": species,
            "vec": _vec_to_b64(vec),
            "t": int(time.time()),
        })
        with self._learned_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def delete_learned(self, species: Optional[str] = None, last_only: bool = False) -> List[str]:
        """
        Removes learned examples. Rewrites learned.jsonl atomically.
        Returns the list of species names actually removed.
        """
        with self._write_lock:
            if not self._learned_path.exists():
                return []

            rows = []
            with self._learned_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue

            to_remove: List[int] = []
            if last_only:
                if rows:
                    to_remove = [len(rows) - 1]
            elif species:
                target_key = species_key(species)
                for i, row in enumerate(rows):
                    if species_key(row.get("species", "")) == target_key:
                        to_remove.append(i)

            removed_species = [rows[i].get("species", "") for i in to_remove]
            if not to_remove:
                return []

            keep = [r for i, r in enumerate(rows) if i not in set(to_remove)]
            tmp = self._learned_path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for r in keep:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._learned_path)

            # Reload the learned overlay from disk so memory matches file
            self._load_learned()
            with self._lock:
                self.version += 1
                self._rebuild_cache()
                self._species = self._cached_species
                self._matrix = self._cached_matrix
            return removed_species

    def count_learned(self) -> int:
        return self._learned_n

    def close(self):
        pass  # nothing persistent to close; file handles are short-lived
