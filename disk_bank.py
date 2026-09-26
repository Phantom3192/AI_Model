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

Key performance property vs the Postgres backend: the whole bank lives in ONE
preallocated fp32 buffer with spare rows, and learn() writes the new vector into the
next free row. Nothing is copied or rebuilt per learn (it used to re-concatenate the
whole matrix, ~0.2 s and +700 MB of temporary RAM at 7 lakh rows), and species lookups
use a dict index instead of scanning every row.
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
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("ai_model.bank.disk")

LEARNED_MARK = "__learned_"
LEARNED_FILE = "learned.jsonl"
LEARNED_CAPACITY_START = 1024   # first preallocation for the learned overlay (load time only)
LEARNED_CAPACITY_GROWTH = 2     # double when full
SPARE_MIN_ROWS = 8192           # free rows kept at the end of the bank buffer for learn()
COPY_CHUNK_ROWS = 65536         # rows copied per step when filling the buffer (no big temporaries)
# fsync after every learned row survives a power cut; turn it off (LEARN_FSYNC=0) for faster bulk learning.
LEARN_FSYNC = os.getenv("LEARN_FSYNC", "1").strip().lower() not in ("0", "false", "no", "off")


def _spare_rows(n: int) -> int:
    return max(SPARE_MIN_ROWS, n // 20)


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
        # Bumped whenever rows change other than by appending (reload / forget). Lets callers keep
        # incremental state (e.g. the fingerprint) across pure-append learns.
        self.generation = 0
        self.backend = "disk"

        # Active bank: rows [0, _n) of _buf. _matrix is the view server code searches; _species has
        # one entry per row. learn() only ever appends, so snapshots taken earlier stay valid.
        self._species: List[str] = []
        self._matrix = np.zeros((0, 256), dtype=np.float32)
        self._buf = self._matrix
        self._n = 0
        self._sp_rows: Dict[str, List[int]] = {}      # species -> row indices
        self._key_to_species: Dict[str, str] = {}     # species_key(species) -> species
        self._lock = threading.RLock()
        self._write_lock = threading.RLock()

        # Learned overlay: only used while (re)loading from learned.jsonl; freed right after.
        self._learned_species: List[str] = []
        self._learned_matrix = np.zeros((LEARNED_CAPACITY_START, 0), dtype=np.float32)
        self._learned_n = 0
        self._learned_path = self.base_dir / LEARNED_FILE

        self._base_features_path = self.base_dir / "base_features.npy"
        self._base_species_path = self.base_dir / "base_species.npy"
        self._base_meta_path = self.base_dir / "base_meta.json"

        self._base_matrix: Optional[np.ndarray] = None   # memory-mapped, never fully in RAM
        self._base_species: List[str] = []
        self.dim_hint = dim

        self.reload()

    # ------------------------------------------------------------------ loading
    def reload(self, validate: Optional[Callable[[np.ndarray], None]] = None,
               on_swap: Optional[Callable[[], None]] = None) -> Tuple[int, int]:
        with self._write_lock:
            self._load_base()
            self._load_learned()
            staged = self._build_staged()
            self._release_overlay()

            with self._lock:
                if validate:
                    validate(staged["matrix"])
                self._install(staged)
                self.version += 1
                self.generation += 1
                if on_swap:
                    on_swap()
            return len(self._species), len(self._sp_rows)

    def _load_base(self):
        if not self._base_features_path.exists():
            log.warning("base_features.npy not found in %s - bank will start empty. "
                        "Run export_to_disk.py first.", self.base_dir)
            self._base_matrix = np.zeros((0, self.dim_hint or 256), dtype=np.float32)
            self._base_species = []
            return

        # mmap_mode="r": nothing is read yet. _build_staged copies it into the fp32 bank buffer in
        # chunks (casting fp16 -> fp32 on the way), so there is never a second full-size copy.
        mm = np.load(self._base_features_path, mmap_mode="r")
        if mm.ndim != 2:
            raise ValueError(f"base_features.npy has wrong shape: {mm.shape}")
        self._base_matrix = mm

        species = np.load(self._base_species_path, allow_pickle=False)
        if species.shape[0] != mm.shape[0]:
            raise ValueError(
                f"base_species.npy has {species.shape[0]} entries but "
                f"base_features.npy has {mm.shape[0]} rows"
            )
        interned: Dict[str, str] = {}   # one str object per species instead of one per row
        self._base_species = [interned.setdefault(str(s), str(s)) for s in species]

        meta = {}
        try:
            meta = json.loads(self._base_meta_path.read_text())
        except Exception:
            pass
        log.info("Loaded base bank: %d rows x %d dims (dtype %s) from %s",
                 mm.shape[0], mm.shape[1], meta.get("dtype", str(mm.dtype)), self.base_dir)

        if meta.get("dim") and int(meta["dim"]) != mm.shape[1]:
            log.warning("base_meta.json dim=%s disagrees with the file's %s",
                        meta["dim"], mm.shape[1])

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
        """In-memory append to the load-time overlay only (no disk write)."""
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

    def _release_overlay(self):
        """The learned rows now live in the bank buffer; free the load-time copy (keep the count)."""
        dim = int(self._learned_matrix.shape[1])
        self._learned_matrix = np.zeros((0, dim), dtype=np.float32)
        self._learned_species = []

    def _build_staged(self) -> dict:
        """Builds the bank (base + learned) into ONE preallocated fp32 buffer with spare rows."""
        base = self._base_matrix
        base_n = int(base.shape[0]) if base is not None else 0
        dim = int(base.shape[1]) if base_n else int(self.dim_hint or 256)
        learned_n = self._learned_n
        n = base_n + learned_n

        buf = np.empty((n + _spare_rows(n), dim), dtype=np.float32)
        for a in range(0, base_n, COPY_CHUNK_ROWS):
            b = min(a + COPY_CHUNK_ROWS, base_n)
            buf[a:b] = base[a:b]
        if learned_n:
            buf[base_n:n] = self._learned_matrix[:learned_n]

        species = list(self._base_species)
        species.extend(self._learned_species[:learned_n])

        sp_rows: Dict[str, List[int]] = {}
        key_to_species: Dict[str, str] = {}
        for i, sp in enumerate(species):
            rows = sp_rows.get(sp)
            if rows is None:
                sp_rows[sp] = [i]
                key_to_species.setdefault(species_key(sp), sp)
            else:
                rows.append(i)
        return {"buf": buf, "n": n, "species": species, "matrix": buf[:n],
                "sp_rows": sp_rows, "key_to_species": key_to_species}

    def _install(self, staged: dict):
        """Makes a staged bank the active one. Caller holds self._lock."""
        self._buf = staged["buf"]
        self._n = staged["n"]
        self._species = staged["species"]
        self._matrix = staged["matrix"]
        self._sp_rows = staged["sp_rows"]
        self._key_to_species = staged["key_to_species"]

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
    def find_species(self, species_list: List[str], name: str) -> Optional[str]:
        """Canonical species name for a user-typed name (case/punctuation-insensitive), or None."""
        key = species_key(name)
        if not key:
            return None
        return self._key_to_species.get(key)

    def suggestions(self, name: str, n: int = 3) -> List[str]:
        return difflib.get_close_matches(new_species_name(name), sorted(self._sp_rows), n=n, cutoff=0.6)

    @property
    def species_count(self) -> int:
        return len(self._sp_rows)

    def snapshot_gen(self):
        """(species, matrix, version, generation), all from the same moment."""
        with self._lock:
            return self._species, self._matrix, self.version, self.generation

    # ------------------------------------------------------------------ learning
    def _grow_buffer(self):
        """Out of spare rows: allocate a bigger buffer. Rare (every ~5% growth); in-flight searches
        keep using the old buffer until they finish."""
        n, dim = self._n, int(self._buf.shape[1])
        new = np.empty((n + _spare_rows(n), dim), dtype=np.float32)
        new[:n] = self._buf[:n]
        self._buf = new

    def learn(self, name: str, vec, allow_new: bool = False, dup_sim: float = 0.995):
        vec = np.asarray(vec, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6 or not np.isfinite(norm):
            return "bad_image", None, None
        vec = vec / norm

        with self._write_lock:
            if self._n and vec.shape[0] != self._buf.shape[1]:
                return "bad_image", None, None

            species = self.find_species(self._species, name)
            if species is None:
                if not allow_new:
                    return "unknown", None, self.suggestions(name)
                species = new_species_name(name)
            else:
                rows = self._sp_rows.get(species)
                if rows and float((self._buf[rows] @ vec).max()) >= dup_sim:
                    return "duplicate", species, None

            # Persist first, then update memory (so a crash mid-write doesn't leave
            # the in-memory bank ahead of the file)
            self._append_learned_to_disk(species, vec)

            with self._lock:
                if self._n >= self._buf.shape[0]:
                    self._grow_buffer()
                i = self._n
                self._buf[i] = vec                     # write the row BEFORE publishing it
                self._species.append(species)
                self._n = i + 1
                self._learned_n += 1
                self._sp_rows.setdefault(species, []).append(i)
                self._key_to_species.setdefault(species_key(species), species)
                self._matrix = self._buf[:self._n]
                self.version += 1
                count = len(self._sp_rows[species])
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
            if LEARN_FSYNC:
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

            drop = set(to_remove)
            keep = [r for i, r in enumerate(rows) if i not in drop]
            tmp = self._learned_path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for r in keep:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._learned_path)

            # Reload the learned overlay from disk so memory matches file
            self._load_learned()
            staged = self._build_staged()
            self._release_overlay()
            with self._lock:
                self._install(staged)
                self.version += 1
                self.generation += 1
            return removed_species

    def count_learned(self) -> int:
        return self._learned_n

    def close(self):
        pass  # nothing persistent to close; file handles are short-lived