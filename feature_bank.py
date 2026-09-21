"""
feature_bank.py - the embedding bank behind the AI_Model API.

Holds every stored (species, vector) pair as one NumPy matrix in memory (so a lookup is a single
matrix multiply) and keeps it in sync with the `pokemon_features` table that train_model.py writes.
It talks to the same database as the trainer: Turso if TURSO_URL is set, otherwise the local
SQLite file DB_PATH.

Deliberately has NO torch / train_model dependency, so the API server can run on onnxruntime alone.
"""

import os
import re
import json
import time
import uuid
import difflib
import sqlite3
import logging
import threading
from typing import Callable, List, Optional, Tuple

import numpy as np

log = logging.getLogger("ai_model.bank")

# Same marker main.py used, so vectors learned through the API and through the old bot commands
# are recognised (and can be forgotten) the same way.
LEARNED_MARK = "__learned_"


def species_key(name: str) -> str:
    """Loose comparison key so 'Mr. Mime', 'mr_mime' and 'mr._mime' all match."""
    return re.sub(r"[^a-z0-9♀♂]+", "", name.lower())


def new_species_name(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip().lower())


def _env(name: str, default: str = "") -> str:
    """os.getenv minus a trailing "  # comment" (some launchers load .env without stripping them)."""
    v = os.getenv(name)
    return default if v is None else re.split(r"\s+#", v, maxsplit=1)[0].strip()


class FeatureBank:
    def __init__(self, turso_url: Optional[str] = None, turso_token: Optional[str] = None,
                 db_path: Optional[str] = None):
        self.turso_url = turso_url if turso_url is not None else _env("TURSO_URL")
        self.turso_token = turso_token if turso_token is not None else _env("TURSO_AUTH_TOKEN")
        self.db_path = db_path or _env("DB_PATH", "pokemon.db")
        self.backend = "sqlite"

        self.version = 0                      # bumped whenever the in-memory bank changes
        self._species: List[str] = []
        self._matrix = np.zeros((0, 256), dtype=np.float32)
        self._lock = threading.RLock()        # guards the (species, matrix, version) triple
        self._write_lock = threading.RLock()  # serialises every DB access + bank mutation
        self._conn = self._connect()
        self._ensure_tables()

    # ------------------------------------------------------------------ connection
    def _connect(self):
        if self.turso_url:
            # No silent SQLite fallback here: a server quietly learning into a throwaway local file
            # while the real bank lives in Turso is worse than a clear startup error.
            import libsql
            conn = libsql.connect(self.turso_url, auth_token=self.turso_token)
            self.backend = "turso"
            return conn
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.backend = "sqlite"
        return conn

    def _reconnect(self):
        try:
            self._conn.close()
        except Exception:
            pass
        self._conn = self._connect()

    def _db(self, fn: Callable):
        """Run fn(conn); on failure reconnect once and retry (Turso connections can go stale)."""
        try:
            return fn(self._conn)
        except Exception as e:
            log.warning("database call failed (%s: %s) - reconnecting and retrying once", type(e).__name__, e)
            self._reconnect()
            return fn(self._conn)

    def _ensure_tables(self):
        def run(conn):
            cur = conn.cursor()
            try:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS pokemon_features (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        species TEXT NOT NULL,
                        variant_name TEXT NOT NULL,
                        feature_vector TEXT NOT NULL,
                        created_at INTEGER DEFAULT (strftime('%s', 'now')),
                        UNIQUE(species, variant_name)
                    )""")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS species_info (
                        species TEXT PRIMARY KEY,
                        count INTEGER DEFAULT 0,
                        last_updated INTEGER DEFAULT (strftime('%s', 'now'))
                    )""")
                conn.commit()
            finally:
                _close(cur)
        with self._write_lock:
            self._db(run)

    def close(self):
        with self._write_lock:
            try:
                self._conn.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ reading
    @property
    def lock(self):
        return self._lock

    def snapshot(self) -> Tuple[List[str], np.ndarray]:
        """(species_list, matrix). Both are replaced, never mutated, so the refs stay valid."""
        with self._lock:
            return self._species, self._matrix

    def snapshot_versioned(self):
        with self._lock:
            return self._species, self._matrix, self.version

    @property
    def dim(self) -> Optional[int]:
        with self._lock:
            return int(self._matrix.shape[1]) if self._matrix.shape[0] else None

    def reload(self, validate: Optional[Callable[[np.ndarray], None]] = None,
               on_swap: Optional[Callable[[], None]] = None) -> Tuple[int, int]:
        """
        Re-reads the whole bank from the database and installs it.
        `validate(matrix)` runs first and may raise to refuse the new bank (nothing changes then).
        `on_swap()` runs while the new bank is being installed, under the same lock predictions read
        it with, so the caller can swap the model at that exact instant and a request never sees
        new-model + old-bank. Returns (vectors, species).
        """
        with self._write_lock:
            species, matrix = self._load_rows()
            if validate:
                validate(matrix)
            with self._lock:
                self._species, self._matrix = species, matrix
                self.version += 1
                if on_swap:
                    on_swap()
        return len(species), len(set(species))

    def _load_rows(self) -> Tuple[List[str], np.ndarray]:
        def run(conn):
            cur = conn.cursor()
            try:
                cur.execute("SELECT species, feature_vector FROM pokemon_features ORDER BY id")
                return cur.fetchall()
            finally:
                _close(cur)
        rows = self._db(run)
        if not rows:
            return [], np.zeros((0, 256), dtype=np.float32)
        species = [r[0] for r in rows]
        matrix = np.asarray([json.loads(r[1]) for r in rows], dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("feature bank contains vectors of different lengths")
        return species, matrix

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
        """
        Adds one example to the DB and the in-memory bank.
        Returns (status, species, extra):
          ("learned", species, examples_for_species)
          ("duplicate", species, None)   bank already holds (almost) this exact image
          ("unknown", None, suggestions) species not in the bank and allow_new is False
          ("bad_image", None, None)      zero / wrong-sized vector
        """
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

            self._insert_learned(species, vec)  # DB first, so memory never gets ahead of it
            with self._lock:
                base = matrix if matrix.shape[0] else np.zeros((0, vec.shape[0]), dtype=np.float32)
                self._matrix = np.vstack([base, vec[None, :]])
                self._species = species_list + [species]
                self.version += 1
                count = sum(1 for s in self._species if s == species)
            return "learned", species, count

    def _insert_learned(self, species: str, vec: np.ndarray):
        variant = f"{species}{LEARNED_MARK}{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        payload = json.dumps(vec.tolist())

        def run(conn):
            cur = conn.cursor()
            try:
                cur.execute(
                    "INSERT INTO pokemon_features (species, variant_name, feature_vector, created_at) "
                    "VALUES (?, ?, ?, strftime('%s', 'now'))",
                    (species, variant, payload),
                )
                cur.execute(
                    "INSERT INTO species_info (species, count, last_updated) "
                    "VALUES (?, 1, strftime('%s', 'now')) "
                    "ON CONFLICT(species) DO UPDATE SET count = count + 1, last_updated = strftime('%s', 'now')",
                    (species,),
                )
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise
            finally:
                _close(cur)
        self._db(run)

    def delete_learned(self, species: Optional[str] = None, last_only: bool = False) -> List[str]:
        """Removes learned examples (never the original training data). Returns the species removed."""
        with self._write_lock:
            def run(conn):
                cur = conn.cursor()
                try:
                    query = "SELECT id, species FROM pokemon_features WHERE instr(variant_name, ?) > 0"
                    params = [LEARNED_MARK]
                    if species:
                        query += " AND species = ?"
                        params.append(species)
                    query += " ORDER BY id DESC"
                    if last_only:
                        query += " LIMIT 1"
                    cur.execute(query, params)
                    rows = [(r[0], r[1]) for r in cur.fetchall()]
                    for row_id, sp in rows:
                        cur.execute("DELETE FROM pokemon_features WHERE id = ?", (row_id,))
                        cur.execute("UPDATE species_info SET count = MAX(count - 1, 0) WHERE species = ?", (sp,))
                    conn.commit()
                    return rows
                except Exception:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    raise
                finally:
                    _close(cur)
            rows = self._db(run)
            if rows:
                new_species, new_matrix = self._load_rows()
                with self._lock:
                    self._species, self._matrix = new_species, new_matrix
                    self.version += 1
            return [sp for _, sp in rows]

    def count_learned(self) -> int:
        def run(conn):
            cur = conn.cursor()
            try:
                cur.execute("SELECT COUNT(*) FROM pokemon_features WHERE instr(variant_name, ?) > 0", (LEARNED_MARK,))
                return int(cur.fetchone()[0])
            finally:
                _close(cur)
        with self._write_lock:
            return self._db(run)


def _close(cur):
    try:
        cur.close()
    except Exception:
        pass
