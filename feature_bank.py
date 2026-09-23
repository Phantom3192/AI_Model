"""
feature_bank.py - the embedding bank behind the AI_Model API.

Holds every stored (species, vector) pair as one NumPy matrix in memory (so a lookup is a single
matrix multiply) and keeps it in sync with the `pokemon_features` table that train_model.py writes.
It talks to the same PostgreSQL database as the trainer (POSTGRES_DSN, or the standard libpq env
vars PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE).

Deliberately has NO torch / train_model dependency, so the API server can run on onnxruntime alone.

Storage format note
-------------------
Feature vectors are stored as BYTEA (raw fp32 bytes), not JSON text. At ~300k rows JSON text is
~3.4x larger on disk and network, and loading it requires json.loads() on every row - which,
multiplied by 300k, materialises hundreds of MB of transient Python floats and is what made the
initial bank load OOM the container. BYTEA is fixed-width, exact, and decodes with a single
np.frombuffer call per row.

The first time this runs against a database that was created by the old (TEXT) version, it will
migrate the pokemon_features.feature_vector column from TEXT to BYTEA row-by-row. That is a
one-time cost.
"""

import os
import re
import json
import time
import uuid
import difflib
import logging
import threading
from typing import Callable, List, Optional, Tuple

import numpy as np
import psycopg2
import psycopg2.extras

log = logging.getLogger("ai_model.bank")

# Same marker main.py used, so vectors learned through the API and through the old bot commands
# are recognised (and can be forgotten) the same way.
LEARNED_MARK = "__learned_"

# How many rows to pull per fetchmany() call when loading the bank. Tuned so peak transient
# memory during a load stays around (chunk_size * row_size), not (total_rows * row_size).
LOAD_CHUNK_ROWS = 2000

# How many rows to insert per executemany() call when migrating TEXT -> BYTEA.
MIGRATE_CHUNK_ROWS = 500


def species_key(name: str) -> str:
    """Loose comparison key so 'Mr. Mime', 'mr_mime' and 'mr._mime' all match."""
    return re.sub(r"[^a-z0-9♀♂]+", "", name.lower())


def new_species_name(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip().lower())


def _env(name: str, default: str = "") -> str:
    """os.getenv minus a trailing "  # comment" (some launchers load .env without stripping them)."""
    v = os.getenv(name)
    return default if v is None else re.split(r"\s+#", v, maxsplit=1)[0].strip()


def _vec_to_bytea(vec: np.ndarray) -> bytes:
    """(256,) float32 -> 1024 raw bytes, little-endian (numpy's native order on x86/ARM)."""
    return np.ascontiguousarray(vec, dtype=np.float32).tobytes()


def _bytea_to_vec(raw) -> np.ndarray:
    """bytes / memoryview / bytearray -> (N,) float32 numpy array. Zero-copy on memoryview."""
    if raw is None:
        raise ValueError("null feature_vector")
    if isinstance(raw, memoryview):
        return np.frombuffer(raw, dtype=np.float32).copy()
    return np.frombuffer(bytes(raw), dtype=np.float32).copy()


class FeatureBank:
    def __init__(self, dsn: Optional[str] = None, db_path: Optional[str] = None):
        # db_path kept in the signature for backwards-compat with any caller that still passes it;
        # it is ignored now that everything goes through PostgreSQL.
        self.dsn = dsn if dsn is not None else _env("POSTGRES_DSN")
        self.backend = "postgres"

        self.version = 0                      # bumped whenever the in-memory bank changes
        self._species: List[str] = []
        self._matrix = np.zeros((0, 256), dtype=np.float32)
        self._lock = threading.RLock()        # guards the (species, matrix, version) triple
        self._write_lock = threading.RLock()  # serialises every DB access + bank mutation
        self._conn = self._connect()
        self._ensure_tables()
        self._migrate_text_to_bytea_if_needed()

    # ------------------------------------------------------------------ connection
    def _connect(self):
        # POSTGRES_DSN if set, else libpq's own env vars (PGHOST, PGUSER, ...).
        conn = psycopg2.connect(self.dsn) if self.dsn else psycopg2.connect()
        # Fail a stuck query loudly rather than letting the client hang until the OS kills it.
        # 5 min is generous for a full-table stream on a small box; the load path itself is now
        # chunked so it should never approach this, but the guard stays.
        try:
            with conn.cursor() as cur:
                cur.execute("SET statement_timeout = '300s'")
                cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ COMMITTED")
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
        return conn

    def _reconnect(self):
        try:
            self._conn.close()
        except Exception:
            pass
        self._conn = self._connect()

    def _db(self, fn: Callable):
        """Run fn(conn); on failure reconnect once and retry (PostgreSQL connections can go stale)."""
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
                        id BIGSERIAL PRIMARY KEY,
                        species TEXT NOT NULL,
                        variant_name TEXT NOT NULL,
                        feature_vector BYTEA NOT NULL,
                        created_at BIGINT DEFAULT (EXTRACT(EPOCH FROM now())::bigint),
                        UNIQUE(species, variant_name)
                    )""")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS species_info (
                        species TEXT PRIMARY KEY,
                        count INTEGER DEFAULT 0,
                        last_updated BIGINT DEFAULT (EXTRACT(EPOCH FROM now())::bigint)
                    )""")
                conn.commit()
            finally:
                _close(cur)
        with self._write_lock:
            self._db(run)

    def _migrate_text_to_bytea_if_needed(self):
        """
        If pokemon_features.feature_vector is still TEXT (JSON), convert every row to BYTEA.
        Runs once: after it completes, the column type is BYTEA and this is a no-op.
        """
        with self._write_lock:
            def detect(conn):
                cur = conn.cursor()
                try:
                    cur.execute("""
                        SELECT data_type FROM information_schema.columns
                        WHERE table_name = 'pokemon_features' AND column_name = 'feature_vector'
                    """)
                    row = cur.fetchone()
                    return row[0].lower() if row else None
                finally:
                    _close(cur)

            dtype = self._db(detect)
            if dtype == "bytea" or dtype is None:
                return
            if dtype not in ("text", "character varying"):
                log.warning("feature_vector column is %s - not migrating", dtype)
                return

            log.info("Migrating pokemon_features.feature_vector from %s to BYTEA (one-time)...", dtype)

            # Add the new column alongside, migrate, then swap - safer than ALTER TYPE USING,
            # which would need a full-table rewrite in one statement.
            def add_col(conn):
                cur = conn.cursor()
                try:
                    cur.execute("ALTER TABLE pokemon_features ADD COLUMN IF NOT EXISTS feature_vector_b BYTEA")
                    conn.commit()
                finally:
                    _close(cur)
            self._db(add_col)

            migrated = 0
            while True:
                def fetch_chunk(conn):
                    cur = conn.cursor()
                    try:
                        cur.execute("""
                            SELECT id, feature_vector FROM pokemon_features
                            WHERE feature_vector_b IS NULL
                            LIMIT %s
                        """, (MIGRATE_CHUNK_ROWS,))
                        return cur.fetchall()
                    finally:
                        _close(cur)

                rows = self._db(fetch_chunk)
                if not rows:
                    break

                def write_chunk(conn, chunk):
                    cur = conn.cursor()
                    try:
                        data = []
                        for row_id, json_text in chunk:
                            try:
                                vec = np.asarray(json.loads(json_text), dtype=np.float32)
                            except Exception:
                                log.warning("row %s: unparseable feature_vector, skipping", row_id)
                                continue
                            data.append((_vec_to_bytea(vec), row_id))
                        if data:
                            psycopg2.extras.execute_batch(
                                cur,
                                "UPDATE pokemon_features SET feature_vector_b = %s WHERE id = %s",
                                data,
                                page_size=200,
                            )
                        conn.commit()
                        return len(data)
                    except Exception:
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                        raise
                    finally:
                        _close(cur)

                n = self._db(lambda c: write_chunk(c, rows))
                migrated += n
                if migrated % 20000 < MIGRATE_CHUNK_ROWS:
                    log.info("  ... migrated %d rows", migrated)

            log.info("Migration: %d rows converted. Swapping columns...", migrated)

            def swap(conn):
                cur = conn.cursor()
                try:
                    cur.execute("ALTER TABLE pokemon_features DROP COLUMN feature_vector")
                    cur.execute("ALTER TABLE pokemon_features RENAME COLUMN feature_vector_b TO feature_vector")
                    cur.execute("ALTER TABLE pokemon_features ALTER COLUMN feature_vector SET NOT NULL")
                    conn.commit()
                finally:
                    _close(cur)
            self._db(swap)

            # VACUUM to reclaim the dead JSON rows. VACUUM cannot run inside a transaction block,
            # so this is on a fresh autocommit cursor.
            try:
                old_iso = self._conn.isolation_level
                self._conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
                cur = self._conn.cursor()
                try:
                    cur.execute("VACUUM ANALYZE pokemon_features")
                finally:
                    _close(cur)
                    self._conn.set_isolation_level(old_iso)
            except Exception as e:
                log.warning("VACUUM after migration failed (non-fatal): %s", e)

            log.info("Migration complete.")

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
        """
        Streams the entire pokemon_features table in LOAD_CHUNK_ROWS-sized fetchmany() windows
        and assembles the (N, dim) matrix incrementally, row by row into a preallocated array.

        Why not the obvious `fetchall()` + `np.asarray([json.loads(r[1]) for r in rows])`:
        at 300k rows that is ~1.1 GB of JSON strings pulled through the socket in one shot,
        followed by ~2.4 GB of transient Python float objects from the list comprehension, in a
        container with a ~3.4 GB limit. That is the exact sequence that was causing
        "could not send data to client: Connection reset by peer" in the Postgres log - the
        client OOM-killed itself mid-select. Streaming + preallocation keeps peak transient
        memory at roughly one chunk's worth (a few MB) regardless of table size.
        """
        # First, get the row count and feature dimension so we can preallocate exactly.
        def count_and_dim(conn):
            cur = conn.cursor()
            try:
                cur.execute("SELECT COUNT(*) FROM pokemon_features")
                n = int(cur.fetchone()[0])
                if n == 0:
                    return 0, 256
                cur.execute("""
                    SELECT octet_length(feature_vector) FROM pokemon_features
                    WHERE feature_vector IS NOT NULL LIMIT 1
                """)
                row = cur.fetchone()
                dim = int(row[0]) // 4 if row and row[0] else 256
                return n, dim
            finally:
                _close(cur)

        n_total, dim = self._db(count_and_dim)
        if n_total == 0:
            return [], np.zeros((0, 256), dtype=np.float32)

        log.info("Loading feature bank: %d rows x %d dims (%.1f MB fp32)...",
                 n_total, dim, n_total * dim * 4 / (1024 * 1024))

        species_list: List[str] = []
        species_list_append = species_list.append
        matrix = np.empty((n_total, dim), dtype=np.float32)

        # Server-side cursor: psycopg2 will use DECLARE/FETCH under the hood, so Postgres
        # streams rows out instead of buffering the whole result set in server memory first.
        def stream(conn):
            cur = conn.cursor(name="bank_load")  # named cursor = server-side
            try:
                cur.itersize = LOAD_CHUNK_ROWS
                cur.execute("SELECT species, feature_vector FROM pokemon_features ORDER BY id")
                i = 0
                while True:
                    rows = cur.fetchmany(LOAD_CHUNK_ROWS)
                    if not rows:
                        break
                    for species, raw in rows:
                        if i >= n_total:
                            break
                        species_list_append(species)
                        try:
                            matrix[i] = _bytea_to_vec(raw)
                        except Exception as e:
                            log.warning("row %d: bad feature_vector (%s), zeroing", i, e)
                            matrix[i] = 0.0
                        i += 1
                return i
            finally:
                _close(cur)

        written = self._db(stream)
        if written != n_total:
            # Trim in case rows were deleted between the COUNT and the SELECT.
            species_list = species_list[:written]
            matrix = matrix[:written]

        if matrix.shape[0] == 0:
            return [], np.zeros((0, dim), dtype=np.float32)

        # Defensive: anything that decoded to all-zeros is a corrupt/missing row; drop it so
        # it can't win a match by accident.
        norms = np.linalg.norm(matrix, axis=1)
        good = norms > 1e-6
        if not good.all():
            log.warning("dropping %d zero-length feature rows", int((~good).sum()))
            matrix = matrix[good]
            species_list = [s for s, keep in zip(species_list, good.tolist()) if keep]

        log.info("Feature bank loaded: %d rows x %d dims", matrix.shape[0], matrix.shape[1])
        return species_list, matrix

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
        payload = _vec_to_bytea(vec)

        def run(conn):
            cur = conn.cursor()
            try:
                cur.execute(
                    "INSERT INTO pokemon_features (species, variant_name, feature_vector, created_at) "
                    "VALUES (%s, %s, %s, EXTRACT(EPOCH FROM now())::bigint)",
                    (species, variant, psycopg2.Binary(payload)),
                )
                cur.execute(
                    "INSERT INTO species_info (species, count, last_updated) "
                    "VALUES (%s, 1, EXTRACT(EPOCH FROM now())::bigint) "
                    "ON CONFLICT(species) DO UPDATE SET count = species_info.count + 1, "
                    "last_updated = EXCLUDED.last_updated",
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
                    query = "SELECT id, species FROM pokemon_features WHERE position(%s in variant_name) > 0"
                    params = [LEARNED_MARK]
                    if species:
                        query += " AND species = %s"
                        params.append(species)
                    query += " ORDER BY id DESC"
                    if last_only:
                        query += " LIMIT 1"
                    cur.execute(query, params)
                    rows = [(r[0], r[1]) for r in cur.fetchall()]
                    for row_id, sp in rows:
                        cur.execute("DELETE FROM pokemon_features WHERE id = %s", (row_id,))
                        cur.execute("UPDATE species_info SET count = GREATEST(count - 1, 0) WHERE species = %s", (sp,))
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
                cur.execute("SELECT COUNT(*) FROM pokemon_features WHERE position(%s in variant_name) > 0", (LEARNED_MARK,))
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
