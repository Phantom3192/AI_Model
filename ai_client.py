"""
ai_client.py - copy this ONE file into the Discord bot project.

It is the bot's whole interface to the AI_Model service: no torch, no onnxruntime, no database.
Only needs aiohttp (which discord.py already installs) and numpy.

    from ai_client import AIModelClient, AIModelError, AIModelUnavailable

    ai = AIModelClient(os.environ["AI_MODEL_URL"], os.getenv("AI_MODEL_API_KEY"))

    pred = await ai.predict(image_bytes, include_embedding=True)
    pred.species, pred.score, pred.confident, pred.neighbors

    # drop-in for the old (winner, score, neighbors, vec) tuples:
    winner, score, neighbors, vec = pred.as_tuple()

    await ai.learn("pikachu", embedding=pred.embedding)   # or image_bytes=...
    await ai.forget(last=True)                            # s!undo
    await ai.close()                                      # on shutdown

Cache tip: every response carries `bank_version`. It changes whenever the feature bank changes
(learn / forget / retrain / reload), so clear your local result cache when it changes.
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np


class AIModelError(Exception):
    """The service answered with an error (4xx/5xx). `.status` and `.detail` describe it."""

    def __init__(self, status: int, detail: Any):
        self.status, self.detail = status, detail
        super().__init__(f"AI_Model HTTP {status}: {detail}")


class AIModelUnavailable(AIModelError):
    """Couldn't reach the service, or it isn't ready yet (503). Safe to retry later."""

    def __init__(self, detail: Any, status: int = 503):
        super().__init__(status, detail)


@dataclass
class Prediction:
    species: str
    score: float
    confident: bool
    threshold: float
    neighbors: List[Tuple[str, float]]
    bank_version: int
    embed_ms: float = 0.0
    match_ms: float = 0.0
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def as_tuple(self):
        """(species, score, [(species, sim), ...], embedding) - same shape main.py already uses."""
        return self.species, self.score, self.neighbors, self.embedding

    @classmethod
    def from_json(cls, j: Dict[str, Any]) -> "Prediction":
        emb = j.get("embedding")
        return cls(
            species=j["species"], score=float(j["score"]), confident=bool(j["confident"]),
            threshold=float(j["threshold"]),
            neighbors=[(n["species"], float(n["score"])) for n in j["neighbors"]],
            bank_version=int(j["bank_version"]),
            embed_ms=float(j["timing_ms"]["embed"]), match_ms=float(j["timing_ms"]["match"]),
            embedding=np.asarray(emb, dtype=np.float32) if emb is not None else None,
        )


@dataclass
class LearnResult:
    status: str                      # learned | duplicate | unknown | bad_image
    species: Optional[str]
    examples: Optional[int]
    suggestions: List[str]
    bank_version: Optional[int]


class AIModelClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 30.0, retries: int = 1):
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key} if api_key else {}
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.retries = max(0, retries)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout, headers=self.headers)
        return self._session

    async def close(self):
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def _request(self, method: str, path: str, *, retry: bool = True, ok_503: bool = False, **kw) -> Dict[str, Any]:
        attempts = 1 + (self.retries if retry else 0)
        last: Optional[Exception] = None
        for i in range(attempts):
            # FormData can only be sent once, so it is rebuilt per attempt via a factory.
            payload = dict(kw)
            if callable(payload.get("data")):
                payload["data"] = payload["data"]()
            try:
                session = await self._get_session()
                async with session.request(method, self.base_url + path, **payload) as resp:
                    text = await resp.text()
                    try:
                        body = json.loads(text) if text else {}
                    except ValueError:
                        body = {"detail": text[:300]}
                    if resp.status == 503 and ok_503:
                        return body
                    if resp.status == 503:
                        raise AIModelUnavailable(body.get("detail") or body.get("error") or "service not ready")
                    if resp.status >= 400:
                        raise AIModelError(resp.status, body.get("detail", body))
                    return body
            except AIModelUnavailable as e:
                last = e
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
                last = AIModelUnavailable(f"{type(e).__name__}: {e}", status=0)
            if i + 1 < attempts:
                await asyncio.sleep(0.5 * (i + 1))
        assert last is not None
        raise last

    # ---------------------------------------------------------------- inference
    async def predict(self, image_bytes: bytes, *, threshold: Optional[float] = None,
                      include_embedding: bool = False) -> Prediction:
        def form():
            f = aiohttp.FormData()
            f.add_field("file", image_bytes, filename="image", content_type="application/octet-stream")
            return f
        params: Dict[str, Any] = {"include_embedding": str(include_embedding).lower()}
        if threshold is not None:
            params["threshold"] = threshold
        return Prediction.from_json(await self._request("POST", "/v1/predict", data=form, params=params))

    async def predict_batch(self, images: List[bytes], *, threshold: Optional[float] = None,
                            include_embedding: bool = False) -> List[Optional[Prediction]]:
        """One result per input image, in order. None = the service couldn't read that image."""
        def form():
            f = aiohttp.FormData()
            for i, b in enumerate(images):
                f.add_field("files", b, filename=f"image{i}", content_type="application/octet-stream")
            return f
        params: Dict[str, Any] = {"include_embedding": str(include_embedding).lower()}
        if threshold is not None:
            params["threshold"] = threshold
        body = await self._request("POST", "/v1/predict/batch", data=form, params=params)
        return [Prediction.from_json(r) if r.get("ok") else None for r in body["results"]]

    # ---------------------------------------------------------------- teaching
    async def learn(self, species: str, *, image_bytes: Optional[bytes] = None,
                    embedding: Optional[Any] = None, allow_new: bool = False) -> LearnResult:
        if (image_bytes is None) == (embedding is None):
            raise ValueError("pass exactly one of image_bytes or embedding")

        def form():
            f = aiohttp.FormData()
            f.add_field("species", species)
            f.add_field("allow_new", str(allow_new).lower())
            if image_bytes is not None:
                f.add_field("file", image_bytes, filename="image", content_type="application/octet-stream")
            else:
                f.add_field("embedding", json.dumps([float(x) for x in embedding]))
            return f
        j = await self._request("POST", "/v1/learn", data=form, retry=False)
        return LearnResult(j["status"], j.get("species"), j.get("examples"),
                           j.get("suggestions") or [], j.get("bank_version"))

    async def forget(self, species: Optional[str] = None, *, last: bool = False) -> Dict[str, Any]:
        """{"removed": n, "species": [...], "bank_version": v}. Raises AIModelError(404) for unknown species."""
        body: Dict[str, Any] = {"last": last}
        if species:
            body["species"] = species
        return await self._request("POST", "/v1/forget", json=body, retry=False)

    # ---------------------------------------------------------------- info / admin
    async def health(self) -> Dict[str, Any]:
        """Never raises for a not-ready service; returns its /health body (check ['ready'])."""
        try:
            return await self._request("GET", "/health", retry=False, ok_503=True)
        except AIModelUnavailable as e:  # couldn't connect at all
            return {"ready": False, "status": "unreachable", "error": str(e.detail)}

    async def stats(self) -> Dict[str, Any]:
        return await self._request("GET", "/v1/stats")

    async def reload(self) -> Dict[str, Any]:
        return await self._request("POST", "/admin/reload", retry=False)

    async def train(self) -> Dict[str, Any]:
        return await self._request("POST", "/admin/train", retry=False)

    async def train_status(self, lines: int = 40) -> Dict[str, Any]:
        return await self._request("GET", "/admin/train/status", params={"lines": lines})
