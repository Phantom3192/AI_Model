# AI_Model

The Pokémon identifier as a standalone HTTP service. The Discord bot sends it an image and gets
the species back; the bot no longer needs the model, torch, onnxruntime, or the database.

```
Discord bot ──(image, X-API-Key)──▶  AI_Model  ──▶  ONNX model + feature bank (Turso)
            ◀──── species, score ────
```

Inference runs on ONNX Runtime, so the server process never imports torch. torch is only used by
the child processes that train the model or export the `.onnx`.

## API

Send the key as `X-API-Key: <API_KEY>` (or `Authorization: Bearer <API_KEY>`). `/health` needs no key.

| Endpoint | What it does |
|---|---|
| `POST /v1/predict` | multipart `file` → `species`, `score`, `confident`, `neighbors`, `bank_version`. Query: `threshold`, `include_embedding` |
| `POST /v1/predict/batch` | multipart `files` (repeat the field) → `results[]` in input order, one model call for all. `{"ok": false}` for unreadable images |
| `POST /v1/learn` | form `species`, `allow_new`, and **either** `file` **or** `embedding` (JSON array from a previous predict) → `status`: `learned` / `duplicate` / `unknown` (+`suggestions`) / `bad_image` |
| `POST /v1/forget` | JSON `{"species": "..."}` (all learned examples of it) or `{"last": true}` (undo). Never touches training data |
| `GET /v1/stats` | bank size, learned count, latency p50/p95 |
| `GET /health` | `200` when ready, `503` while loading / on error (body says why) |
| `POST /admin/reload` | re-read the model and feature bank without a restart |
| `POST /admin/train` | start `run_training.py` in a subprocess; reloads automatically on success |
| `GET /admin/train/status` | state (`running`/`succeeded`/`failed`/…) + log tail |
| `POST /admin/train/stop` | abort the run |

`/admin/*` returns 403 until `API_KEY` is set. Interactive docs: `/docs`.

Every prediction/learn/forget response includes `bank_version`, which changes whenever the bank
changes. Clear the bot's local result cache when it moves.

## Run it

1. Copy `.env.example` to `.env`, set `API_KEY`, and set `TURSO_URL` / `TURSO_AUTH_TOKEN` to the
   **same** database the bot uses today (the feature bank is already there).
2. Put the model next to the code: `models/pokemon_classifier.pt`, `models/pokemon_classifier.onnx`
   and `models/pokemon_classifier.onnx.json` (from your `Poke-Glitchh` repo). Also copy
   `Extra pokemons.zip` here; retraining needs it.
3. Start:
   ```bash
   pip install -r requirements.txt         # serving only
   python server.py                        # http://0.0.0.0:8000
   ```
   Docker: `docker build -t ai-model .` then `docker run --env-file .env -p 8000:8000 ai-model`
   (`--build-arg WITH_TRAINING=0` gives a small image without torch).

Keep it to **one worker process**: the feature bank is held in memory.

If the model or bank isn't there yet the server still starts; `/health` says what's missing and
`/admin/reload` picks it up once you've fixed it.

## Using it from the bot

Copy `ai_client.py` into the bot project (it needs only `aiohttp` and `numpy`). Set `AI_MODEL_URL` and `AI_MODEL_API_KEY`.

```python
from ai_client import AIModelClient, AIModelError, AIModelUnavailable

ai = AIModelClient(os.environ["AI_MODEL_URL"], os.getenv("AI_MODEL_API_KEY"))

pred = await ai.predict(image_bytes, include_embedding=True)
winner, score, neighbors, vec = pred.as_tuple()      # same shape main.py uses today
confident = score >= CONFIDENCE_THRESHOLD            # keep s!threshold on the bot side

await ai.learn(name, embedding=vec, allow_new=False) # s!learn / auto-learn
await ai.forget(species)                             # s!forget
await ai.forget(last=True)                           # s!undo
await ai.reload()                                    # s!reload
await ai.close()                                     # on shutdown
```

`AIModelUnavailable` means the service can't be reached or isn't ready (safe to retry later);
`AIModelError` carries the HTTP status for anything else.

## Retraining

`POST /admin/train`, then poll `GET /admin/train/status`. It runs `run_training.py` with the
training variables from `.env`, and on success rebuilds the `.onnx` and reloads the model and bank.
It needs `pip install -r requirements-train.txt` and enough RAM for training.
The start response lists warnings (for example a missing `Extra pokemons.zip`).

With `REPLACE_DB_FEATURES=true` the whole bank is rewritten, which drops learned vectors made by
the old model (they don't match the new one). Keep those images in `Extra pokemons.zip` before you retrain.

## Files

- `server.py`: the API. `feature_bank.py`: DB + in-memory matrix (no torch). `ai_client.py`: bot-side client.
- `train_model.py`, `export_onnx.py`, `onnx_backend.py`, `predict.py`, `run_training.py`: unchanged from the trainer.
- Removed from `Model_Trainer`: `publish_model.py` / `fetch_model_for_bot.py`. The bot no longer holds the model,
  so nothing needs to be published or fetched. (`publish_model.py` also imported a name, `ONNX_MODEL_PATH`,
  that `train_model.py` doesn't define, so it would have crashed on start.)

## Tests

`pip install pytest httpx && pytest -q` runs 14 API tests against your real ONNX model and a temporary
SQLite bank (never Turso). Set `TEST_MODEL_PT` / `TEST_MODEL_ONNX` if the model isn't in `models/`.
