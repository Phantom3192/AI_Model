# Pokémon Trainer — standalone deployment

This is a slimmed-down copy of the training half of `Model_Trainer`, meant to
run as its **own** hosted server, separate from the Discord bot (`main.py`).
Nothing here is a rewrite — `train_model.py`, `export_onnx.py`,
`onnx_backend.py`, and `predict.py` are unmodified copies of your originals.
Only new files were added on top.

## What's different from the full repo

- **No `main.py`, no `discord.py`/`aiohttp`** — this deployment never runs
  the bot, only training, so those dependencies are dropped from
  `requirements.txt`.
- **`run_training.py`** — new entrypoint. `train_model.py` itself never calls
  `load_dotenv()` (the original repo relied on `main.py` doing that first);
  this one-line wrapper does it so the trainer works standalone.
- **`publish_model.py`** / **`fetch_model_for_bot.py`** — new, optional.
  See "Getting the trained model back to your bot" below.

## How the two deployments share data

| What | How it's shared |
|---|---|
| Feature database (embeddings) | Point **both** deployments at the same `TURSO_URL` / `TURSO_AUTH_TOKEN`. Training writes directly into it — no file transfer needed. |
| Model file (`pokemon_classifier.pt`) | **Not** shared automatically — it only exists on this trainer's local disk after training. See below. |

### Getting the trained model back to your bot

The bot needs the exact `.pt` file this trainer produced (the feature bank's
vectors only make sense next to the model that generated them). Two options:

**Option A — Hugging Face Hub (included, recommended)**
```bash
# after training finishes, on the trainer:
python publish_model.py --repo yourname/pokemon-classifier

# on the bot deployment, before starting main.py:
python fetch_model_for_bot.py
```
Both scripts read `HF_TOKEN` / `HF_MODEL_REPO` from `.env`. `fetch_model_for_bot.py`
belongs on the *bot* host — it's included here only because it's the other
half of the same workflow.

**Option B — manual**
Copy `models/pokemon_classifier.pt` (and, if present, the `.onnx` +
`.onnx.json`) from this container to the bot deployment however your panel
supports file transfer, then restart the bot.

## Running it

Set your `.env` from `.env.example` first (copy it, fill in `TURSO_URL`,
`HF_TOKEN`, and tune `MAX_IMAGES_PER_SPECIES` etc.), then:

```bash
pip install -r requirements.txt
python run_training.py
```

Or with Docker:
```bash
docker build -t poketrainer .
docker run --env-file .env poketrainer
```

If your panel wants a plain start command instead of Docker, use:
```
python run_training.py
```
as the startup command, after an install step of `pip install -r requirements.txt`.

## Sizing this deployment

Training is CPU/RAM-heavy in bursts (embedding images through EfficientNet-B0)
but doesn't need to run 24/7 like the bot does — you can likely give this
deployment more cores/memory than the bot's box and only spin it up when
retraining, rather than paying for it continuously. `STREAM_BATCH_SIZE` in
`.env` auto-lowers itself if it detects the container's cgroup memory limit
is tight (see the comment in `train_model.py` around `stream_train()`), so
it's reasonably safe to just try a size and let it self-adjust.
