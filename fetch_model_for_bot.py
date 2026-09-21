"""
fetch_model_for_bot.py - Run this on the BOT deployment (not the trainer) to
pull down whatever publish_model.py last pushed, before the bot starts.

This file lives here for convenience since it's the other half of the
publish_model.py workflow, but it belongs on your main.py bot deployment,
not on this trainer deployment. Copy it over there.

Usage (on the bot host, before starting main.py):
    python fetch_model_for_bot.py
"""

import os
import sys

from dotenv import load_dotenv
load_dotenv()


def main():
    repo = os.getenv("HF_MODEL_REPO")
    if not repo:
        sys.exit("❌ HF_MODEL_REPO not set - nothing to fetch.")

    hf_token = os.getenv("HF_TOKEN")
    model_output = os.getenv("MODEL_OUTPUT", "models/pokemon_classifier.pt")
    onnx_output = os.getenv("ONNX_MODEL_PATH", os.path.splitext(model_output)[0] + ".onnx")

    from huggingface_hub import hf_hub_download

    os.makedirs(os.path.dirname(model_output) or ".", exist_ok=True)

    print(f"⬇️  Fetching {os.path.basename(model_output)} from {repo} ...")
    downloaded = hf_hub_download(repo_id=repo, filename=os.path.basename(model_output), token=hf_token)
    _copy(downloaded, model_output)

    for fname in (os.path.basename(onnx_output), os.path.basename(onnx_output) + ".json"):
        try:
            downloaded = hf_hub_download(repo_id=repo, filename=fname, token=hf_token)
            _copy(downloaded, os.path.join(os.path.dirname(onnx_output) or ".", fname))
        except Exception:
            pass  # optional - main.py will re-export the onnx itself if missing

    print("✅ Model files ready. Start main.py normally.")


def _copy(src, dst):
    import shutil
    shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
