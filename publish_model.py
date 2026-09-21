"""
publish_model.py - Push the trained model (+ onnx export) to a Hugging Face
model repo, so a SEPARATE bot deployment can pull it down.

Why this is needed: TURSO_URL already lets the trainer and the bot share the
feature DATABASE directly (same Turso instance, no transfer needed). But the
model FILE (models/pokemon_classifier.pt) only exists on the trainer's local
disk after training - the bot deployment needs that exact file, since the
feature bank's vectors only make sense alongside the model that produced them
(see onnx_backend.py's docstring: embeddings must match the feature bank).

This script uploads both files to a HF model repo you own. Pair it with
fetch_model.py (bot side) which downloads them back down before the bot
starts serving predictions.

Usage:
    python publish_model.py
    python publish_model.py --repo yourname/pokemon-classifier
"""

import os
import sys
import argparse

from dotenv import load_dotenv
load_dotenv()


def main():
    ap = argparse.ArgumentParser(description="Publish the trained model to a HF model repo")
    ap.add_argument("--repo", default=os.getenv("HF_MODEL_REPO"),
                     help="HF model repo id, e.g. yourname/pokemon-classifier")
    ap.add_argument("--model", default=None, help="path to the .pt (default: MODEL_OUTPUT)")
    ap.add_argument("--onnx", default=None, help="path to the .onnx (default: ONNX_MODEL_PATH, skipped if missing)")
    args = ap.parse_args()

    if not args.repo:
        sys.exit("❌ No repo given. Set HF_MODEL_REPO in .env or pass --repo yourname/pokemon-classifier")

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        sys.exit("❌ HF_TOKEN is not set - needed to push to a model repo.")

    from train_model import MODEL_OUTPUT, ONNX_MODEL_PATH
    model_path = args.model or MODEL_OUTPUT
    onnx_path = args.onnx or ONNX_MODEL_PATH

    if not os.path.exists(model_path):
        sys.exit(f"❌ {model_path} not found - run training first.")

    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)

    print(f"📦 Creating/checking repo {args.repo} ...")
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=True)

    print(f"⬆️  Uploading {model_path} ...")
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=os.path.basename(model_path),
        repo_id=args.repo,
        repo_type="model",
    )

    if os.path.exists(onnx_path):
        print(f"⬆️  Uploading {onnx_path} ...")
        api.upload_file(
            path_or_fileobj=onnx_path,
            path_in_repo=os.path.basename(onnx_path),
            repo_id=args.repo,
            repo_type="model",
        )
        meta_path = onnx_path + ".json"
        if os.path.exists(meta_path):
            api.upload_file(
                path_or_fileobj=meta_path,
                path_in_repo=os.path.basename(meta_path),
                repo_id=args.repo,
                repo_type="model",
            )
    else:
        print(f"   (no {onnx_path} found - skipping, bot will re-export from the .pt on startup "
              f"if AUTO_EXPORT_ONNX is on)")

    print(f"✅ Published to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
