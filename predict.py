"""
predict.py - Use the trained Pokemon classifier to identify a species from an image.

Loads the ONNX/Torch model from models/pokemon_classifier.pt and the feature bank
from bank/ (same files the API server uses). No database involved.

Usage:
    python predict.py path/to/image.jpg
    python predict.py path/to/image.jpg --top 10 --show-all
    python predict.py path/to/dir_of_images/

Environment:
    MODEL_OUTPUT  -> path to the saved model (default models/pokemon_classifier.pt)
    BANK_DIR      -> feature bank directory (default bank)
"""

import os
import sys
import glob
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train_model import PokemonFeatureExtractor, MODEL_OUTPUT

BANK_DIR = Path(os.getenv("BANK_DIR", "bank"))
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_extractor(model_path: str) -> PokemonFeatureExtractor:
    if not os.path.exists(model_path):
        sys.exit(f"❌ Model file not found: {model_path}\n"
                 f"   Run train_model.py first, or pass --model to point at the right file.")
    extractor = PokemonFeatureExtractor(embedding_dim=256)
    state_dict = torch.load(model_path, map_location="cpu")
    extractor.load_state_dict(state_dict)
    extractor.eval()
    return extractor


def load_feature_bank(bank_dir: Path):
    features_path = bank_dir / "base_features.npy"
    species_path = bank_dir / "base_species.npy"
    if not features_path.exists() or not species_path.exists():
        sys.exit(f"❌ Feature bank not found in {bank_dir}/. Run train_model.py first.")

    matrix = np.load(features_path).astype(np.float32)
    species_arr = np.load(species_path, allow_pickle=False)
    species_list = [str(s) for s in species_arr]

    # Fold in learned.jsonl if present
    learned_path = bank_dir / "learned.jsonl"
    if learned_path.exists():
        import json, base64
        learned_species = []
        learned_vecs = []
        with learned_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    vec = np.frombuffer(base64.b64decode(row["vec"]), dtype=np.float32).copy()
                    learned_species.append(row["species"])
                    learned_vecs.append(vec)
                except Exception:
                    continue
        if learned_vecs:
            species_list += learned_species
            matrix = np.concatenate([matrix, np.stack(learned_vecs)], axis=0)
            print(f"   (also loaded {len(learned_vecs)} learned examples from learned.jsonl)")

    return species_list, matrix


def predict_species(query_vec: np.ndarray, species_list, matrix: np.ndarray, top_k: int = 5):
    sims = matrix @ query_vec
    order = np.argsort(-sims)[:top_k]
    neighbors = [(species_list[i], float(sims[i])) for i in order]
    votes = Counter(sp for sp, _ in neighbors)
    winner, _ = max(
        votes.items(),
        key=lambda kv: (kv[1], max(s for sp, s in neighbors if sp == kv[0]))
    )
    best_score = max(s for sp, s in neighbors if sp == winner)
    return winner, best_score, neighbors


def iter_image_paths(path: str):
    if os.path.isdir(path):
        for ext in IMAGE_EXTS:
            yield from glob.glob(os.path.join(path, f"*{ext}"))
    else:
        yield path


def main():
    parser = argparse.ArgumentParser(description="Identify a Pokemon species from an image.")
    parser.add_argument("image_path", help="Path to an image file, or a directory of images")
    parser.add_argument("--model", default=MODEL_OUTPUT, help="Path to the trained model file")
    parser.add_argument("--bank", default=str(BANK_DIR), help="Feature bank directory")
    parser.add_argument("--top", type=int, default=5, help="Number of nearest neighbors to vote over")
    parser.add_argument("--show-all", action="store_true", help="Print all top-k neighbors")
    args = parser.parse_args()

    print(f"📦 Loading model from {args.model} ...")
    extractor = load_extractor(args.model)

    print(f"🗄️  Loading feature bank from {args.bank}/ ...")
    species_list, matrix = load_feature_bank(Path(args.bank))
    print(f"   Loaded {matrix.shape[0]} feature vectors across {len(set(species_list))} species\n")

    for img_path in iter_image_paths(args.image_path):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️  Skipping {img_path}: {e}")
            continue

        query_vec = extractor.extract(img)
        winner, score, neighbors = predict_species(query_vec, species_list, matrix, top_k=args.top)

        print(f"🔍 {os.path.basename(img_path)}")
        print(f"   → Predicted: {winner}  (similarity {score:.3f})")
        if args.show_all:
            for sp, sim in neighbors:
                print(f"      {sp:<20s} {sim:.3f}")
        print()


if __name__ == "__main__":
    main()
