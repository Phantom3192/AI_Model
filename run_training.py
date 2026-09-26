"""
run_training.py - Entrypoint for the standalone trainer deployment.

train_model.py itself never calls load_dotenv() - in the original repo that
was done once by main.py (the Discord bot) before importing train_model.
Since this deployment runs training on its own, without main.py, this tiny
wrapper does that one step and then calls the same stream_train() function
main.py's sibling script uses. Nothing in train_model.py is modified.

Usage:
    python run_training.py
"""

from dotenv import load_dotenv
load_dotenv()

from train_model import stream_train

if __name__ == "__main__":
    stream_train()
