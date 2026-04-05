# main.py
from src.config import Config
from src.train import run_5fold_training

if __name__ == "__main__":
    cfg = Config()
    run_5fold_training(cfg)