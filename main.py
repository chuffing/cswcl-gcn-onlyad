# main.py
import argparse

from src.config import Config, VALID_ABLATION_MODES
from src.train import run_5fold_training


def parse_args():
    parser = argparse.ArgumentParser(description="Run 5-fold training on supported datasets.")
    parser.add_argument(
        "--dataset",
        dest="dataset_name",
        default="nc_smc_lmci",
        choices=["nc_smc_lmci", "data_5"],
        help="选择要运行的数据集",
    )
    parser.add_argument(
        "--mode",
        dest="ablation_mode",
        choices=sorted(VALID_ABLATION_MODES),
        default=None,
        help="选择消融模式；不传时使用该数据集默认模式",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(dataset_name=args.dataset_name)
    if args.ablation_mode is not None:
        cfg.ablation_mode = args.ablation_mode
    run_5fold_training(cfg)