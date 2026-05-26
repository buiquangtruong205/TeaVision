import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(command):
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def prepare(args, include_kfold=False):
    command = [
        sys.executable,
        "scripts/prepare_data.py",
        "--dataset-dir",
        args.dataset_dir,
        "--splits-dir",
        args.splits_dir,
        "--config-dir",
        args.config_dir,
        "--image-size",
        str(args.image_size),
        "--val-size",
        str(args.val_size),
        "--test-size",
        str(args.test_size),
        "--seed",
        str(args.seed),
    ]
    if include_kfold:
        command.extend(["--kfold", "--n-splits", str(args.n_splits)])
    run(command)


def parse_args():
    parser = argparse.ArgumentParser(description="Run TeaVision pipeline steps.")
    parser.add_argument("--mode", choices=["prepare", "crossval", "full"], default="prepare")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "prepare":
        prepare(args, include_kfold=False)
    elif args.mode == "crossval":
        prepare(args, include_kfold=True)
    elif args.mode == "full":
        prepare(args, include_kfold=True)
        print("Train/eval/export steps will be connected after the training entrypoint is implemented.")


if __name__ == "__main__":
    main()
