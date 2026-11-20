import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dafnet.runtime import load_config

from dafnet.runtime import evaluate

def main():
    parser = argparse.ArgumentParser(description="DAF-Net evaluate")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    evaluate(config, efficiency=True)


if __name__ == "__main__":
    main()
