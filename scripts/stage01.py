import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dafnet.runtime import load_config

from dafnet import stage01
from dafnet.runtime import require_directory

def main():
    parser = argparse.ArgumentParser(description="DAF-Net stage01")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    require_directory(config['stage01']['input_dir'])
    stage01.INPUT_DIR = config['stage01']['input_dir']
    stage01.OUTPUT_DIR = config['stage01']['output_dir']
    stage01.main()


if __name__ == "__main__":
    main()
