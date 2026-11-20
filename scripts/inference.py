import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dafnet.runtime import load_config

from dafnet.runtime import enhance, resolve_path

def main():
    parser = argparse.ArgumentParser(description="DAF-Net inference")
    parser.add_argument("--config", required=True)
    parser.add_argument('--input', required=True, help='Image file or directory')
    parser.add_argument('--output', required=True, help='Output directory')
    args = parser.parse_args()
    config = load_config(args.config)
    count = enhance(config, args.input, args.output)
    print(f'Saved {count} enhanced images to {args.output}')


if __name__ == "__main__":
    main()
