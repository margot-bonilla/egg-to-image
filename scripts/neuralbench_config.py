"""Write ~/.neuralbench/config.json pointing NeuralBench at this repo's data/, cache/ and save/.

Replaces NeuralBench's interactive first-run prompt. An existing config is left
untouched unless --force is given.

    python scripts/neuralbench_config.py [--force]
"""
import argparse
import getpass
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIG = Path("~/.neuralbench/config.json").expanduser()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="overwrite an existing config")
    args = parser.parse_args()

    if CONFIG.exists() and not args.force:
        print(f"{CONFIG} already exists, leaving it as is (use --force to overwrite):")
        print(CONFIG.read_text())
        raise SystemExit(0)

    user = getpass.getuser()
    config = {
        "USER": user,
        "ENTITY_NAME": user,
        "PROJECT_NAME": "neuralbench",
        "DATA_DIR": str(REPO / "data"),
        "CACHE_DIR": str(REPO / "cache"),
        "SAVE_DIR": str(REPO / "save"),
        "WANDB_HOST": "",
        "SLURM_PARTITION": "",
        "SLURM_CONSTRAINT": "",
        "N_CPUS": os.cpu_count(),
        "CLUSTER": "auto",
    }
    for key in ("DATA_DIR", "CACHE_DIR", "SAVE_DIR"):
        Path(config[key]).mkdir(parents=True, exist_ok=True)
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(config, indent=2))
    print(f"Wrote {CONFIG}")
