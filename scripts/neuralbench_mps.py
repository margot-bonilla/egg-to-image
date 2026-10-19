"""`neuralbench` CLI on a Mac: same arguments, DINOv2 targets and training on the Apple GPU.

    python scripts/neuralbench_mps.py eeg image -m eegnet [--seeds 33,34]

- Applies scripts/mps_patch.py (HuggingFace extractors on ``mps``; cache keys unchanged).
- Runs everything in this process: writes configs/neuralbench.local.json, a copy of
  ~/.neuralbench/config.json with ``CLUSTER: null`` (otherwise exca spawns a
  local subprocess where the patch does not apply), and points NEURALBENCH_CONFIG at it.
- Keeps Hugging Face downloads inside the repo (cache/huggingface).
- Lightning picks ``mps`` for training on its own (accelerator="auto").
- Copies each run's ``best.ckpt`` to ``best_kept.ckpt`` (same folder) before NeuralBench
  deletes it, so it can be packaged for Codabench. The run config, and so its cache
  key, stays the canonical one.
"""
import json
import logging
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCAL_CONFIG = REPO / "configs" / "neuralbench.local.json"


def write_local_config() -> None:
    user_config = Path("~/.neuralbench/config.json").expanduser()
    config = json.loads(user_config.read_text()) if user_config.exists() else {}
    config.setdefault("USER", os.environ.get("USER", "user"))
    config.setdefault("ENTITY_NAME", config["USER"])
    config.setdefault("PROJECT_NAME", "neuralbench")
    for key in ("DATA_DIR", "CACHE_DIR", "SAVE_DIR"):
        config.setdefault(key, str(REPO / key.split("_")[0].lower()))
    config.update(CLUSTER=None, N_CPUS=os.cpu_count(), WANDB_HOST=config.get("WANDB_HOST", ""),
                  SLURM_PARTITION="", SLURM_CONSTRAINT="")
    LOCAL_CONFIG.write_text(json.dumps(config, indent=2))


def keep_best_checkpoints() -> None:
    from neuralbench.main import Experiment

    cleanup = Experiment._cleanup

    def cleanup_keeping_best(self, trainer) -> None:
        best = getattr(trainer.checkpoint_callback, "best_model_path", None)
        if best and Path(best).exists():
            kept = Path(best).with_name("best_kept.ckpt")
            shutil.copyfile(best, kept)
            logging.getLogger(__name__).warning("Kept best checkpoint: %s", kept)
        cleanup(self, trainer)

    Experiment._cleanup = cleanup_keeping_best


def override_seed_grid(seeds: list[int]) -> None:
    """Replace NeuralBench's default seed grid (33, 34, 35) with ``seeds``."""
    from neuralbench import experiment_config

    load = experiment_config.load_yaml_config

    def load_with_seeds(path, *args, **kwargs):
        config = load(path, *args, **kwargs)
        if Path(path) == experiment_config.DEFAULTS_DIR / "grid.yaml":
            config["seed"] = seeds
        return config

    experiment_config.load_yaml_config = load_with_seeds


if __name__ == "__main__":
    write_local_config()
    os.environ["NEURALBENCH_CONFIG"] = str(LOCAL_CONFIG)
    os.environ.setdefault("HF_HOME", str(REPO / "cache" / "huggingface"))

    sys.path.insert(0, str(REPO / "scripts"))
    import mps_patch

    mps_patch.apply()
    keep_best_checkpoints()
    if "--seeds" in sys.argv:
        i = sys.argv.index("--seeds")
        seeds = [int(s) for s in sys.argv[i + 1].split(",")]
        del sys.argv[i:i + 2]
        override_seed_grid(seeds)

    from neuralbench import run_benchmark_cli

    sys.argv[0] = "neuralbench"
    run_benchmark_cli()
