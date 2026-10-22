"""Shared locations for train.py, validation.py and predict.py.

Everything lives under <repo>/data/yolo (data/ is git-ignored), so the scripts
run the same wherever the repo is cloned - a laptop, or /home/<user>/point_cloud
on the cluster - without hand-editing any paths:

    data/yolo/dataset/<split>/images, labels   training data (splits named in config.yaml)
    data/yolo/sets/<name>/images, labels       image sets for validation.py / predict.py
    data/yolo/runs/<run>/weights/best.pt       output of training, validation and prediction
"""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
YOLO_DIR = REPO_ROOT / "data" / "yolo"
DATASET_DIR = YOLO_DIR / "dataset"
SETS_DIR = YOLO_DIR / "sets"
RUNS_DIR = YOLO_DIR / "runs"

# Keys in config.yaml / config_val.yaml that train.py and validation.py read themselves, not part of the dataset.
SCRIPT_KEYS = ("model", "training_args", "validation_args", "image_set")


def resolve_data_config(config_path, dataset_dir):
    """Ultralytics resolves a relative `path:` against the current directory (or
    its own datasets_dir setting), not the YAML's location, so a relative path in
    config.yaml only works from one working directory. This writes a copy of
    config_path with `path:` set to the absolute dataset_dir and returns the
    copy's path, for passing as `data=`. Settings that only the scripts read
    (SCRIPT_KEYS) are left out of the copy."""
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    for key in SCRIPT_KEYS:
        cfg.pop(key, None)
    cfg["path"] = str(dataset_dir)

    resolved_dir = YOLO_DIR / "resolved_configs"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = resolved_dir / config_path.name
    with open(resolved_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return str(resolved_path)
