from pathlib import Path

import yaml
from ultralytics import YOLO

from yolo_paths import RUNS_DIR, SETS_DIR, resolve_data_config

#Everything you normally change (which run to validate, which images, device, ...) is set in config_val.yaml
CONFIG = Path(__file__).with_name("config_val.yaml")
with open(CONFIG, "r") as f:
    cfg = yaml.safe_load(f)

name = cfg["image_set"]
model = YOLO(str(RUNS_DIR / cfg["model"]["run"] / "weights" / "best.pt"))

#Results go in RUNS_DIR. validation_args from config_val.yaml come second, so they can override the name or project folder if needed.
results = model.val(data= resolve_data_config(CONFIG, SETS_DIR / name) #Uses validation congfig.
                    ,**{"name": f"{name}_validation", "project": str(RUNS_DIR), **(cfg.get("validation_args") or {})})
