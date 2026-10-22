from pathlib import Path

import yaml
from ultralytics import YOLO

from yolo_paths import DATASET_DIR, RUNS_DIR, resolve_data_config

#Everything you normally change (new or continued model, model size, epochs, GPUs, ...) is set in config.yaml
CONFIG = Path(__file__).with_name("config.yaml")
with open(CONFIG, "r") as f:
    cfg = yaml.safe_load(f)

#load a model
if cfg["model"]["new_model"]:
    model = YOLO(cfg["model"]["architecture"])
else:
    model = YOLO(str(RUNS_DIR / cfg["model"]["resume_run"] / "weights" / "last.pt"))      #last.pt of the run being continued

#train the model
#Run folders (train, train2, ...) are created in RUNS_DIR, so weights end up in data/yolo/runs/<run>/weights.
#training_args from config.yaml come second, so they can override the project folder if needed.
results = model.train(data = resolve_data_config(CONFIG, DATASET_DIR)    #Dataset section of config.yaml: training and validation images, as well as the classes
                      ,**{"project": str(RUNS_DIR), **(cfg.get("training_args") or {})})
