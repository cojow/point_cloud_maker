import os
import yaml

# --- 0. CONFIG ---
# Shared by auto_reconstruct.py (reconstruction) and building_extractor.py
# (ground_model/vehicle_rejection/imagery/performance).
DEFAULT_CONFIG = {
    "ground_model": {
        "cell_size": 2.0,
        "opening_span": 20.0,
        "relevel": False,
    },
    "vehicle_rejection": {
        "reject_small_structures": True,
        "max_area_sqft": 800.0,
        "max_height_ft": 16.0,
    },
    "imagery": {
        "nadir_pitch_threshold": -65.0,
    },
    "performance": {
        "ransac_max_fit_points": 50000,
    },
    "reconstruction": {
        "depthmap_method": None,
        "depthmap_resolution": None,
        "depthmap_min_consistent_views": None,
        "depthmap_min_patch_sd": None,
        "depthmap_num_neighbors": None,
        "depthmap_num_matching_views": None,
        "matching_gps_neighbors": None,
        "matching_gps_distance": None,
        "depthmap_nadir_only": False,
        "nadir_pitch_threshold": -60.0,
        "ascii_ply": False,
        "cleanup": False,
    },
}

def load_config(config_path):
    """Loads config.yml, filling in anything missing (or the whole file, if
    the path doesn't exist) from DEFAULT_CONFIG - a partial or absent config
    is not an error, just uses the built-in default for whatever it doesn't
    specify, section by section."""
    cfg = {section: dict(values) for section, values in DEFAULT_CONFIG.items()}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_cfg = yaml.safe_load(f) or {}
        for section, values in user_cfg.items():
            if section in cfg and isinstance(values, dict):
                cfg[section].update(values)
            else:
                cfg[section] = values
    else:
        print(f"      [!] No config file found at {config_path} - using built-in defaults for every setting.")
    return cfg
