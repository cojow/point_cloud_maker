# Point Cloud Maker

Turns photo sets (taken from drone images) into georeferenced point
clouds. Individual buildings are extracted from those point clouds with
measurements (area, volume, height) and a cropped photo.

[DJI Flight Planner](https://github.com/cojow/flight_planner): For those using DJI drones, this is a free open-source flight planner designed to be used with this code.

The pipeline is two independent scripts, run one after the other on the same project folder:

1. [`py/auto_reconstruct.py`](py/auto_reconstruct.py) — photos → point cloud (`scene_dense.ply`)
2. [`py/building_extractor.py`](py/building_extractor.py) — point cloud → per-building measurements + photos

> For the full BYU supercomputer account/directory setup and Slurm reference, see
> [`documentation/Point_Cloud_Maker.qmd`](documentation/Point_Cloud_Maker.qmd). This README covers the
> scripts' inputs and outputs.

## Setup

#### Clone this repository
Simple as that. Save it somewhere you'll remember.

#### Python environment
Create and activate a Python environment, then install dependencies:

If using Conda: 
```Bash
conda create -n point_cloud python=3.11
conda activate point_cloud
pip install -r requirements.txt
```
The real job scripts in [`scjobs/`](scjobs) assume a conda env named `yolomodel_v_1` and activate it with
`source ~/miniconda3/bin/activate yolomodel_v_1`.
Edit the `source` line in each script to match your env name.

#### Container engine (reconstruction stage only)
`auto_reconstruct.py` uses [OpenDroneMap](https://github.com/opendronemap/ODM)'s depth-densification code,
which runs inside a container. 
It picks the engine automatically based on OS.
**Docker** for Mac and Windows,
**Apptainer** on Linux 
`building_extractor.py` needs neither; it's pure Python.

- **Docker**: nothing to install beyond Docker itself — the `opendronemap/odm:latest` image is pulled
  automatically on first run.
- **Apptainer** (Linux): load the module, then pull the image and convert it to a sandbox (a plain,
  uncompressed directory that is required because some compute nodes can't mount a `.sif` file directly):
  ```
  module load apptainer
  apptainer pull odm.sif docker://opendronemap/odm:latest
  apptainer build --sandbox odm_sandbox odm.sif
  ```
  Then edit `APPTAINER_IMAGE` near the top of [`py/auto_reconstruct.py`](py/auto_reconstruct.py) (currently
  hardcoded to `/home/willicon/point_cloud/odm_sandbox`) to point at wherever you built `odm_sandbox`. 
  This path lives outside the repo and is different per machine/account.

## Running the pipeline

Both scripts take a **project directory** as their only required argument — a folder holding one drone
survey's images, e.g. `data/900EBlock/`.

### 1. Reconstruction: photos → point cloud

```Bash
python py/auto_reconstruct.py data/<project_name>
```

Put your drone images directly in `data/<project_name>/` first (any loose `.jpg`/`.jpeg`/`.tif`/`.tiff` files
get sorted into an `images/` subfolder automatically). This runs the OpenSfM pipeline (feature detection,
matching, reconstruction, undistortion, depthmap densification) inside the container, then converts the
result to a binary PLY. It produces:

- `data/<project_name>/images/`: your photos, sorted
- `data/<project_name>/reconstruction.json`: camera poses (read by `building_extractor.py` later for
  photo cropping)
- `data/<project_name>/scene_dense.ply`: the point cloud, the main deliverable of this stage

#### Configuration

All tuning lives in [`py/config.yml`](py/config.yml)'s `reconstruction:` section.
Edit it directly or pass `--config /path/to/your.yml` to use a separate config per project instead of editing the shared one in place. 
This is the **same file** `building_extractor.py` reads by default (see below). 
Any key left out (or set to `null`) falls back to whatever OpenSfM's own stock default (or whatever's already in that project's `config.yaml`) is.

| Key | Default | What it controls |
|---|---|---|
| `reconstruction.depthmap_resolution` | `null` (OpenSfM default 640) | Depth estimation resolution per image. The single biggest lever on point cloud detail — cost scales roughly with the square of this value. |
| `reconstruction.matching_gps_neighbors` | `null` (OpenSfM default 0 = uncapped) | Match each image only against its N nearest neighbors by GPS. Feature matching is the only O(n²) stage — capping this is the main way to keep large flights from scaling quadratically. Try 20–30 for a normal-overlap mapping flight. |
| `reconstruction.matching_gps_distance` | `null` (OpenSfM default 150 m) | Max GPS distance between two images to be considered a candidate pair. Applied together with the neighbor cap. |
| `reconstruction.depthmap_min_consistent_views` | `null` (OpenSfM default 3) | Minimum agreeing neighbor views before a depth estimate is kept. Lower = more points, more noise. |
| `reconstruction.depthmap_min_patch_sd` | `null` (OpenSfM default ~1.0) | Minimum texture required to attempt a depth estimate. Lower helps low-texture surfaces like roofs. |
| `reconstruction.depthmap_num_neighbors` | `null` (OpenSfM default 10) | Neighboring images considered per shot during depth estimation. |
| `reconstruction.depthmap_num_matching_views` | `null` (OpenSfM default 5) | Matching views used per depth estimate. |
| `reconstruction.depthmap_nadir_only` | `false` | For a two-flight setup (near-nadir pass for the point cloud + a separate oblique pass shot for something else, e.g. building photos). Skips densification for the non-nadir shots — they still get camera poses, just not depthmaps — since densification is the single most expensive stage. Requires DJI gimbal XMP telemetry. |
| `reconstruction.nadir_pitch_threshold` | `-60.0` | Gimbal pitch cutoff for `depthmap_nadir_only` (DJI convention: -90 = straight down). Not the same setting as `imagery.nadir_pitch_threshold` below — that one is `building_extractor.py`'s nadir/oblique photo-crop split. |
| `reconstruction.ascii_ply` | `false` | Keep `scene_dense.ply` as ASCII instead of converting to binary. Binary is smaller (~half the size) and is what `building_extractor.py` expects to load quickly — only set this if an external tool needs ASCII. |
| `reconstruction.cleanup` | `false` | After a successful run, delete everything in the project folder except `images/`, `reconstruction.json`, and `scene_dense.ply` (removes OpenSfM's working state, usually much larger than what's kept). Skipped if the run failed, so a failed run's intermediates aren't lost. |

See [`scjobs/auto_reconstruct.sh`](scjobs/auto_reconstruct.sh) for a real Slurm job using several of these settings.

### 2. Extraction: point cloud → buildings

```
python py/building_extractor.py data/<project_name>
```

Reads `scene_dense.ply`, `reconstruction.json`, and `images/` from the same project folder the reconstruction
stage wrote to. 
Isolates ground-level clutter and vegetation from a local ground-contour model, groups the remaining points into individual buildings, fits a synthetic floor, and finds the best source photo (nadir and oblique) for each one. 
Produces, in `data/<project_name>/analysis_<project_name>_v9/`:

- `measurements.csv`: one row per building: `Area_sqft`, `Volume_cuft`, `Height_ft`, UTM + lat/lon location, best/original image filenames for both the nadir and oblique view, whether each was a verified precision crop, and two review flags — `Needs_Review` (likely vegetation contamination) and `Possible_Vehicle` (small/short enough it might be a parked car rather than a building)
- `individual_houses/`: per-building `.ply` point clouds
- `best_images/` and `best_images_cropped/`: source photos and crops for each building
- `diagnostics/ground_surface.ply`: the ground-contour model used to separate buildings from terrain

#### Configuration

All tuning lives in the same [`py/config.yml`](py/config.yml) used by the reconstruction stage above (see its
`ground_model`/`vehicle_rejection`/`imagery`/`performance` sections).
 Any section or key left out falls back to the default shown below.

| Key | Default | What it controls |
|---|---|---|
| `ground_model.cell_size` | `2.0` (m) | Grid resolution of the local ground-contour model. Smaller = more detail but noisier. |
| `ground_model.opening_span` | `20.0` (m) | Must be wider than a normal building's footprint, or the ground model misreads roofs as ground. |
| `vehicle_rejection.reject_small_structures` | `True` | Drops `Possible_Vehicle` candidates from the output entirely instead of just flagging them. No geometric test can reliably tell a parked car apart from a small legitimate structure (shed, small garage) — turning this on accepts losing some real small buildings in exchange for not having to manually discard cars from `measurements.csv`. |
| `vehicle_rejection.max_area_sqft` / `max_height_ft` | `800.0` / `16.0` | A candidate under **both** thresholds is flagged `Possible_Vehicle`. Calibrated against one real dataset — a site-specific starting point, not a universal threshold. |
| `imagery.nadir_pitch_threshold` | `-65.0` | DJI gimbal pitch cutoff used to split source photos into nadir vs. oblique for the two best-image crops. Not universal — check the "Indexed N nadir + M oblique" line in a run's log and adjust per site. |
| `performance.ransac_max_fit_points` | `50000` | Caps how many points a single RANSAC roof-plane fit searches against. Lower = faster on messy/oversized candidates, untested below the low tens of thousands. |

See [`scjobs/extract_buildings.sh`](scjobs/extract_buildings.sh) for a real Slurm job (note it does **not**
load Apptainer — this stage never touches the container).

### Utilities

- [`py/estimate_resources.py`](py/estimate_resources.py) — pre-flight helper for sizing Slurm `--cpus-per-task`/`--mem` requests before submitting a job.
- [`py/view_log.py`](py/view_log.py) — follow/inspect a running or finished job's log output.

### Not part of this pipeline

[`py/yolo_training/`](py/yolo_training), [`py/random_forest_model/`](py/random_forest_model), and
[`scjobs/yolo_job_testbeta.sh`](scjobs/yolo_job_testbeta.sh) are a separate, experimental object-detection
effort (likely meant to consume the cropped building photos this pipeline produces) — neither
`auto_reconstruct.py` nor `building_extractor.py` depends on them, and the job script itself points at a
training script outside this repo. `ultralytics` and `joblib` in `requirements.txt` are commented out and only
needed if you're working with those.

## Known issues / in progress

- Best images and best cropped images are still a work in progress. Code works well if the picture is looking at the house from plan view, but not as well from the front of the house.
- House segmenting still has problems, but works pretty well. 
