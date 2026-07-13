# Point Cloud Maker

{THIS IS AN AI GENERATED FIRST DRAFT. HEAVY REVISIONS ARE NEEDED. TAKE WITH A GRAIN OF SALT}

Turns photo sets (taken from drone images) into georeferenced point
clouds. Individual buildings are extracted from those point clouds with
measurements (area, volume, height) and a cropped photo for each one.

[DJI Flight Planner](https://github.com/cojow/flight_planner): For those using DJI drones, this is an free open-source flight planner desinged to be used with this code. 

## Running the pipeline
### 1. Install dependecies 

```
pip install -r requirements.txt
```

You'll also need a container engine:
- **macOS / local dev**: [Docker](https://www.docker.com/) (uses the
  `opendronemap/odm:latest` image, pulled automatically on first run)
- **Linux / supercomputer**: [Apptainer](https://apptainer.org/)

```
apptainer pull odm.sif docker://opendronemap/odm:latest
```

> **Note:** [`py/auto_reconstruct.py`](py/auto_reconstruct.py) hardcodes the
> path to `odm.sif` (`APPTAINER_IMAGE`) for the Apptainer/Linux case, since
> that path lives outside this repo and differs per machine/account. If
> you're running on a different account or machine, update that path at the
> top of the file before running.



### 1. Reconstruction — photos to point cloud

```
python py/auto_reconstruct.py data/<project_name>
```

Put your drone images (and `.MRK` RTK files, if you have them) in
`data/<project_name>/`. This will:
- Sort images into a `images/` subfolder
- Inject precise RTK GPS from `.MRK` files into the photo metadata, if present
- Run the OpenSfM reconstruction pipeline (feature detection, matching,
  reconstruction, undistortion, depthmap densification)
- Produce `data/<project_name>/scene_dense.ply`

### 2. Extraction — point cloud to buildings

```
python py/ex_building_elev.py data/<project_name> --ground-mode ransac
```

`--ground-mode` selects how "true ground elevation" is determined:
- `ransac` (default) — fits a single global ground plane to the point cloud
- `usgs` — looks up ground elevation from an external elevation API per
  building *(currently uses Open-Elevation, not the USGS EPQS endpoint it's
  named after — this mode is a known work-in-progress, see below)*
- `local` — samples a local GeoTIFF DEM you provide with `--dem-file`

This produces, under `data/<project_name>/analysis_<project_name>_v5/`:
- `measurements.csv` — area (sqft), volume (cuft), height (ft) per building
- `location_lookup.csv` — UTM + lat/lon per building
- `individual_houses/` — per-building `.ply` point clouds
- `best_images/` and `best_images_cropped/` — the nearest source photo for
  each building, and a version cropped to just that building

## Known issues / in progress

- `--ground-mode usgs` doesn't yet query USGS; it calls Open-Elevation
  instead. Needs to be pointed at the real USGS EPQS endpoint.

## TODO (fill in as the project develops)

- [ ] Describe the flight planner tool and how images are captured
- [ ] Document the expected `.MRK` file format
- [ ] Add example project / sample data instructions
