#!/bin/bash
# Before submitting a dataset you haven't run before, get a starting point with:
#   python /home/willicon/point_cloud/py/estimate_resources.py /home/willicon/point_cloud/data/<project>
# After the job finishes, check this job's .out log for the "Peak memory used" /
# "Image count this run" lines auto_reconstruct.py prints at the end - note them
# against the image count, and use those real numbers to size --mem/--cpus-per-task
# next time you run a similarly-sized image set.
#SBATCH --job-name=odm_reconstruct
#SBATCH --output=reconstruct_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16              # ODM loves CPU cores, 16 is a good number
#SBATCH --mem=64G                       # OpenMVS Densification is memory hungry. NOTE: OpenSfM
                                        # does NOT see this limit - see "MEMORY" note at the bottom
                                        # before raising image resolution.
#SBATCH --time=12:00:00                 # Give it plenty of time (12 hours)

# 1. Load required modules
module load apptainer

# 2. Activate Python environment (if necessary)
source ~/miniconda3/bin/activate yolomodel_v_1

# 3. Run the pipeline
echo "Starting ODM Reconstruction pipeline..."
python /home/willicon/point_cloud/py/auto_reconstruct.py /home/willicon/point_cloud/data/fir_cir

# Notable keys in reconstruction: (see config.yml for the full list/comments)
#
#   depthmap_resolution: 1280           (stock default 640)
#     Raises the detail ceiling itself, rather than how thoroughly it's searched.
#     Scales roughly with the square of the value, per image, so this is expensive.
#
#   cleanup: true                       - after a successful run, deletes everything in the
#     project directory except images/, reconstruction.json and scene_dense.ply (the only
#     three things building_extractor.py reads). Skipped automatically if scene_dense.ply wasn't
#     produced, so a failed run stays debuggable. Enable once you trust a run's output.

echo "Pipeline finished."