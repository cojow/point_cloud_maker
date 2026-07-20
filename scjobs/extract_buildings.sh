#!/bin/bash
# Before submitting a dataset you haven't run before, get a starting point with:
#   python /home/willicon/point_cloud/py/estimate_resources.py /home/willicon/point_cloud/data/<project>
# After the job finishes, check this job's .out log for the "Peak memory used" /
# "CPU cores used" / "Point cloud size this run" lines extract_buildings.py prints
# at the end - note them against the point count, and use those real numbers to
# size --mem/--cpus-per-task next time you run a similarly-sized point cloud.
# SBATCH --job-name=house_extraction # Name of your job
#SBATCH --output=house_%j.out       # Standard output and error log (%j becomes the job ID)
#SBATCH --nodes=1                        # Run all processes on a single node
#SBATCH --ntasks=1                       # Run a single task...
#SBATCH --cpus-per-task=12                # ...but allocate 6 CPUs (cores) to that task for multiprocessing
#SBATCH --mem=180G                        # NOTE: this was manually bumped up at some point and the comment
                                           # never got updated - re-check against estimate_resources.py / a
                                           # past run's "Peak memory used" line rather than trusting this value
#SBATCH --time=04:00:00                  # Maximum time limit (HH:MM:SS)


# --- Environment Setup ---
# Load the environment
source ~/miniconda3/bin/activate yolomodel_v_1

# --- Run the Script ---
# Make sure your python script name matches what you uploaded.
echo "Starting Open3D extraction job..."

# extract_buildings.py builds on the verified-working extract_green_buffer.py
# pipeline, with the flat single-plane ground swapped for a local contour
# ground model (built straight from the point cloud), so buildings scattered
# across a sloped/uneven flight area each get their own correct floor. Tune
# --ground-opening-span upward if it's still larger than your biggest
# building's footprint (in meters) and roofs are leaking through as ground.
# ex_building_elev.py is retired - do not use, isolation is broken in it.
#
# KNOWN ISSUE: the synthetic floor (cut-to-lowest-point-under-footprint) is
# currently landing way too low on some buildings. Try
# extract_buildings_no_floor.py below instead - despite the filename it does
# add a floor, just the MEDIAN ground-model elevation under the footprint
# (same ground_surface.ply diagnostic) instead of the minimum.
python /home/willicon/point_cloud/py/extract_buildings.py /home/willicon/point_cloud/data/walnut_all --ground-cell-size 2.0 --ground-opening-span 20.0

# Median-floor variant (try this until the min-based floor above is fixed):
#python /home/willicon/point_cloud/py/extract_buildings_no_floor.py /home/willicon/point_cloud/data/walnut_all --ground-cell-size 2.0 --ground-opening-span 20.0

echo "Job finished."