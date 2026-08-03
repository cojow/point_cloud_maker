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

python /home/willicon/point_cloud/py/building_extractor.py /home/willicon/point_cloud/data/walnut_all --ground-cell-size 2.0 --ground-opening-span 20.0

echo "Job finished."