#!/bin/bash
# SBATCH --job-name=house_extraction # Name of your job
#SBATCH --output=house_%j.out       # Standard output and error log (%j becomes the job ID)
#SBATCH --nodes=1                        # Run all processes on a single node	
#SBATCH --ntasks=1                       # Run a single task...
#SBATCH --cpus-per-task=12                # ...but allocate 6 CPUs (cores) to that task for multiprocessing
#SBATCH --mem=180G                        # Request 32 GB of RAM (adjust if needed)
#SBATCH --time=04:00:00                  # Maximum time limit (HH:MM:SS)


# --- Environment Setup ---
# Load the environment
source ~/miniconda3/bin/activate yolomodel_v_1

# --- Run the Script ---
# Make sure your python script name matches what you uploaded.
echo "Starting Open3D extraction job..."

# Using local geotif (Update File Path)
#python /home/willicon/point_cloud/py/ex_building_elev.py /home/willicon/point_cloud/data/walnut_all --ground-mode local --dem-file /path/to/project/local_dem.tif
# Using USGS elevations (Update File Path)
python /home/willicon/point_cloud/py/ex_building_elev.py /home/willicon/point_cloud/data/walnut_all --ground-mode usgs
# Guessing ground from point cloud (Update File Path)
#python /home/willicon/point_cloud/py/ex_building_elev.py /home/willicon/point_cloud/data/walnut_all

echo "Job finished."