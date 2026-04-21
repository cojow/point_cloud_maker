#!/bin/bash
# SBATCH --job-name=house_extraction_test # Name of your job
#SBATCH --output=house_test_%j.out       # Standard output and error log (%j becomes the job ID)
#SBATCH --nodes=1                        # Run all processes on a single node	
#SBATCH --ntasks=1                       # Run a single task...
#SBATCH --cpus-per-task=6                # ...but allocate 6 CPUs (cores) to that task for multiprocessing
#SBATCH --mem=32G                        # Request 32 GB of RAM (adjust if needed)
#SBATCH --time=04:00:00                  # Maximum time limit (HH:MM:SS)

# --- Environment Setup ---
# Load the environment
source ~/miniconda3/bin/activate yolomodel_v_1

# --- Run the Script ---
# Make sure your python script name matches what you uploaded.
echo "Starting Open3D extraction job..."
python /home/willicon/point_cloud/py/extract_buildings.py /home/willicon/point_cloud/data/900E
echo "Job finished."