#!/bin/bash
#SBATCH --job-name=odm_reconstruct
#SBATCH --output=reconstruct_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16              # ODM loves CPU cores, 16 is a good number
#SBATCH --mem=64G                       # OpenMVS Densification is memory hungry
#SBATCH --time=12:00:00                 # Give it plenty of time (12 hours)

# 1. Load required modules
module load apptainer

# 2. Activate Python environment (if necessary)
# source /path/to/your/venv/bin/activate

# 3. Run the pipeline
echo "Starting ODM Reconstruction pipeline..."
python /home/willicon/point_cloud/py/auto_reconstruct.py /home/willicon/point_cloud/data/900E
echo "Pipeline finished."