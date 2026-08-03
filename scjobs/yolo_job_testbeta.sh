#!/bin/bash

#SBATCH --time=04:00:00          # Walltime (gave you a bit more time)
#SBATCH --nodes=1                # Force everything onto ONE computer
#SBATCH --ntasks=1               # Run 1 main Python instance
#SBATCH --cpus-per-task=8        # Give PyTorch 8 CPU cores for data loading
#SBATCH --mem=64G                # 64 GB RAM Total (Safe amount)
#SBATCH --gres=gpu:2             # Request 2 GPUs on this specific node
#SBATCH -J "aerial3"             # Job name
#SBATCH --mail-user=willicon@byu.edu
#SBATCH --mail-type=BEGIN,END

# Load the environment
source ~/miniconda3/bin/activate yolomodel_v_1

# Run the script
# Note: Ensure train.py has device=[0,1] inside it
python /home/willicon/training/ultralytics_model/train.py







'''
#!/bin/bash

#SBATCH --time=02:0:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH --gpus=2
#SBATCH --mem-per-cpu=102400M   # memory per CPU core
#SBATCH -J "aerial3"   # job name
#SBATCH --mail-user=willicon@byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

source ~/miniconda3/bin/activate yolomodel_v_1
python /home/willicon/training/ultralytics_model/train.py
'''

