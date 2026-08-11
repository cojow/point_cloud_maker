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
# Densification tuning levers (all optional - omitting a flag leaves that
# setting at whatever's already in config.yaml/OpenSfM's own default, which
# may be a leftover override from a PREVIOUS experiment on this same project
# directory - if you're reusing a directory rather than copying it fresh per
# experiment, pass every flag you care about explicitly rather than relying
# on "unset = stock default").
#
#
# CURRENTLY TESTING (active command above) - raises the detail ceiling itself, rather
# than how thoroughly that ceiling gets searched. Scales roughly with the square of the
# value, per image, so this is the expensive one:
#   --depthmap-resolution 1280          (stock default 640)
#     CONFIRMED WORKING: 640 -> 8,565,711 points; 1280 -> 33,150,527 points (~4x, i.e.
#     the expected resolution-squared scaling). Cost ~22 min of the ~36 min run on 240 pictures.
#
#   --matching-gps-neighbors 25         (stock default 0 = no cap)
#     Only image pairs among each image's 25 nearest neighbours by GPS
#     get matched. match_features is the pipeline's only O(n^2) stage and it was doing
#     almost all of its work for nothing: the 245-image run attempted 17,858 pairs and
#     only 1,839 (10.3%) produced a usable match, burning ~5.98 min. Capping neighbours
#     makes the pair count grow linearly with image count instead of quadratically, which
#     is what makes larger image sets practical at all.
#
#     HOW TO TELL IF IT WORKED - two lines in this job's .out:
#       "Matching N image pairs"        should drop well below 17,858
#       "Reconstruction 0: N images"    should STAY around 211 (it was 211/245 before)
#     If the second number drops meaningfully, the cap is too tight - raise 25 to 40 and
#     rerun. Registering fewer images is a real quality loss; a faster match that loses
#     images is not a win.
#
#   --cleanup                           - after a successful run, deletes everything in the
#     project directory except images/, reconstruction.json and scene_dense.ply (the only
#     three things extract_buildings_floor.py reads). On the fir run that frees ~8 GB
#     (undistorted/ alone was 7.9 GB). Skipped automatically if scene_dense.ply wasn't
#     produced, so a failed run stays debuggable. Enable once you trust a run's output.
# ALSO AVAILABLE (not enabled here):
#   --matching-gps-distance <metres>    (stock default 150) - the other half of pair
#     selection, ANDed with the neighbour cap. At a ~100 m survey altitude over a
#     neighbourhood, 150 m is wide enough that nearly every image is a candidate for
#     nearly every other, which is why the neighbour cap above is the more effective knob.

#   --ascii-ply                         - scene_dense.ply is now written as binary PLY by
#     default (~1.8 GB -> ~0.9 GB, and far faster for extract_buildings_floor.py to load).
#     Only pass this if some external tool needs the old ASCII format.
#
# ----------------------------------------------------------------------------------
# MEMORY - why --mem above is not the whole story
#
# OpenSfM sizes its own work queue against the NODE's RAM, not this job's --mem. The
# .out log says so directly:
#     "Planning to use 111592.916015625 MB of RAM ..."   (~109 GB, vs the 64 GB requested)
#     "Scale-space expected size of a single image : 95.625 MB"
#     "Expecting to queue at most 200 images while parallel processing of 16 images"
# psutil/free inside the container read /proc/meminfo, which reports the physical node,
# not the SLURM cgroup - so OpenSfM cannot see the limit it will be killed for exceeding.
#
# It survives today because the queue is capped at 200 images: 200 x 95.6 MB = ~19 GB,
# comfortably inside 64 GB (measured peak was 5.9 GB). The exposure scales with IMAGE
# RESOLUTION, not image count, because that 95.6 MB is per-image scale-space size. Shoot
# images with ~4x the pixels and it becomes 200 x ~380 MB = ~76 GB > 64 GB -> OOM kill.
#
# So before moving to a higher-resolution camera, read "Scale-space expected size of a
# single image" out of the .out log and multiply by 200. If that exceeds --mem, either
# raise --mem or lower --cpus-per-task (fewer parallel workers = less concurrent memory).
# ----------------------------------------------------------------------------------

echo "Pipeline finished."