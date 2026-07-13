"""
estimate_resources.py

Pre-flight SLURM resource sizing helper. Run this on the login node (no
allocation needed - it just reads file headers and directory listings, takes
a second or two) BEFORE submitting a job, to get a starting point for
--cpus-per-task / --mem / --time for a dataset you haven't run before.

This gives a *rough, conservative* first estimate based on the structure of
the pipeline code, not a measured benchmark - there is no way to know your
cluster's actual throughput without running on it. The reliable long-term
source of truth is the "Peak memory used" / "Total elapsed" lines that
auto_reconstruct.py and extract_buildings.py print at the end of every real
run. Keep a note of those per dataset size, and use them (not this script) to
refine your numbers once you have 2-3 real data points of a similar size.

Run using: python py/estimate_resources.py data/900EBlock
"""
import os
import sys
import glob
import math
import argparse
from PIL import Image

PLY_TYPE_SIZES = {
    'char': 1, 'uchar': 1, 'int8': 1, 'uint8': 1,
    'short': 2, 'ushort': 2, 'int16': 2, 'uint16': 2,
    'int': 4, 'uint': 4, 'int32': 4, 'uint32': 4, 'float': 4, 'float32': 4,
    'double': 8, 'float64': 8,
}


def read_ply_vertex_info(ply_path):
    """Parses just the PLY header (a few KB at most) to get the vertex count
    and on-disk bytes-per-vertex, without loading any point data - safe to run
    on a multi-GB file from a login node."""
    vertex_count = None
    bytes_per_vertex = 0
    in_vertex_element = False
    with open(ply_path, 'rb') as f:
        while True:
            raw = f.readline()
            if not raw:
                break
            line = raw.decode('ascii', errors='ignore').strip()
            if not line or line == 'end_header':
                break
            parts = line.split()
            if parts[:2] == ['element', 'vertex']:
                vertex_count = int(parts[2])
                in_vertex_element = True
                continue
            if parts[0] == 'element':
                in_vertex_element = False
                continue
            if in_vertex_element and parts[0] == 'property':
                bytes_per_vertex += PLY_TYPE_SIZES.get(parts[1], 4)
    return vertex_count, bytes_per_vertex


def _round_up(value, step):
    return int(math.ceil(value / step) * step)


def print_sbatch_block(cpus, mem_gb, time_est):
    print(f"  #SBATCH --nodes=1")
    print(f"  #SBATCH --ntasks=1")
    print(f"  #SBATCH --cpus-per-task={cpus}")
    print(f"  #SBATCH --mem={mem_gb}G")
    print(f"  #SBATCH --time={time_est}")


def estimate_extraction_resources(project_path):
    ply_path = os.path.join(project_path, 'scene_dense.ply')
    if not os.path.exists(ply_path):
        return False

    print("\n=== extract_buildings.py (building extraction) ===")
    n, bpv = read_ply_vertex_info(ply_path)
    if n is None:
        print(f"  [!] Could not read a vertex count from the header of {ply_path}")
        return True

    raw_data_gb = (n * bpv) / (1024 ** 3)
    # Structural estimate: several float64 xyz/color arrays stay alive
    # simultaneously across pipeline stages (raw cloud, above-ground subset,
    # pruned subset, plus per-candidate blocks handed to parallel workers),
    # on top of KDTree/DBSCAN overhead. The x30 multiplier is a conservative
    # rule of thumb, NOT a measured number for this codebase - refine it using
    # the "Peak memory used" line extract_buildings.py prints at the end of a
    # real run on a similarly-sized cloud.
    safety_multiplier = 30
    est_mem_gb = raw_data_gb * safety_multiplier
    recommended_mem_gb = max(16, _round_up(est_mem_gb, 8))

    if n < 5_000_000:
        cpus, time_est = 8, "00:30:00"
    elif n < 20_000_000:
        cpus, time_est = 12, "01:30:00"
    elif n < 60_000_000:
        cpus, time_est = 16, "03:00:00"
    else:
        cpus, time_est = 24, "05:00:00"

    print(f"  scene_dense.ply: {n:,} points, {bpv} bytes/vertex on disk ({raw_data_gb:.2f} GB raw)")
    print(f"  Rough peak memory estimate: ~{est_mem_gb:.1f} GB (raw size x{safety_multiplier} structural multiplier)")
    print("\n  Suggested SBATCH header:")
    print_sbatch_block(cpus, recommended_mem_gb, time_est)
    print("\n  NOTE: conservative first-guess numbers, not a measured benchmark. After the run,")
    print("  check the job's .out log for the 'Peak memory used' and 'Total elapsed' lines")
    print("  extract_buildings.py prints, and use those real numbers next time you run a")
    print("  similarly-sized point cloud.")
    return True


def estimate_reconstruction_resources(project_path):
    image_dir = os.path.join(project_path, 'images')
    if not os.path.isdir(image_dir):
        image_dir = project_path
    files = (glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.JPG")) +
             glob.glob(os.path.join(image_dir, "*.tif")) + glob.glob(os.path.join(image_dir, "*.tiff")))
    if not files:
        return False

    print("\n=== auto_reconstruct.py (photogrammetry reconstruction) ===")
    n_images = len(files)
    total_bytes = sum(os.path.getsize(f) for f in files)
    avg_mb = (total_bytes / n_images) / (1024 ** 2)

    resolution_str = ""
    try:
        with Image.open(files[0]) as im:
            w, h = im.size
            resolution_str = f", sample resolution {w}x{h}"
    except Exception:
        pass

    print(f"  {n_images} images found in {image_dir}, average {avg_mb:.1f} MB each{resolution_str}")

    # These are simple image-count TIERS, not a computed formula. Reconstruction
    # memory/time depends heavily on scene overlap, matching neighbor count and
    # resolution settings that this script has no visibility into - treat this
    # as a rough floor with wide error bars, not a precise prediction.
    if n_images < 200:
        cpus, mem_gb, time_est = 8, 32, "02:00:00"
    elif n_images < 800:
        cpus, mem_gb, time_est = 16, 64, "06:00:00"
    elif n_images < 2000:
        cpus, mem_gb, time_est = 24, 96, "12:00:00"
    else:
        cpus, mem_gb, time_est = 32, 128, "20:00:00"

    print("\n  Suggested SBATCH header:")
    print_sbatch_block(cpus, mem_gb, time_est)
    print("\n  NOTE: reconstruction resource needs are much harder to predict than extraction -")
    print("  they depend on scene overlap/matching settings, not just image count. Treat this as")
    print("  a rough floor, and lean heavily on the 'Peak memory used' / 'Total elapsed' lines")
    print("  from your own past auto_reconstruct.py runs once you have a few of a similar size.")
    return True


def main(project_path):
    project_path = os.path.abspath(project_path)
    print(f"Inspecting: {project_path}")

    found_any = False
    found_any |= estimate_reconstruction_resources(project_path)
    found_any |= estimate_extraction_resources(project_path)

    if not found_any:
        print("\n[!] Found neither an images/ directory nor a scene_dense.ply under this path.")
        print("    Point this at a project directory that has one or both.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rough pre-flight SLURM resource sizing for a point-cloud-maker project directory."
    )
    parser.add_argument("project_path", type=str, help="Path to the project directory (e.g. data/900EBlock).")
    args = parser.parse_args()
    main(args.project_path)
