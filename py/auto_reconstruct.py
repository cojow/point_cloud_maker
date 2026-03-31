import os
import subprocess
import json
import glob
import re
import sys
import shutil
import platform

'''
 Run using python py/auto_reconstruct.py data/"""foldername"""

'''

def is_mac():
    """Detects if the script is running on macOS (Apple Silicon/Intel)."""
    return platform.system().lower() == "darwin"

def run_docker_command(image, command_list, project_path, work_dir_suffix="", entrypoint=None, extra_docker_args=None):
    """Spins up a targeted Docker container for a specific task."""
    work_dir = os.path.join(project_path, work_dir_suffix) if work_dir_suffix else project_path
    
    docker_cmd = [
        "docker", "run", "-i", "--rm",
        "-v", f"{project_path}:{project_path}",
        "-w", work_dir
    ]
    
    if extra_docker_args:
        docker_cmd.extend(extra_docker_args)
        
    if entrypoint:
        docker_cmd.extend(["--entrypoint", entrypoint])
        
    docker_cmd.append(image)
    docker_cmd.extend(command_list)
    
    print(f"Executing Docker Engine [{image}]: {' '.join(docker_cmd)}")
    result = subprocess.run(docker_cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error executing Docker engine: {image}")
        sys.exit(1)

def get_odm_opensfm_path():
    """Probes the ODM container for the OpenSfM executable."""
    print("--- Probing ODM container for OpenSfM executable ---")
    cmd = [
        "docker", "run", "--rm", "--entrypoint", "sh", 
        "opendronemap/odm:latest", "-c", 
        "find /code /usr -name opensfm -type f 2>/dev/null | grep bin | head -n 1"
    ]
    try:
        output = subprocess.check_output(cmd, text=True).strip()
        if not output:
            print("Error: Could not locate OpenSfM inside ODM container.")
            sys.exit(1)
        return output
    except Exception as e:
        print(f"Error executing Docker probe: {e}")
        sys.exit(1)

def get_odm_openmvs_path():
    """Probes the ODM container for the OpenMVS executable."""
    print("--- Probing ODM container for OpenMVS executable ---")
    cmd = [
        "docker", "run", "--rm", "--entrypoint", "sh", 
        "opendronemap/odm:latest", "-c", 
        "find /code /usr -name DensifyPointCloud -type f 2>/dev/null | head -n 1"
    ]
    try:
        output = subprocess.check_output(cmd, text=True).strip()
        if not output:
            print("Error: Could not locate DensifyPointCloud inside ODM container.")
            sys.exit(1)
        return output
    except Exception as e:
        print(f"Error executing Docker probe: {e}")
        sys.exit(1)

def organize_folders(project_path):
    images_dir = os.path.join(project_path, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    all_files = os.listdir(project_path)
    valid_extensions = ('.jpg', '.jpeg', '.tif', '.tiff')
    
    image_files = [
        os.path.join(project_path, f) for f in all_files
        if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(project_path, f))
    ]

    for img in image_files:
        if os.path.dirname(img) != images_dir:
            shutil.move(img, os.path.join(images_dir, os.path.basename(img)))
            
    print(f"Moved {len(image_files)} images into {images_dir}")

def inject_mrk_data(project_path, mrk_files):
    if not mrk_files:
        return

    exif_path = os.path.join(project_path, 'exif')
    mrk_data = {}
    
    for mrk_file in mrk_files:
        filename = os.path.basename(mrk_file)
        prefix_match = re.match(r"^(F\d)_", filename)
        prefix = prefix_match.group(1) if prefix_match else "DEFAULT"
        
        if prefix not in mrk_data:
            mrk_data[prefix] = {}
            
        with open(mrk_file, 'r') as f:
            for line in f:
                lat_match = re.search(r"([-+]?\d*\.\d+|\d+),Lat", line)
                lon_match = re.search(r"([-+]?\d*\.\d+|\d+),Lon", line)
                alt_match = re.search(r"([-+]?\d*\.\d+|\d+),Ellh", line)
                idx_match = re.match(r"^(\d+)", line.strip()) 
                
                if lat_match and lon_match and alt_match and idx_match:
                    seq_num = int(idx_match.group(1))
                    mrk_data[prefix][seq_num] = {
                        "lat": float(lat_match.group(1)), 
                        "lon": float(lon_match.group(1)), 
                        "alt": float(alt_match.group(1))
                    }
    
    json_files = glob.glob(os.path.join(exif_path, "*.json"))
    for json_file in json_files:
        filename = os.path.basename(json_file)
        prefix_match = re.match(r"^(F\d)_", filename)
        prefix = prefix_match.group(1) if prefix_match else "DEFAULT"
        seq_match = re.search(r"_(\d{4})_", filename)
        
        if seq_match:
            seq_num = int(seq_match.group(1)) 
            if prefix in mrk_data and seq_num in mrk_data[prefix]:
                with open(json_file, 'r') as f: 
                    data = json.load(f)
                
                if 'gps' not in data:
                    data['gps'] = {}
                    
                data['gps'].update({
                    'latitude': mrk_data[prefix][seq_num]['lat'], 
                    'longitude': mrk_data[prefix][seq_num]['lon'], 
                    'altitude': mrk_data[prefix][seq_num]['alt'], 
                    'dop': 0.01 
                })
                
                with open(json_file, 'w') as f: 
                    json.dump(data, f, indent=4)

def main(project_path):
    project_path = os.path.abspath(project_path)
    organize_folders(project_path)
    
    search_pattern = os.path.join(project_path, "*.MRK")
    mrk_files = [f for f in glob.glob(search_pattern) if os.path.isfile(f)]
    
    opensfm_bin = get_odm_opensfm_path()
    print(f"--- Located OpenSfM Engine at: {opensfm_bin} ---")
    
    # Phase 1: OpenSfM Pipeline via ODM Container
    steps = ["extract_metadata", "detect_features", "match_features", "create_tracks", "reconstruct", "undistort"]
    for step in steps:
        if step == "detect_features" and mrk_files: 
            inject_mrk_data(project_path, mrk_files)
        
        run_docker_command(
            image="opendronemap/odm:latest",
            command_list=[step, project_path],
            project_path=project_path,
            entrypoint=opensfm_bin
        )

    # Phase 2 & 3: Environment-Aware Densification
    if is_mac():
        print("\n--- Mac Detected: Running Safe OpenSfM Densification ---")
        run_docker_command(
            image="opendronemap/odm:latest",
            command_list=["compute_depthmaps", project_path],
            project_path=project_path,
            entrypoint=opensfm_bin
        )
        # Record where OpenSfM saves its dense cloud
        source_ply = os.path.join(project_path, 'undistorted', 'depthmaps', 'merged.ply')
        
    else:
        print("\n--- Linux Detected: Running High-Performance OpenMVS Densification ---")
        densify_bin = get_odm_openmvs_path()
        
        run_docker_command(
            image="opendronemap/odm:latest",
            command_list=["export_openmvs", project_path],
            project_path=project_path,
            entrypoint=opensfm_bin
        )
        
        linux_memory_hacks = [
            "-e", "MALLOC_CHECK_=0",     
            "-e", "OMP_NUM_THREADS=2",   
            "-e", "OPENBLAS_NUM_THREADS=2"
        ]
        
        run_docker_command(
            image="opendronemap/odm:latest",
            command_list=["--resolution-level", "2", "scene.mvs"],
            project_path=project_path,
            work_dir_suffix="undistorted/openmvs",
            entrypoint=densify_bin,
            extra_docker_args=linux_memory_hacks
        )
        # Record where OpenMVS saves its dense cloud
        source_ply = os.path.join(project_path, 'undistorted', 'openmvs', 'scene_dense.ply')

    # --- THE NEW UNIFICATION STEP ---
    print("\n--- Finalizing Project Files ---")
    final_ply_path = os.path.join(project_path, 'scene_dense.ply')
    
    if os.path.exists(source_ply):
        # We copy it up to the main folder so the downstream Open3D script is totally blind
        # to whether we used a Mac or a Supercomputer. It just looks for scene_dense.ply!
        shutil.copy2(source_ply, final_ply_path)
        print(f"Pipeline Complete! Point cloud successfully extracted to: \n -> {final_ply_path}")
    else:
        print(f"Error: Expected point cloud not found at {source_ply}. Densification may have failed.")

if __name__ == "__main__":
    if len(sys.argv) < 2: 
        print("Usage: python py/auto_reconstruct.py <path>")
    else: 
        main(sys.argv[1])