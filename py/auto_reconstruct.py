import os
import subprocess
import json
import glob
import re
import sys
import shutil
import platform
import time

'''
 Run using: python auto_reconstruct.py data/900EBlock
 
 Note for Linux/Supercomputer: 
 Make sure 'odm.sif' is at the absolute path specified below.
 Run: apptainer pull odm.sif docker://opendronemap/odm:latest
'''

APPTAINER_IMAGE = "/home/willicon/point_cloud/odm.sif"
DOCKER_IMAGE = "opendronemap/odm:latest"

def is_linux():
    """Detects if the script is running on a Linux machine (like the supercomputer)."""
    return platform.system().lower() == "linux"

def get_engine():
    """Determines which container engine to use based on the OS."""
    return "apptainer" if is_linux() else "docker"

def run_container_command(engine, command_list, host_project_path, work_dir_suffix="", entrypoint=None, env_vars=None):
    """Spins up the appropriate container engine mapping the host path to a static internal Linux path."""
    container_project_path = "/project"
    
    # Ensure internal paths use forward slashes for Linux, regardless of host OS
    if work_dir_suffix:
        safe_suffix = work_dir_suffix.replace("\\", "/")
        internal_work_dir = f"{container_project_path}/{safe_suffix}"
    else:
        internal_work_dir = container_project_path

    if engine == "docker":
        cmd = [
            "docker", "run", "-i", "--rm",
            "-v", f"{os.path.abspath(host_project_path)}:{container_project_path}",
            "-w", internal_work_dir
        ]
        if env_vars:
            for k, v in env_vars.items():
                cmd.extend(["-e", f"{k}={v}"])
        if entrypoint:
            cmd.extend(["--entrypoint", entrypoint])
            
        cmd.append(DOCKER_IMAGE)
        cmd.extend(command_list)
        
    elif engine == "apptainer":
        cmd = [
            "apptainer", "exec", 
            "--cleanenv",
            "--bind", f"{os.path.abspath(host_project_path)}:{container_project_path}", 
            "--pwd", internal_work_dir
        ]
        if env_vars:
            for k, v in env_vars.items():
                cmd.extend(["--env", f"{k}={v}"])
                
        cmd.append(APPTAINER_IMAGE)
        
        if entrypoint:
            cmd.append(entrypoint)
            
        cmd.extend(command_list)
        
    else:
        print(f"Error: Unknown engine '{engine}'")
        sys.exit(1)
        
    print(f"Executing {engine.capitalize()} Engine: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error executing {engine} engine.")
        sys.exit(1)

def get_binary_path(engine, binary_name, search_cmd):
    """Probes the selected container engine for an executable."""
    print(f"--- Probing {engine} container for {binary_name} executable ---")
    
    if engine == "docker":
        cmd = ["docker", "run", "--rm", "--entrypoint", "sh", DOCKER_IMAGE, "-c", search_cmd]
    else:
        cmd = ["apptainer", "exec", APPTAINER_IMAGE, "sh", "-c", search_cmd]
        
    try:
        output = subprocess.check_output(cmd, text=True).strip()
        if not output:
            print(f"Error: Could not locate {binary_name} inside container.")
            sys.exit(1)
        return output
    except Exception as e:
        print(f"Error executing container probe: {e}")
        sys.exit(1)

def get_odm_opensfm_path(engine):
    return get_binary_path(engine, "OpenSfM", "find /code /usr -name opensfm -type f 2>/dev/null | grep bin | head -n 1")

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
    global_start_time = time.time()
    
    engine = get_engine()
    print(f"--- Detected OS: {platform.system()} | Selected Engine: {engine.upper()} ---")

    if engine == "apptainer" and not os.path.exists(APPTAINER_IMAGE):
        print(f"Error: Could not find '{APPTAINER_IMAGE}'.")
        print("Run 'apptainer pull odm.sif docker://opendronemap/odm:latest' first.")
        sys.exit(1)

    project_path = os.path.abspath(project_path)
    organize_folders(project_path)
    
    search_pattern = os.path.join(project_path, "*.MRK")
    mrk_files = [f for f in glob.glob(search_pattern) if os.path.isfile(f)]
    
    opensfm_bin = get_odm_opensfm_path(engine)
    print(f"--- Located OpenSfM Engine at: {opensfm_bin} ---")
    
    # --- DYNAMIC CORE DETECTION ---
    slurm_cores = os.environ.get('SLURM_CPUS_PER_TASK')
    
    if slurm_cores and slurm_cores.isdigit():
        max_cores = slurm_cores
        print(f"--- SLURM Allocation Detected: Utilizing {max_cores} CPU cores ---")
    else:
        cpu_count = os.cpu_count()
        max_cores = str(max(1, cpu_count - 1)) if cpu_count else "1"
        print(f"--- Local Environment Detected: Utilizing {max_cores} CPU cores ---")
    
    config_path = os.path.join(project_path, "config.yaml")
    config_lines = []
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_lines = f.readlines()
            
    config_lines = [line for line in config_lines if not line.strip().startswith("processes:")]
    config_lines.append(f"processes: {max_cores}\n")
    
    with open(config_path, 'w') as f:
        f.writelines(config_lines)
    
    # Phase 1: OpenSfM Pipeline
    steps = ["extract_metadata", "detect_features", "match_features", "create_tracks", "reconstruct", "undistort"]
    for step in steps:
        if step == "detect_features" and mrk_files: 
            inject_mrk_data(project_path, mrk_files)
        
        run_container_command(
            engine=engine,
            command_list=[step, "/project"],
            host_project_path=project_path,
            entrypoint=opensfm_bin
        )

    # Phase 2: Lightweight OpenSfM Densification (Universal)
    print("\n--- Running Lightweight OpenSfM Densification ---")
    run_container_command(
        engine=engine,
        command_list=["compute_depthmaps", "/project"],
        host_project_path=project_path,
        entrypoint=opensfm_bin
    )
    source_ply = os.path.join(project_path, 'undistorted', 'depthmaps', 'merged.ply')

    # Finalizing
    print("\n--- Finalizing Project Files ---")
    final_ply_path = os.path.join(project_path, 'scene_dense.ply')
    
    if os.path.exists(source_ply):
        shutil.copy2(source_ply, final_ply_path)
        print(f"Pipeline Complete! Point cloud successfully extracted to: \n -> {final_ply_path}")
    else:
        print(f"Error: Expected point cloud not found at {source_ply}. Densification may have failed.")

    total_elapsed = time.time() - global_start_time
    hours, remainder = divmod(total_elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("-" * 40)
    print(f"Total Processing Time: {int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s")
    print("-" * 40)

if __name__ == "__main__":
    if len(sys.argv) < 2: 
        print("Usage: python auto_reconstruct.py <path>")
    else: 
        main(sys.argv[1])