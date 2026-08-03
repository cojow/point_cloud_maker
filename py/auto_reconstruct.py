import os
import subprocess
import json
import glob
import re
import sys
import shutil
import platform
import time
import argparse
import shlex
import numpy as np
from resource_monitor import MemoryMonitor
from reconstruction_leveling import level_reconstruction, read_dji_gimbal_attitude

# Line-buffer stdout: when redirected to a file (e.g. a SLURM .out log),
# Python block-buffers by default and only flushes at exit. If the job gets
# killed (timeout, OOM) instead of exiting cleanly, everything this script
# printed - peak memory, step progress, resource sizing info - is lost. The
# subprocess calls to docker/apptainer are unaffected by this; they inherit
# the fd directly and were already showing up in real time.
sys.stdout.reconfigure(line_buffering=True)

'''
 Run using: python auto_reconstruct.py data/900EBlock

 Note for Linux/Supercomputer:
 Make sure APPTAINER_IMAGE below points at a working image.
 Run: apptainer pull odm.sif docker://opendronemap/odm:latest

 If apptainer fails with something like "squashfuse_ll ... fuse: device not
 found", the compute node doesn't have FUSE available to mount the .sif.
 Workaround: convert it to a sandbox (a plain directory, no FUSE mount
 needed either to build it or to run against it) and point APPTAINER_IMAGE
 there instead:
   apptainer build --sandbox /home/willicon/point_cloud/odm_sandbox /home/willicon/point_cloud/odm.sif
'''

# NOTE: This is an absolute path tied to a specific account/machine (e.g. the
# supercomputer). It cannot be made portable/relative because it points outside
# this repo. If you're running this on a different account or machine, update
# this path to wherever you pulled odm.sif (or built the sandbox - see above)
# on that system. Works identically whether this points at a .sif file or a
# sandbox directory - apptainer exec treats them the same way.
APPTAINER_IMAGE = "/home/willicon/point_cloud/odm_sandbox"
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

# Where the OpenSfM launcher actually lives in the opendronemap/odm image. The
# probe below still runs, but it checks this path first and only falls back to
# walking /code and /usr when it isn't there. That walk is not cheap: against a
# --sandbox (an uncompressed directory of hundreds of thousands of small files,
# often on a network filesystem) it stats the entire image to rediscover a path
# that hasn't moved.
KNOWN_OPENSFM_PATH = "/code/SuperBuild/install/bin/opensfm/bin/opensfm"

def get_odm_opensfm_path(engine):
    return get_binary_path(
        engine, "OpenSfM",
        f"if [ -f {KNOWN_OPENSFM_PATH} ]; then echo {KNOWN_OPENSFM_PATH}; "
        f"else find /code /usr -name opensfm -type f 2>/dev/null | grep bin | head -n 1; fi"
    )

def _resolve_undistorted_shot_image(images_dir, undistorted_filename):
    """undistorted/reconstruction.json's shot keys are the UNDISTORTED image's
    OWN filename, which OpenSfM writes by appending its own '.jpg' onto the
    ORIGINAL filename - confirmed against real project output:
    'DJI_..._D.JPG' -> 'DJI_..._D.JPG.jpg' (also visible in any log's
    "Pruning depthmap for image X.JPG.jpg" lines). The gimbal telemetry we
    need lives in the original file in images_dir, not that undistorted copy
    - recover it by trying the key as-is first (in case this convention ever
    changes), then falling back to stripping one trailing '.jpg'/'.JPG'.
    Returns a path that may not exist if neither form resolves; the caller's
    normal not-found handling (read_dji_gimbal_attitude returns None) covers
    that case the same way it always has."""
    direct = os.path.join(images_dir, undistorted_filename)
    if os.path.exists(direct):
        return direct
    if undistorted_filename.lower().endswith('.jpg'):
        stripped = os.path.join(images_dir, undistorted_filename[:-4])
        if os.path.exists(stripped):
            return stripped
    return direct

def restrict_depthmap_shots(project_path, pitch_threshold):
    """Temporarily strips non-nadir shots out of the UNDISTORTED dataset's own
    reconstruction.json (undistorted/reconstruction.json) - the copy OpenSfM's
    compute_depthmaps actually reads. compute_depthmaps iterates
    reconstruction.shots.values() with no built-in way to restrict to a
    subset, and its neighbor search whitelists candidates against that same
    shots dict (confirmed against OpenSfM's own source: shots absent from it
    are just skipped, not an error) - so removing shots here is exactly
    equivalent to telling it "don't densify these, and don't use them as
    neighbors for anyone else's depthmap either."

    This is for a two-flight setup where a nadir pass is the actual point
    cloud source and a separate oblique/frontal pass exists mainly to
    photograph elevations for something else (e.g. an object detection
    model) - those shots still need real poses (for cropping that photo
    later), so they go through extract_metadata/detect_features/
    match_features/reconstruct/undistort normally. They just never need to
    become point-cloud points, and compute_depthmaps is the single most
    expensive stage in the whole pipeline (resolution^2 per image) - so
    skipping it for the shots that don't need it is a real, direct saving,
    not a workaround.

    The PROJECT-ROOT reconstruction.json (what extract_buildings_floor.py
    reads for camera poses) is a completely separate file and is never
    touched here.

    Classifies by DJI gimbal pitch - reusing the same telemetry
    reconstruction_leveling.py already reads for gravity correction - rather
    than filename or folder, so it works regardless of how the two flights'
    images happen to be named. A shot with no readable gimbal telemetry
    (non-DJI drone, or tags missing) is kept rather than guessed away, since
    silently losing coverage we can't classify is worse than leaving it in.

    Returns (backup_bytes, n_kept, n_skipped). Pass backup_bytes to
    restore_depthmap_shots afterward - always, success or failure - so the
    project directory doesn't end up in a permanently-modified state."""
    recon_path = os.path.join(project_path, 'undistorted', 'reconstruction.json')
    images_dir = os.path.join(project_path, 'images')
    if not os.path.exists(recon_path):
        return None, 0, 0

    with open(recon_path, 'rb') as f:
        backup_bytes = f.read()
    reconstruction = json.loads(backup_bytes)
    recon = reconstruction[0]
    shots = recon.get('shots', {})

    kept = {}
    n_skipped = 0
    n_unclassifiable = 0
    for filename, shot in shots.items():
        attitude = read_dji_gimbal_attitude(_resolve_undistorted_shot_image(images_dir, filename))
        if attitude is not None and attitude[1] > pitch_threshold:
            n_skipped += 1              # pitch above threshold -> not near-nadir
        else:
            kept[filename] = shot       # near-nadir, or unclassifiable -> keep it
            if attitude is None:
                n_unclassifiable += 1
    recon['shots'] = kept

    # A shot only ends up here as "unclassifiable" if its source image
    # couldn't be found/read at all - normally rare. If it's happening for
    # most or all shots, something is silently defeating classification
    # entirely (e.g. a path/filename mismatch) rather than genuinely lacking
    # telemetry, and --depthmap-nadir-only will look like a no-op even though
    # it ran - worth surfacing loudly rather than only in a quiet count.
    if shots and n_unclassifiable / len(shots) > 0.5:
        print(f"      [!] WARNING: {n_unclassifiable}/{len(shots)} shots were unclassifiable "
              f"(no readable gimbal telemetry) and were kept as a safe default - if you expected "
              f"most of these to be skipped, this flag likely isn't finding your source images "
              f"correctly rather than genuinely lacking telemetry.")

    with open(recon_path, 'w') as f:
        json.dump(reconstruction, f, indent=4)
    return backup_bytes, len(kept), n_skipped

def restore_depthmap_shots(project_path, backup_bytes):
    """Undoes restrict_depthmap_shots - always call this after
    compute_depthmaps runs (success or failure) so undistorted/
    reconstruction.json ends up back in its normal, complete state."""
    if backup_bytes is None:
        return
    recon_path = os.path.join(project_path, 'undistorted', 'reconstruction.json')
    with open(recon_path, 'wb') as f:
        f.write(backup_bytes)

def run_opensfm_steps(engine, steps, project_path, opensfm_bin):
    """Runs several OpenSfM steps inside ONE container invocation.

    Each `apptainer exec` costs 12-23s of startup before any work happens
    (measured across the stage gaps in the fir job logs), and the original code
    paid that for all seven steps separately. The steps are chained with && so a
    failure still stops the batch, and each one is announced first so a failure
    is still attributable to a specific step in the log."""
    inner = " && ".join(
        f"echo '>>> OpenSfM step: {step}' && {shlex.quote(opensfm_bin)} {step} /project"
        for step in steps
    )
    print(f"--- Container batch ({len(steps)} step{'s' if len(steps) != 1 else ''}): {', '.join(steps)} ---")
    run_container_command(
        engine=engine,
        command_list=["-c", inner],
        host_project_path=project_path,
        entrypoint="sh"
    )

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

PLY_TYPE_MAP = {
    'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1',
    'short': 'i2', 'int16': 'i2', 'ushort': 'u2', 'uint16': 'u2',
    'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4',
    'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
}

# OpenMVS writes colours under non-standard property names, which is why
# Open3D silently loads the cloud with no colours and extract_buildings_floor.py
# carries a repair_openmvs_ply_colors() workaround. Renaming them here means
# that workaround finds nothing to fix and becomes a no-op, which also saves it
# from making its own full-size copy of the point cloud downstream.
PLY_COLOR_RENAMES = {'diffuse_red': 'red', 'diffuse_green': 'green', 'diffuse_blue': 'blue'}

def _write_ply_block(block, fout, dtype, names, n_props):
    """Parses one whitespace-separated ASCII block into the packed binary
    record layout and appends it. Returns the number of vertices written."""
    vals = np.fromstring(block.decode('ascii'), dtype=np.float64, sep=' ')
    if vals.size == 0:
        return 0
    if vals.size % n_props:
        raise ValueError(f"ASCII body block has {vals.size} values, not a multiple of {n_props} properties")
    rows = vals.reshape(-1, n_props)
    rec = np.empty(len(rows), dtype=dtype)
    for k, nm in enumerate(names):
        rec[nm] = rows[:, k]
    fout.write(rec.tobytes())
    return len(rows)

def convert_ply_ascii_to_binary(src_path, dst_path, chunk_bytes=1 << 24):
    """Rewrites an ASCII PLY as binary_little_endian, streaming in chunks so
    memory stays flat regardless of point count.

    OpenSfM emits merged.ply as ASCII: the 245-image fir run produced a 1.8 GB
    file for 33.1M points, where the same data packed binary is ~0.9 GB. The
    size is only half of it - the real cost is downstream, where Open3D has to
    text-parse 33 million lines every time extract_buildings_floor.py loads the
    cloud.

    Returns (n_vertices, src_bytes, dst_bytes) on success, or None if the file
    isn't a plain single-element ASCII point cloud - in which case the caller
    should just move it unchanged rather than risk mangling it."""
    with open(src_path, 'rb') as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                return None                      # ran out of file before end_header
            header.append(line)
            if line.strip() == b'end_header':
                break
            if len(header) > 200:
                return None                      # implausible header, don't guess
        body_offset = f.tell()

    fmt, elements, props = None, [], []
    for raw in header:
        parts = raw.decode('ascii', 'replace').strip().split()
        if not parts:
            continue
        if parts[0] == 'format' and len(parts) > 1:
            fmt = parts[1]
        elif parts[0] == 'element' and len(parts) > 2:
            elements.append((parts[1], int(parts[2])))
        elif parts[0] == 'property':
            if parts[1] == 'list':
                return None                      # faces/variable-length: not a point cloud
            props.append((parts[1], parts[2]))

    # Only handle the shape merged.ply actually has: one element, fixed-width
    # scalar properties, ASCII. Anything else falls back to a plain move.
    if fmt != 'ascii' or len(elements) != 1 or not props:
        return None
    if any(t not in PLY_TYPE_MAP for t, _ in props):
        return None

    n_vertices = elements[0][1]
    names = [PLY_COLOR_RENAMES.get(n, n) for _, n in props]
    if len(set(names)) != len(names):
        return None                              # rename would collide; leave it alone
    dtype = np.dtype([(n, PLY_TYPE_MAP[t]) for n, (t, _) in zip(names, props)])
    n_props = len(props)

    out_header = ["ply", "format binary_little_endian 1.0", f"element {elements[0][0]} {n_vertices}"]
    out_header += [f"property {t} {n}" for (t, _), n in zip(props, names)]
    out_header.append("end_header")

    written, buf = 0, b''
    with open(src_path, 'rb') as fin, open(dst_path, 'wb') as fout:
        fin.seek(body_offset)
        fout.write(("\n".join(out_header) + "\n").encode('ascii'))
        while True:
            data = fin.read(chunk_bytes)
            if not data:
                break
            buf += data
            cut = buf.rfind(b'\n')               # only parse whole lines
            if cut == -1:
                continue
            block, buf = buf[:cut], buf[cut + 1:]
            written += _write_ply_block(block, fout, dtype, names, n_props)
        if buf.strip():
            written += _write_ply_block(buf, fout, dtype, names, n_props)

    if written != n_vertices:
        raise ValueError(f"converted {written} vertices but header declared {n_vertices}")
    return n_vertices, os.path.getsize(src_path), os.path.getsize(dst_path)

def cleanup_intermediate_files(project_path):
    """Deletes everything in project_path except what extract_buildings_floor.py
    actually reads: images/, reconstruction.json, and scene_dense.ply. Everything
    else here (features/, matches/, undistorted/ depthmaps and undistorted images,
    tracks.csv, camera_models.json, config.yaml, *.MRK, etc.) is OpenSfM/ODM working
    state needed only to produce those three things, not to extract buildings
    from them - and is often much larger than the outputs actually kept."""
    KEEP = {"images", "reconstruction.json", "scene_dense.ply"}

    print("\n--- Cleaning up intermediate reconstruction files ---")
    freed_bytes = 0
    for entry in sorted(os.listdir(project_path)):
        if entry in KEEP:
            continue
        entry_path = os.path.join(project_path, entry)
        try:
            if os.path.isdir(entry_path) and not os.path.islink(entry_path):
                size = sum(os.path.getsize(os.path.join(dp, f))
                           for dp, _, fnames in os.walk(entry_path) for f in fnames)
                shutil.rmtree(entry_path)
            else:
                size = os.path.getsize(entry_path)
                os.remove(entry_path)
            freed_bytes += size
            print(f"      removed {entry} ({size / (1024**2):.1f} MB)")
        except OSError as e:
            print(f"      [!] Could not remove {entry}: {e}")

    print(f"      -> Freed {freed_bytes / (1024**3):.2f} GB. Kept: {', '.join(sorted(KEEP))}")

def set_config_value(config_lines, key, value):
    """Replaces (or adds) a single top-level 'key: value' line in an OpenSfM
    config.yaml's lines, leaving every other line untouched."""
    config_lines = [line for line in config_lines if not line.strip().startswith(f"{key}:")]
    # Guarantee the line we're about to append starts on its own line. If the
    # file's last line had no trailing newline (a hand-edited config.yaml - the
    # README tells you to edit these with nano - or one written by any tool
    # other than this function), a bare append silently welds the new setting
    # onto the end of the previous one: "feature_type: SIFTprocesses: 16",
    # destroying BOTH settings with no error. Every config.yaml this script has
    # written so far ends in a newline, which is the only reason this hasn't
    # bitten yet.
    if config_lines and not config_lines[-1].endswith("\n"):
        config_lines[-1] += "\n"
    config_lines.append(f"{key}: {value}\n")
    return config_lines

def main(args):
    global_start_time = time.time()
    project_path = args.project_path

    mem_monitor = MemoryMonitor()
    if mem_monitor.start():
        print("--- Memory monitor: sampling via psutil (main + containerized reconstruction process) ---")
    else:
        print("--- Memory monitor: psutil not installed, falling back to a cruder end-of-run estimate "
              "(pip install psutil for an accurate number) ---")

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
            
    config_lines = set_config_value(config_lines, "processes", max_cores)

    # Densification + matching tuning - only touches config.yaml for flags the
    # user actually passed; anything left as None keeps whatever OpenSfM's own
    # default (or whatever's already in this project's config.yaml) is.
    depthmap_overrides = {
        "depthmap_method": args.depthmap_method,
        "depthmap_resolution": args.depthmap_resolution,
        "depthmap_min_consistent_views": args.depthmap_min_consistent_views,
        "depthmap_min_patch_sd": args.depthmap_min_patch_sd,
        "depthmap_num_neighbors": args.depthmap_num_neighbors,
        "depthmap_num_matching_views": args.depthmap_num_matching_views,
        "matching_gps_neighbors": args.matching_gps_neighbors,
        "matching_gps_distance": args.matching_gps_distance,
    }
    for key, value in depthmap_overrides.items():
        if value is not None:
            config_lines = set_config_value(config_lines, key, value)
            print(f"--- Overriding {key}: {value} ---")

    with open(config_path, 'w') as f:
        f.writelines(config_lines)
    
    # Phase 1: OpenSfM Pipeline.
    #
    # Batched into as few container invocations as the pipeline's two Python-side
    # hooks allow, rather than one per step:
    #   - inject_mrk_data has to run after extract_metadata (it edits the exif/
    #     JSONs that step writes) and before detect_features. Only needed when
    #     there are .MRK files, so with no .MRK files this split disappears.
    #   - level_reconstruction has to run after reconstruct and before undistort,
    #     so the gravity correction is baked into the densified cloud.
    # Result: 2 container launches (or 3 with .MRK files) instead of 7.
    if mrk_files:
        run_opensfm_steps(engine, ["extract_metadata"], project_path, opensfm_bin)
        inject_mrk_data(project_path, mrk_files)
        run_opensfm_steps(engine, ["detect_features", "match_features", "create_tracks", "reconstruct"],
                          project_path, opensfm_bin)
    else:
        run_opensfm_steps(engine, ["extract_metadata", "detect_features", "match_features",
                                   "create_tracks", "reconstruct"], project_path, opensfm_bin)

    # Correct the reconstruction's orientation to true gravity before
    # undistort/densification bake the (possibly tilted) frame into
    # scene_dense.ply. See reconstruction_leveling.py for the method.
    print("\n--- Leveling reconstruction to true gravity ---")
    level_reconstruction(project_path)

    # Phase 2: undistort + Lightweight OpenSfM Densification (Universal)
    print("\n--- Undistorting and running Lightweight OpenSfM Densification ---")
    if args.depthmap_nadir_only:
        # Needs a Python-side hook between undistort and compute_depthmaps, so
        # (unlike the default path) these can't be one batched container call.
        run_opensfm_steps(engine, ["undistort"], project_path, opensfm_bin)
        backup, n_kept, n_skipped = restrict_depthmap_shots(project_path, args.nadir_pitch_threshold)
        print(f"      -> --depthmap-nadir-only: densifying {n_kept} near-nadir shot(s) "
              f"(gimbal pitch <= {args.nadir_pitch_threshold}° or unclassifiable), "
              f"skipping {n_skipped} oblique/frontal shot(s)")
        try:
            run_opensfm_steps(engine, ["compute_depthmaps"], project_path, opensfm_bin)
        finally:
            restore_depthmap_shots(project_path, backup)
    else:
        run_opensfm_steps(engine, ["undistort", "compute_depthmaps"], project_path, opensfm_bin)
    source_ply = os.path.join(project_path, 'undistorted', 'depthmaps', 'merged.ply')

    # Finalizing
    print("\n--- Finalizing Project Files ---")
    final_ply_path = os.path.join(project_path, 'scene_dense.ply')
    
    if os.path.exists(source_ply):
        converted = None
        if not args.ascii_ply:
            try:
                t0 = time.time()
                converted = convert_ply_ascii_to_binary(source_ply, final_ply_path)
                if converted:
                    n_pts, src_b, dst_b = converted
                    os.remove(source_ply)
                    print(f"Converted point cloud to binary PLY in {time.time()-t0:.1f}s: "
                          f"{n_pts:,} points, {src_b/(1024**3):.2f} GB ASCII -> "
                          f"{dst_b/(1024**3):.2f} GB binary ({100*(1-dst_b/src_b):.0f}% smaller)")
                else:
                    print("Point cloud is not a plain ASCII point cloud; moving it unchanged.")
            except Exception as e:
                # Never let a conversion problem cost the run its output.
                print(f"[!] Binary PLY conversion failed ({e}); falling back to moving the ASCII file.")
                if os.path.exists(final_ply_path):
                    os.remove(final_ply_path)
                converted = None

        if not converted:
            # Move, don't copy. merged.ply is a consumed intermediate and
            # scene_dense.ply is the deliverable, so copying 1.8 GB only to
            # delete the original (which --cleanup does anyway) is pure waste.
            # os.replace is an instant same-filesystem rename; the copy is kept
            # only as a cross-device fallback.
            try:
                os.replace(source_ply, final_ply_path)
            except OSError:
                shutil.copy2(source_ply, final_ply_path)

        print(f"Pipeline Complete! Point cloud successfully extracted to: \n -> {final_ply_path}")
    else:
        print(f"Error: Expected point cloud not found at {source_ply}. Densification may have failed.")

    peak_gb, peak_method = mem_monitor.stop_and_report()

    if args.cleanup:
        if os.path.exists(final_ply_path):
            cleanup_intermediate_files(project_path)
        else:
            print("--- Skipping cleanup: scene_dense.ply was not produced, keeping intermediate "
                  "files around so the failed run can still be debugged ---")

    image_dir = os.path.join(project_path, 'images')
    n_images = len([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.tif', '.tiff'))]) \
        if os.path.exists(image_dir) else 0

    total_elapsed = time.time() - global_start_time
    hours, remainder = divmod(total_elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("-" * 40)
    print(f"Total Processing Time: {int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s")
    print(f"Peak memory used: {peak_gb:.1f} GB  ({peak_method})")
    print(f"Image count this run: {n_images}")
    if engine == "docker":
        print("NOTE: under Docker, the memory monitor cannot see inside the container process tree")
        print("(Docker's client/daemon architecture means it isn't a real child process of this script),")
        print("so this number is likely a severe undercount. This reporting is only reliable under")
        print("Apptainer (i.e. on the actual supercomputer run), where the contained process IS a real child.")
    print("Keep a note of the peak memory line against this image count for calibrating --mem next time.")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct a point cloud from drone photos via OpenSfM/ODM.")
    parser.add_argument("project_path", type=str, help="Path to the project directory (e.g. data/900EBlock).")

    parser.add_argument("--depthmap-method", type=str, default=None,
                        choices=["PATCH_MATCH", "PATCH_MATCH_SAMPLE", "BRUTE_FORCE"],
                        help="Depth estimation algorithm (OpenSfM stock default: PATCH_MATCH_SAMPLE, confirmed "
                             "via job logs). PATCH_MATCH_SAMPLE is a faster, sparser approximation - it attempts "
                             "fewer per-pixel estimates and refines them less. PATCH_MATCH is the full method: "
                             "more coverage (fewer un-attempted pixels) and more refined estimates, at real extra "
                             "cost - likely one of the more expensive levers here, not a cheap one. BRUTE_FORCE is "
                             "an older exhaustive-search fallback, generally not preferred. Leave unset to keep "
                             "the existing config.yaml/OpenSfM default.")
    parser.add_argument("--depthmap-resolution", type=int, default=None,
                        help="Depth estimation resolution in pixels, per image (OpenSfM stock default: 640). "
                             "The single biggest lever on point cloud detail - but cost scales roughly with "
                             "the square of this value, per image. Leave unset to keep the existing config.yaml/"
                             "OpenSfM default.")
    parser.add_argument("--depthmap-min-consistent-views", type=int, default=None,
                        help="Minimum number of neighboring images that must agree before a depth estimate is "
                             "kept (OpenSfM stock default: 3). Lower = more points survive, more noise. Cheap "
                             "to tune - just a filter on estimates already computed. Leave unset to keep the "
                             "existing config.yaml/OpenSfM default.")
    parser.add_argument("--depthmap-min-patch-sd", type=float, default=None,
                        help="Minimum patch texture (standard deviation) required to attempt a depth estimate "
                             "(OpenSfM stock default: ~1.0). Lowering it helps fill in low-texture surfaces like "
                             "roofs and painted walls. Leave unset to keep the existing config.yaml/OpenSfM default.")
    parser.add_argument("--depthmap-num-neighbors", type=int, default=None,
                        help="Number of neighboring images considered per shot during depth estimation (OpenSfM "
                             "stock default: 10). More = better accuracy/coverage, more compute. Leave unset to "
                             "keep the existing config.yaml/OpenSfM default.")
    parser.add_argument("--depthmap-num-matching-views", type=int, default=None,
                        help="Number of matching views used per depth estimate (OpenSfM stock default: 5). "
                             "Leave unset to keep the existing config.yaml/OpenSfM default.")
    parser.add_argument("--matching-gps-neighbors", type=int, default=None,
                        help="Match each image only against its N nearest neighbors by GPS position "
                             "(OpenSfM stock default: 0, meaning 'no neighbor cap' - the distance bound "
                             "alone decides). THIS IS THE MAIN SCALING LEVER. match_features is the only "
                             "O(n^2) stage in the pipeline: on the 245-image fir run it attempted 17,858 "
                             "pairs and only 1,839 (10.3%%) produced a usable match, so ~90%% of that "
                             "5.98-minute stage was spent proving that non-overlapping images don't "
                             "overlap. Capping neighbors makes the pair count grow linearly with image "
                             "count instead of quadratically. Try 20-30 for a mapping flight with normal "
                             "overlap. Verify the effect on the next run via the 'Matching N image pairs' "
                             "line in the log, and confirm the reconstruction still registers about as "
                             "many images ('Reconstruction 0: N images'). Leave unset to keep the "
                             "existing config.yaml/OpenSfM default.")
    parser.add_argument("--matching-gps-distance", type=float, default=None,
                        help="Maximum GPS distance in meters between two images for them to be considered "
                             "a candidate pair (OpenSfM stock default: 150). This bound and "
                             "--matching-gps-neighbors are applied together (nearest N, subject to being "
                             "within this distance), so tightening either one shrinks the pair list. At a "
                             "typical ~100m survey altitude over a neighborhood, 150m is wide enough that "
                             "nearly every image is a candidate for nearly every other, which is why the "
                             "neighbor cap above is usually the more effective knob. Leave unset to keep "
                             "the existing config.yaml/OpenSfM default.")
    parser.add_argument("--depthmap-nadir-only", action="store_true",
                        help="For a two-flight setup (a near-nadir pass for the point cloud, plus a "
                             "separate oblique/frontal pass shot mainly for something else, e.g. object "
                             "detection training images) - skip compute_depthmaps for the frontal shots "
                             "entirely, classified by DJI gimbal pitch. The frontal shots still get real "
                             "camera poses (registered normally via reconstruct/undistort) for downstream "
                             "photo cropping; they just never pay for densification, which is the single "
                             "most expensive stage in the pipeline (cost scales with depthmap_resolution "
                             "squared, PER IMAGE). Lets you raise --depthmap-resolution for the nadir set "
                             "without that cost also landing on every frontal-flight image, whose pixels "
                             "were never meant to become point-cloud points anyway. Requires DJI XMP "
                             "gimbal telemetry in the images (same requirement as the leveling step) - "
                             "shots without it are kept/densified rather than silently dropped.")
    parser.add_argument("--nadir-pitch-threshold", type=float, default=-60.0,
                        help="Gimbal pitch cutoff for --depthmap-nadir-only, in degrees (DJI convention: "
                             "-90 = straight down, 0 = level). Shots with pitch AT OR BELOW this (i.e. "
                             "steeper / more downward-looking than the cutoff) are treated as nadir and "
                             "densified; shots above it (more level/oblique) are skipped. Default -60 "
                             "assumes a clear separation between a near-nadir pass and a much more level "
                             "frontal pass - check a few of your frontal images' actual GimbalPitchDegree "
                             "if the two flights are closer together in pitch than that.")
    parser.add_argument("--ascii-ply", action="store_true",
                        help="Keep scene_dense.ply in OpenSfM's native ASCII format instead of converting "
                             "it to binary_little_endian. Conversion is on by default because it roughly "
                             "halves the file (1.8 GB -> ~0.9 GB for the 245-image fir run's 33.1M points) "
                             "and, more importantly, spares extract_buildings_floor.py from text-parsing 33 "
                             "million lines on every load. Conversion also renames OpenMVS's non-standard "
                             "diffuse_red/green/blue colour properties to the standard red/green/blue, so "
                             "Open3D loads colours directly. Pass this flag if you need the ASCII file for "
                             "an external tool that can't read binary PLY.")
    parser.add_argument("--cleanup", action="store_true",
                        help="After the pipeline finishes, delete everything in the project directory "
                             "except images/, reconstruction.json, and scene_dense.ply - the only three "
                             "things extract_buildings_floor.py reads. This removes OpenSfM's working "
                             "state (features/, matches/, undistorted/ depthmaps and undistorted images, "
                             "tracks.csv, camera_models.json, config.yaml, etc.), which is usually much "
                             "larger than what's kept. Skipped automatically if scene_dense.ply wasn't "
                             "produced, so a failed run's intermediate files aren't lost.")

    args = parser.parse_args()
    main(args)