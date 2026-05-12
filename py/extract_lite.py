import os
import json
import numpy as np
import open3d as o3d
import csv
import sys
import time
import shutil
import glob
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay, KDTree
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS
from pyproj import Transformer
import concurrent.futures

# --- SYSTEM CONFIGURATION ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

EPSG_CODE = "EPSG:32612"  # Utah UTM Zone 12N
NUM_CORES = os.cpu_count() or 1

# --- 1. AUTOMATION & REPAIR HELPERS ---

def auto_detect_offsets(project_dir):
    """Automatically pulls the UTM anchor from reconstruction.json."""
    json_path = os.path.join(project_dir, 'reconstruction.json')
    if not os.path.exists(json_path):
        print("      [!] Warning: reconstruction.json not found. Using (0,0) offset.")
        return 0.0, 0.0
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        ref = data[0].get('reference_lla')
        trans = Transformer.from_crs("EPSG:4326", EPSG_CODE, always_xy=True)
        utm_x, utm_y = trans.transform(ref['longitude'], ref['latitude'])
        print(f"      [AUTO] Coordinate Anchor: X={utm_x:.3f}, Y={utm_y:.3f}")
        return utm_x, utm_y
    except Exception as e:
        print(f"      [!] Error parsing offsets: {e}")
        return 0.0, 0.0

def repair_openmvs_ply_colors(ply_path):
    """Patches binary header for OpenMVS colors to ensure Open3D compatibility."""
    with open(ply_path, 'rb') as f: header_chunk = f.read(2000)
    if b'diffuse_red' in header_chunk:
        print("      -> [FIX] Patching OpenMVS header color tags...")
        fixed_path = ply_path.replace('.ply', '_color_fixed.ply')
        with open(ply_path, 'rb') as f: content = f.read()
        content = content.replace(b'property uchar diffuse_red', b'property uchar red        ')
        content = content.replace(b'property uchar diffuse_green', b'property uchar green      ')
        content = content.replace(b'property uchar diffuse_blue', b'property uchar blue       ')
        with open(fixed_path, 'wb') as f: f.write(content)
        return fixed_path
    return ply_path

# --- 2. MAIN PIPELINE ---

def process_reconstruction_v4_1(project_path):
    global_start_time = time.time()
    
    # Path Configuration
    ply_path = os.path.join(project_path, 'scene_dense.ply')
    output_dir = os.path.join(project_path, "lite_analysis_v4_1")
    diag_dir = os.path.join(output_dir, "diagnostics")
    for d in [output_dir, diag_dir]: os.makedirs(d, exist_ok=True)

    # STEP 1: COORDINATE SYNC
    print(f"[1/6] Detecting Global Offsets...")
    GLOBAL_OFFSET_X, GLOBAL_OFFSET_Y = auto_detect_offsets(project_path)

    # STEP 2: LOAD & REPAIR
    print(f"[2/6] Loading point cloud from {os.path.basename(ply_path)}...")
    t0 = time.time()
    safe_ply_path = repair_openmvs_ply_colors(ply_path)
    pcd = o3d.io.read_point_cloud(safe_ply_path)
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    print(f"      -> Points after downsample: {len(pcd.points)} (Done in {time.time()-t0:.2f}s)")

    # STEP 3: LEVELING (DIAGNOSTIC 3)
    print("[3/6] Leveling scene geometry...")
    t0 = time.time()
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.2)
    pts_clean = np.asarray(pcd.points)
    ground_sample = pts_clean[pts_clean[:, 2] < np.percentile(pts_clean[:, 2], 30)]
    centroid = np.mean(ground_sample, axis=0)
    _, evecs = np.linalg.eigh(np.cov(ground_sample.T))
    norm = evecs[:, 0] if evecs[2, 0] > 0 else -evecs[:, 0] 

    v = np.cross(norm, [0, 0, 1])
    s, c = np.linalg.norm(v), np.dot(norm, [0, 0, 1])
    if s > 1e-6:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rot_mat = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s**2))
        pcd.rotate(rot_mat, center=(0,0,0))
    
    pcd.translate((-centroid[0], -centroid[1], 0))
    o3d.io.write_point_cloud(os.path.join(diag_dir, "step3_leveled.ply"), pcd)
    print(f"      Saved: step3_leveled.ply (Done in {time.time()-t0:.2f}s)")

    # STEP 4: GROUND REMOVAL (DIAGNOSTIC 4)
    print("[4/6] Applying RANSAC Ground Removal...")
    t0 = time.time()
    _, inliers = pcd.segment_plane(distance_threshold=0.5, ransac_n=3, num_iterations=1000)
    gz = np.median(np.asarray(pcd.select_by_index(inliers).points)[:, 2])
    ag_pcd = pcd.select_by_index(inliers, invert=True)
    ag_pcd.translate((0, 0, -gz))
    
    ag_pts = np.asarray(ag_pcd.points)
    ag_colors = np.asarray(ag_pcd.colors)
    o3d.io.write_point_cloud(os.path.join(diag_dir, "step4_above_ground.ply"), ag_pcd)
    print(f"      Saved: step4_above_ground.ply (Done in {time.time()-t0:.2f}s)")

    # STEP 5: ROOF ISOLATION (DIAGNOSTIC 5)
    print(f"[5/6] Isolating roof facets (Parallel Search - {NUM_CORES} cores)...")
    t0 = time.time()
    high_idx = np.where(ag_pts[:, 2] > 2.2)[0]
    h_pts = ag_pts[high_idx]
    
    if len(h_pts) > 0:
        kdtree_high = KDTree(h_pts)
        _, n_idx = kdtree_high.query(h_pts, k=12, workers=-1) 
        planar_mask = np.std(h_pts[n_idx][:, :, 2], axis=1) < 0.20
        roof_pcd = ag_pcd.select_by_index(high_idx[planar_mask])
        o3d.io.write_point_cloud(os.path.join(diag_dir, "step5_potential_roofs.ply"), roof_pcd)
        print(f"      Saved: step5_potential_roofs.ply (Done in {time.time()-t0:.2f}s)")
    else:
        print("      [!] Error: No points above 2.2m.")
        return

    # STEP 5b: SPECTRAL-GEOMETRIC HYBRID PRUNING (DIAGNOSTIC 5b)
    print(f"[5b/6] Pruning vegetation (Geometric Consistency Filter - {NUM_CORES} cores)...")
    t0 = time.time()
    
    # A. Z-SCORE SUSPICION TRIGGER
    mean_h, std_h = np.mean(ag_pts[:, 2]), np.std(ag_pts[:, 2])
    is_tall = (ag_pts[:, 2] - mean_h) / std_h > 2.5 

    # B. SPECTRAL CHECK (Excess Green)
    r, g, b = ag_colors[:, 0], ag_colors[:, 1], ag_colors[:, 2]
    exg = (2 * g) - r - b
    is_green = (exg > 0.07) & (ag_pts[:, 2] > 0.8)

    # Initial vegetation candidates
    candidate_mask = is_green | is_tall
    candidate_indices = np.where(candidate_mask)[0]
    master_tree_mask = is_green.copy()

    if len(candidate_indices) > 0:
        ag_tree = KDTree(ag_pts)
        # Search neighborhood for geometric consistency (Radius 0.8m)
        neighbors = ag_tree.query_ball_point(ag_pts[candidate_indices], r=0.8, workers=-1)
        unique_neighbors = np.unique(np.fromiter((item for sublist in neighbors for item in sublist), dtype=int))
        
        # Analyze Local Geometry (k=32)
        _, n_idx_local = ag_tree.query(ag_pts[unique_neighbors], k=32, workers=-1)
        z_vals = ag_pts[n_idx_local][:, :, 2]
        
        # C. PLANAR CONSENSUS METRICS
        roughness = np.std(z_vals, axis=1)
        v_range = np.ptp(z_vals, axis=1) # Vertical range (thickness)
        
        # PRUNING LOGIC: Flag as tree if it is "thick" or "jagged"
        is_chaos = (v_range > 0.30) | (roughness > 0.07)
        master_tree_mask[unique_neighbors[is_chaos]] = True
        print(f"      -> {np.sum(master_tree_mask)} points flagged as organic/noise.")

    # Save Diagnostic RGB file (The "Cleaned" Cloud)
    pruned_ag_pcd = ag_pcd.select_by_index(np.where(~master_tree_mask)[0])
    o3d.io.write_point_cloud(os.path.join(diag_dir, "step5b_pruned_rgb.ply"), pruned_ag_pcd)
    print(f"      Saved: step5b_pruned_rgb.ply (Done in {time.time()-t0:.2f}s)")

    # STEP 6: BROAD CLUSTERING & MORPHOLOGY (DIAGNOSTIC 6)
    print(f"[6/6] Clustering with Slenderness Firewall...")
    t0 = time.time()
    final_pts = np.asarray(pruned_ag_pcd.points)
    # Target only high points for building seeds
    final_h_idx = np.where(final_pts[:, 2] > 2.2)[0]
    final_seeds = final_pts[final_h_idx]
    
    if len(final_seeds) < 100:
        return print("      [!] Error: No structures survived the pruning process.")

    seeds_sq = np.copy(final_seeds)
    seeds_sq[:, 2] *= 0.1 
    db = DBSCAN(eps=2.5, min_samples=15, n_jobs=-1).fit(seeds_sq)
    labels = db.labels_
    
    # MORPHOLOGY FILTER: Remove poles and narrow remains
    valid_mask = np.zeros(len(final_seeds), dtype=bool)
    num_buildings = 0
    for i in range(labels.max() + 1):
        idx_set = np.where(labels == i)[0]
        c_pts = final_seeds[idx_set]
        
        # Calculate aspect ratio
        m, M = c_pts.min(axis=0), c_pts.max(axis=0)
        w, d, h = max(M[0]-m[0], 0.1), max(M[1]-m[1], 0.1), M[2]-m[2]
        
        slenderness = h / np.sqrt(w * d)
        
        # If it is not a skinny pole and has enough density
        if slenderness < 2.5 and len(idx_set) > 150:
            valid_mask[idx_set] = True
            num_buildings += 1

    # Visualization Export
    clustered_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(final_seeds[valid_mask]))
    v_labels = labels[valid_mask]
    colors = np.random.rand(v_labels.max() + 1, 3)
    color_array = np.zeros((len(v_labels), 3))
    for i in range(len(v_labels)):
        if v_labels[i] >= 0: color_array[i] = colors[v_labels[i]]
        else: color_array[i] = [0.1, 0.1, 0.1]
    clustered_pcd.colors = o3d.utility.Vector3dVector(color_array)
    
    o3d.io.write_point_cloud(os.path.join(diag_dir, "step6_broad_clusters.ply"), clustered_pcd)
    print(f"      Saved: step6_broad_clusters.ply (Identified {num_buildings} buildings).")
    print(f"      Total Pipeline Time: {time.time() - global_start_time:.2f}s")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_v4_1.py /path/to/project_dir")
        sys.exit(1)
    process_reconstruction_v4_1(os.path.abspath(sys.argv[1]))