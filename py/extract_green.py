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
    json_path = os.path.join(project_dir, 'reconstruction.json')
    if not os.path.exists(json_path):
        return 0.0, 0.0
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        ref = data[0].get('reference_lla')
        trans = Transformer.from_crs("EPSG:4326", EPSG_CODE, always_xy=True)
        return trans.transform(ref['longitude'], ref['latitude'])
    except:
        return 0.0, 0.0

def build_spatial_image_index(image_dir, transformer):
    paths, coords = [], []
    if not image_dir or not os.path.exists(image_dir): return None, None
    files = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.JPG"))
    for p in files:
        try:
            image = Image.open(p)
            exif = image._getexif()
            if not exif: continue
            gps = {}
            for k, v in exif.items():
                if TAGS.get(k) == 'GPSInfo':
                    for t, val in v.items(): gps[GPSTAGS.get(t, t)] = val
            def to_deg(v): return float(v[0]) + (float(v[1])/60.0) + (float(v[2])/3600.0)
            lat, lon = to_deg(gps['GPSLatitude']), to_deg(gps['GPSLongitude'])
            if gps['GPSLatitudeRef'] != 'N': lat = -lat
            if gps['GPSLongitudeRef'] != 'E': lon = -lon
            x, y = transformer.transform(lon, lat)
            paths.append(p); coords.append([x, y])
        except: continue
    return (paths, KDTree(np.array(coords))) if coords else (None, None)

def repair_openmvs_ply_colors(ply_path):
    with open(ply_path, 'rb') as f: header_chunk = f.read(2000)
    if b'diffuse_red' in header_chunk:
        fixed_path = ply_path.replace('.ply', '_color_fixed.ply')
        with open(ply_path, 'rb') as f: content = f.read()
        content = content.replace(b'property uchar diffuse_red', b'property uchar red        ')
        content = content.replace(b'property uchar diffuse_green', b'property uchar green      ')
        content = content.replace(b'property uchar diffuse_blue', b'property uchar blue       ')
        with open(fixed_path, 'wb') as f: f.write(content)
        return fixed_path
    return ply_path

# --- 2. GEOMETRY & EXTRACTION HELPERS ---

def calculate_alpha_shape(points_2d, alpha=1.2):
    if len(points_2d) < 4: return None
    try:
        tri = Delaunay(points_2d)
        edges = np.array([points_2d[tri.simplices[:, [0, 1]]], points_2d[tri.simplices[:, [1, 2]]], points_2d[tri.simplices[:, [2, 0]]]])
        lengths = np.sqrt(np.sum((edges[:, :, 0, :] - edges[:, :, 1, :])**2, axis=2))
        circum_r = lengths[0,:]*lengths[1,:]*lengths[2,:] / (np.sqrt((lengths[0,:]+lengths[1,:]+lengths[2,:])*(-lengths[0,:]+lengths[1,:]+lengths[2,:])*(lengths[0,:]-lengths[1,:]+lengths[2,:])*(lengths[0,:]+lengths[1,:]-lengths[2,:])))
        valid = tri.simplices[np.nan_to_num(circum_r) < alpha]
        if len(valid) == 0: return None
        hull_pts_3d = np.concatenate([points_2d, np.zeros((len(points_2d), 1))], axis=1)
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(hull_pts_3d), o3d.utility.Vector3iVector(valid))
        return o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=0.3)
    except: return None

def apply_house_cookie_cutter(ag_pts, ag_colors, voxel_grid):
    contain = voxel_grid.check_if_included(o3d.utility.Vector3dVector(ag_pts * [1., 1., 0.]))
    f_pts, f_cols = ag_pts[contain], ag_colors[contain]
    if len(f_pts) < 100: return None
    
    mins, maxs = voxel_grid.get_min_bound(), voxel_grid.get_max_bound()
    res, cell_a = 0.6, 0.36
    vol, area = 0.0, 0.0
    grid = np.array([[x, y, 0.] for x in np.arange(mins[0], maxs[0], res) for y in np.arange(mins[1], maxs[1], res)])
    
    if len(grid) > 0:
        f_mask = voxel_grid.check_if_included(o3d.utility.Vector3dVector(grid))
        valid_grid = grid[f_mask]
        area = len(valid_grid) * cell_a
        avg_h = f_pts[:, 2].mean()
        for g_pt in valid_grid:
            col_m = (f_pts[:, 0] >= g_pt[0]) & (f_pts[:, 0] < g_pt[0]+res) & (f_pts[:, 1] >= g_pt[1]) & (f_pts[:, 1] < g_pt[1]+res)
            vol += (cell_a * f_pts[col_m, 2].mean()) if np.any(col_m) else (cell_a * avg_h)

        # --- RESTORE SYNTHETIC FLOOR ---
        floor_pts = valid_grid.copy()
        floor_cols = np.full((len(floor_pts), 3), [0.4, 0.4, 0.4]) # Gray
        combined_pts = np.vstack([f_pts, floor_pts])
        combined_cols = np.vstack([f_cols, floor_cols])
    else:
        return None

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(combined_pts))
    pcd.colors = o3d.utility.Vector3dVector(combined_cols)
    return pcd, area, vol

# --- 3. PARALLEL WORKER ---

def worker_extraction(args):
    idx, seeds, local_pts, local_cols, gz, rot, off_xy, g_off, img_data, out_dir = args
    
    # Local Base-Trim: Dissolve ground bridges survived by RANSAC
    local_mask = local_pts[:, 2] > 0.20
    local_pts, local_cols = local_pts[local_mask], local_cols[local_mask]
    if len(local_pts) < 100: return []
    
    # RANSAC Plane Refinement (Find the actual roof)
    valid_roof_pts = []
    rem = seeds.copy()
    while len(rem) > 100:
        tmp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rem))
        _, inliers = tmp.segment_plane(0.25, 3, 250)
        if len(inliers) < 100: break
        valid_roof_pts.append(rem[inliers])
        rem = np.delete(rem, inliers, axis=0)
    
    if not valid_roof_pts: return []
    pure = np.vstack(valid_roof_pts)
    b_labels = DBSCAN(eps=2.0, min_samples=15).fit(pure[:, :2]).labels_
    
    houses = []
    for b_id in range(b_labels.max() + 1):
        footprint = calculate_alpha_shape(pure[b_labels == b_id][:, :2])
        if not footprint: continue
        
        res = apply_house_cookie_cutter(local_pts, local_cols, footprint)
        if res:
            p, a, v = res
            p.translate((0, 0, gz))
            p.translate((off_xy[0], off_xy[1], 0))
            p.rotate(rot.T, center=(0, 0, 0))
            
            cent = np.mean(np.asarray(p.points), axis=0)
            gx, gy = cent[0] + g_off[0], cent[1] + g_off[1]
            uid = f"H_{abs(gx):.3f}_{abs(gy):.3f}".replace('.', 'd')
            
            img_p = "N/A"
            if img_data[1]:
                _, i_idx = img_data[1].query([gx, gy], k=1)
                img_p = img_data[0][i_idx]
                shutil.copy2(img_p, os.path.join(out_dir, "best_images", f"{uid}.jpg"))
            
            o3d.io.write_point_cloud(os.path.join(out_dir, "individual_houses", f"{uid}.ply"), p)
            houses.append({"house_ID": uid, "Area_sqft": round(a*10.76, 2), "Volume_cuft": round(v*35.31, 2), "Best_Image": img_p})
    return houses

# --- 4. MASTER PIPELINE ---

def process_reconstruction_v3_7(project_path):
    global_start_time = time.time()
    out = os.path.join(project_path, "final_analysis_v3_7")
    for d in ["individual_houses", "best_images", "diagnostics"]: os.makedirs(os.path.join(out, d), exist_ok=True)

    trans = Transformer.from_crs("EPSG:4326", EPSG_CODE, always_xy=True)
    print("[1/7] Syncing Global Offsets...")
    g_off = auto_detect_offsets(project_path)
    img_data = build_spatial_image_index(os.path.join(project_path, 'images'), trans)

    print("[2/7] Loading and cleaning PLY...")
    t0 = time.time()
    pcd = o3d.io.read_point_cloud(repair_openmvs_ply_colors(os.path.join(project_path, 'scene_dense.ply')))
    pcd = pcd.voxel_down_sample(0.05)
    pcd, _ = pcd.remove_statistical_outlier(20, 2.2)

    print("[3/7] Leveling scene geometry...")
    pts = np.asarray(pcd.points)
    grnd = pts[pts[:, 2] < np.percentile(pts[:, 2], 30)]
    centroid = np.mean(grnd, axis=0)
    _, evecs = np.linalg.eigh(np.cov(grnd.T))
    norm = evecs[:, 0] if evecs[2, 0] > 0 else -evecs[:, 0]
    v = np.cross(norm, [0, 0, 1])
    s, c = np.linalg.norm(v), np.dot(norm, [0, 0, 1])
    rot = np.eye(3)
    if s > 1e-6:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rot = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s**2))
        pcd.rotate(rot, center=(0,0,0))
    off_xy = centroid[:2]
    pcd.translate((-off_xy[0], -off_xy[1], 0))
    o3d.io.write_point_cloud(os.path.join(out, "diagnostics", "step3_leveled.ply"), pcd)

    print("[4/7] Multi-Tier Ground Cut & SOR...")
    # Aggressive RANSAC (0.35m)
    _, inliers = pcd.segment_plane(distance_threshold=0.35, ransac_n=3, num_iterations=1000)
    gz = np.median(np.asarray(pcd.select_by_index(inliers).points)[:, 2])
    ag_pcd = pcd.select_by_index(inliers, invert=True)
    
    # Statistical filter to snap ground bridges
    ag_pcd, _ = ag_pcd.remove_statistical_outlier(nb_neighbors=12, std_ratio=1.2)
    ag_pcd.translate((0, 0, -gz))
    ag_pts, ag_colors = np.asarray(ag_pcd.points), np.asarray(ag_pcd.colors)
    o3d.io.write_point_cloud(os.path.join(out, "diagnostics", "step4_above_ground.ply"), ag_pcd)

    print("[5/7] Geometric Consistency Filtering...")
    mean_h, std_h = np.mean(ag_pts[:, 2]), np.std(ag_pts[:, 2])
    r, g, b = ag_colors[:, 0], ag_colors[:, 1], ag_colors[:, 2]
    exg = (2 * g) - r - b
    cand_mask = (exg > 0.07) | ((ag_pts[:, 2] - mean_h) / std_h > 2.5)
    cand_idx = np.where(cand_mask)[0]
    tree_mask = (exg > 0.07).copy()
    
    if len(cand_idx) > 0:
        tree = KDTree(ag_pts)
        neighbors = tree.query_ball_point(ag_pts[cand_idx], r=0.8, workers=-1)
        unique_n = np.unique(np.concatenate(neighbors))
        _, n_idx_l = tree.query(ag_pts[unique_n], k=32, workers=-1)
        z_v = ag_pts[n_idx_l][:, :, 2]
        is_chaos = (np.ptp(z_v, axis=1) > 0.35) | (np.std(z_v, axis=1) > 0.08)
        tree_mask[unique_n[is_chaos]] = True
    
    pruned_ag_pcd = ag_pcd.select_by_index(np.where(~tree_mask)[0])
    ag_pts, ag_colors = np.asarray(pruned_ag_pcd.points), np.asarray(pruned_ag_pcd.colors)
    o3d.io.write_point_cloud(os.path.join(out, "diagnostics", "step5b_pruned_rgb.ply"), pruned_ag_pcd)

    print("[6/7] Broad Clustering & Slenderness Firewall...")
    h_idx = np.where(ag_pts[:, 2] > 2.2)[0]
    h_pts = ag_pts[h_idx]
    kdt_h = KDTree(h_pts)
    _, n_idx = kdt_h.query(h_pts, k=12, workers=-1)
    p_mask = np.std(h_pts[n_idx][:, :, 2], axis=1) < 0.20
    seeds = h_pts[p_mask]
    
    # Balanced Z-Squish (0.05)
    seeds_sq = np.copy(seeds); seeds_sq[:, 2] *= 0.05
    db = DBSCAN(eps=2.2, min_samples=15, n_jobs=-1).fit(seeds_sq)
    labels = db.labels_

    worker_args = []
    for i in range(labels.max() + 1):
        idx = np.where(labels == i)[0]
        blob = seeds[idx]
        m, M = blob.min(axis=0), blob.max(axis=0)
        w, d, h = max(M[0]-m[0], 0.1), max(M[1]-m[1], 0.1), M[2]-m[2]
        if h / np.sqrt(w*d) > 2.5 or len(idx) < 150: continue # Morphology Check
        
        buf = 5.0
        mask = (ag_pts[:, 0] >= m[0]-buf) & (ag_pts[:, 0] <= M[0]+buf) & (ag_pts[:, 1] >= m[1]-buf) & (ag_pts[:, 1] <= M[1]+buf)
        worker_args.append((i, blob, ag_pts[mask], ag_colors[mask], gz, rot, off_xy, g_off, img_data, out))

    print(f"[7/7] Parallel Extraction ({len(worker_args)} buildings)...")
    final_res = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(NUM_CORES, 32)) as executor:
        futures = [executor.submit(worker_extraction, a) for a in worker_args]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res: final_res.extend(res)

    with open(os.path.join(out, "house_measurements.csv"), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["house_ID", "Area_sqft", "Volume_cuft", "Best_Image"])
        writer.writeheader()
        writer.writerows(final_res)
    
    print(f"SUCCESS. Found {len(final_res)} buildings in {time.time()-global_start_time:.2f}s")

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(1)
    process_reconstruction_v3_7(os.path.abspath(sys.argv[1]))