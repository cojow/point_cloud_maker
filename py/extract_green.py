import os
import json
import numpy as np
import open3d as o3d
import csv
import sys
import time
import shutil
import glob
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import Delaunay, KDTree
from scipy.ndimage import binary_erosion, label
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

EPSG_CODE = "EPSG:32612"  
NUM_CORES = os.cpu_count() or 1

# --- 1. AUTOMATION HELPERS ---
def auto_detect_offsets(project_dir):
    json_path = os.path.join(project_dir, 'reconstruction.json')
    if not os.path.exists(json_path): return 0.0, 0.0
    try:
        with open(json_path, 'r') as f: data = json.load(f)
        ref = data[0].get('reference_lla')
        trans = Transformer.from_crs("EPSG:4326", EPSG_CODE, always_xy=True)
        return trans.transform(ref['longitude'], ref['latitude'])
    except: return 0.0, 0.0

def build_spatial_image_index(image_dir, transformer):
    paths, coords = [], []
    if not image_dir or not os.path.exists(image_dir): return None, None
    files = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.JPG"))
    for p in files:
        try:
            exif = Image.open(p)._getexif()
            if not exif: continue
            gps = {GPSTAGS.get(t, t): v for t, v in exif[list(TAGS.keys())[list(TAGS.values()).index('GPSInfo')]].items()}
            def to_d(v): return float(v[0]) + float(v[1])/60.0 + float(v[2])/3600.0
            lat, lon = to_d(gps['GPSLatitude']), to_d(gps['GPSLongitude'])
            if gps['GPSLatitudeRef'] != 'N': lat = -lat
            if gps['GPSLongitudeRef'] != 'E': lon = -lon
            paths.append(p); coords.append(transformer.transform(lon, lat))
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

# --- 2. EXTRACTION HELPERS ---
def calculate_alpha_shape(points_2d, alpha=1.2):
    if len(points_2d) < 4: return None
    try:
        tri = Delaunay(points_2d)
        edges = np.array([points_2d[tri.simplices[:, [0, 1]]], points_2d[tri.simplices[:, [1, 2]]], points_2d[tri.simplices[:, [2, 0]]]])
        lengths = np.sqrt(np.sum((edges[:, :, 0, :] - edges[:, :, 1, :])**2, axis=2))
        circum_r = lengths[0,:]*lengths[1,:]*lengths[2,:] / (np.sqrt((lengths[0,:]+lengths[1,:]+lengths[2,:])*(-lengths[0,:]+lengths[1,:]+lengths[2,:])*(lengths[0,:]-lengths[1,:]+lengths[2,:])*(lengths[0,:]+lengths[1,:]-lengths[2,:])))
        valid = tri.simplices[np.nan_to_num(circum_r) < alpha]
        if len(valid) == 0: return None
        hull_pts = np.concatenate([points_2d, np.zeros((len(points_2d), 1))], axis=1)
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(hull_pts), o3d.utility.Vector3iVector(valid))
        return o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=0.3), valid, hull_pts
    except: return None

def apply_house_cookie_cutter(ag_pts, ag_colors, footprint_data):
    voxel_grid, _, _ = footprint_data
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
            
        floor_pts = valid_grid.copy()
        floor_cols = np.full((len(floor_pts), 3), [0.4, 0.4, 0.4])
        combined_pts, combined_cols = np.vstack([f_pts, floor_pts]), np.vstack([f_cols, floor_cols])
    else: return None

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(combined_pts))
    pcd.colors = o3d.utility.Vector3dVector(combined_cols)
    return pcd, area, vol

# --- 3. PARALLEL WORKER ---
def worker_extraction(args):
    idx, seeds, local_pts, local_cols, gz, rot, off_xy, g_off, img_data, out_dir = args
    local_mask = local_pts[:, 2] > 0.20
    local_pts, local_cols = local_pts[local_mask], local_cols[local_mask]
    if len(local_pts) < 100: return []
    
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
            
            # --- HEIGHT AND RATIO MATH ---
            z_vals = np.asarray(p.points)[:, 2]
            height_m = z_vals.max() - z_vals.min()
            height_ft = height_m * 3.28084
            area_sqft = a * 10.76
            vol_cuft = v * 35.31
            
            ratio = area_sqft / height_ft if height_ft > 0 else 0
            
            cent = np.mean(np.asarray(p.points), axis=0)
            gx, gy = cent[0] + g_off[0], cent[1] + g_off[1]
            uid = f"H_{abs(gx):.3f}_{abs(gy):.3f}".replace('.', 'd')
            img_p = "N/A"
            if img_data[1]:
                _, i_idx = img_data[1].query([gx, gy], k=1)
                img_p = img_data[0][i_idx]
                shutil.copy2(img_p, os.path.join(out_dir, "best_images", f"{uid}.jpg"))
            o3d.io.write_point_cloud(os.path.join(out_dir, "individual_houses", f"{uid}.ply"), p)
            houses.append({
                "house_ID": uid, 
                "Area_sqft": round(area_sqft, 2), 
                "Volume_cuft": round(vol_cuft, 2), 
                "Height_ft": round(height_ft, 2),
                "Ratio_Area_to_Height": round(ratio, 2),
                "Best_Image": img_p
            })
    return houses

# --- 4. MASTER PIPELINE ---
def process_reconstruction_v4_1(project_path):
    global_start_time = time.time()
    out = os.path.join(project_path, "analysis_v4_1")
    for d in ["individual_houses", "best_images", "diagnostics"]: os.makedirs(os.path.join(out, d), exist_ok=True)

    print("[1/8] Syncing Global Offsets...")
    g_off = auto_detect_offsets(project_path)
    img_data = build_spatial_image_index(os.path.join(project_path, 'images'), Transformer.from_crs("EPSG:4326", EPSG_CODE, always_xy=True))

    print("[2/8] Loading and cleaning PLY...")
    pcd = o3d.io.read_point_cloud(repair_openmvs_ply_colors(os.path.join(project_path, 'scene_dense.ply')))
    pcd = pcd.voxel_down_sample(0.05)
    pcd, _ = pcd.remove_statistical_outlier(20, 2.2)

    print("[3/8] Leveling scene geometry...")
    pts = np.asarray(pcd.points)
    grnd = pts[pts[:, 2] < np.percentile(pts[:, 2], 30)]
    _, evecs = np.linalg.eigh(np.cov(grnd.T))
    norm = evecs[:, 0] if evecs[2, 0] > 0 else -evecs[:, 0]
    v = np.cross(norm, [0, 0, 1])
    s, c = np.linalg.norm(v), np.dot(norm, [0, 0, 1])
    rot = np.eye(3)
    if s > 1e-6:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rot = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s**2))
    pcd.rotate(rot, center=(0,0,0))
    off_xy = np.mean(grnd, axis=0)[:2]
    pcd.translate((-off_xy[0], -off_xy[1], 0))

    print("[4/8] Multi-Tier Ground Cut & SOR...")
    _, inliers = pcd.segment_plane(0.35, 3, 1000)
    gz = np.median(np.asarray(pcd.select_by_index(inliers).points)[:, 2])
    ag_pcd = pcd.select_by_index(inliers, invert=True)
    ag_pcd, _ = ag_pcd.remove_statistical_outlier(12, 1.2)
    ag_pcd.translate((0, 0, -gz))
    ag_pts, ag_colors = np.asarray(ag_pcd.points), np.asarray(ag_pcd.colors)

    print("[5/8] Standard Geometric Consistency...")
    mean_h, std_h = np.mean(ag_pts[:, 2]), np.std(ag_pts[:, 2])
    r, g, b = ag_colors[:, 0], ag_colors[:, 1], ag_colors[:, 2]
    exg = (2 * g) - r - b
    cand_idx = np.where((exg > 0.07) | ((ag_pts[:, 2] - mean_h) / std_h > 2.5))[0]
    tree_mask = (exg > 0.07).copy()
    
    if len(cand_idx) > 0:
        tree = KDTree(ag_pts)
        unique_n = np.unique(np.concatenate(tree.query_ball_point(ag_pts[cand_idx], 0.8, workers=-1)))
        _, n_idx_l = tree.query(ag_pts[unique_n], k=32, workers=-1)
        z_v = ag_pts[n_idx_l][:, :, 2]
        tree_mask[unique_n[(np.ptp(z_v, axis=1) > 0.35) | (np.std(z_v, axis=1) > 0.08)]] = True
    
    pruned_ag_pcd = ag_pcd.select_by_index(np.where(~tree_mask)[0])
    ag_pts, ag_colors = np.asarray(pruned_ag_pcd.points), np.asarray(pruned_ag_pcd.colors)
    o3d.io.write_point_cloud(os.path.join(out, "diagnostics", "step5b_pruned_rgb.ply"), pruned_ag_pcd)

    print("[6/8] Morphological Erosion (Bridge Snapper)...")
    final_pts = np.asarray(pruned_ag_pcd.points)
    seeds = final_pts[np.where(final_pts[:, 2] > 2.2)[0]]
    
    res = 0.25
    min_b, max_b = seeds[:, :2].min(axis=0), seeds[:, :2].max(axis=0)
    grid_shape = np.ceil((max_b - min_b) / res).astype(int) + 1
    coords = np.floor((seeds[:, :2] - min_b) / res).astype(int)
    
    occupancy_grid = np.zeros(grid_shape, dtype=bool)
    occupancy_grid[coords[:, 0], coords[:, 1]] = True
    
    eroded_grid = binary_erosion(occupancy_grid, iterations=3)
    labeled_grid, num_buildings = label(eroded_grid)
    
    core_mask = eroded_grid[coords[:, 0], coords[:, 1]]
    core_pts = seeds[core_mask]
    core_labels = labeled_grid[coords[core_mask, 0], coords[core_mask, 1]]
    
    if len(core_pts) < 100:
        return print("      [!] Error: Erosion removed all structures.")
        
    core_tree = KDTree(core_pts[:, :2])
    dists, nearest_core_idx = core_tree.query(seeds[:, :2], k=1, workers=-1)
    
    labels = core_labels[nearest_core_idx]
    labels[dists > 2.0] = 0 

    worker_args = []
    for i in range(1, num_buildings + 1):
        idx = np.where(labels == i)[0]
        blob = seeds[idx]
        if len(idx) < 150: continue
        m, M = blob.min(axis=0), blob.max(axis=0)
        mask = (ag_pts[:, 0] >= m[0]-5.0) & (ag_pts[:, 0] <= M[0]+5.0) & (ag_pts[:, 1] >= m[1]-5.0) & (ag_pts[:, 1] <= M[1]+5.0)
        worker_args.append((i, blob, ag_pts[mask], ag_colors[mask], gz, rot, off_xy, g_off, img_data, out))

    print(f"[7/8] Parallel Extraction ({len(worker_args)} candidates)...")
    raw_res = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(NUM_CORES, 32)) as ex:
        for res_list in ex.map(worker_extraction, worker_args):
            if res_list: raw_res.extend(res_list)

    print(f"[8/8] Data-Driven Artifact Purge (Ratio Shield + 1D Clustering)...")
    final_res = []
    if len(raw_res) > 0:
        # We now analyze the Area/Height Ratio instead of raw Area
        ratios = np.array([r["Ratio_Area_to_Height"] for r in raw_res])
        
        # 1. THE IQR SHIELD (Prevents massive commercial ratios from pulling the math)
        q1, q3 = np.percentile(ratios, [25, 75])
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        
        # 2. ISOLATE NON-COMMERCIAL DATA
        eval_ratios = ratios[ratios <= upper_bound]
        
        # 3. 1D CLUSTERING (JENKS EQUIVALENT)
        if len(eval_ratios) >= 2:
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(eval_ratios.reshape(-1, 1))
            centers = kmeans.cluster_centers_.flatten()
            natural_break = np.mean(centers)
            # Fail-safe ceiling: We cap the deletion ratio at 20.0 to protect valid sheds
            threshold = min(natural_break, 20.0) 
        else:
            threshold = 10.0 # Fallback
            
        print(f"      -> IQR Shield Boundary: {upper_bound:.2f} Ratio")
        print(f"      -> Computed Artifact Threshold: {threshold:.2f} Ratio")
        
        # 4. ENFORCE DELETION
        for r in raw_res:
            uid = r["house_ID"]
            ratio = r["Ratio_Area_to_Height"]
            
            if ratio > upper_bound or ratio >= threshold:
                final_res.append(r)
            else:
                ply_p = os.path.join(out, "individual_houses", f"{uid}.ply")
                img_p = os.path.join(out, "best_images", f"{uid}.jpg")
                if os.path.exists(ply_p): os.remove(ply_p)
                if os.path.exists(img_p): os.remove(img_p)
                
    with open(os.path.join(out, "measurements.csv"), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["house_ID", "Area_sqft", "Volume_cuft", "Height_ft", "Ratio_Area_to_Height", "Best_Image"])
        writer.writeheader()
        writer.writerows(final_res)
    print(f"SUCCESS. Validated {len(final_res)} buildings in {time.time()-global_start_time:.2f}s")

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(1)
    process_reconstruction_v4_1(os.path.abspath(sys.argv[1]))