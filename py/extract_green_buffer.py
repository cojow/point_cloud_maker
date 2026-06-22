import os
import json
import numpy as np
import open3d as o3d
import csv
import sys
import time
import shutil
import glob
import cv2 
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
    if not image_dir or not os.path.exists(image_dir): 
        return None, None
    files = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.JPG"))
    for p in files:
        try:
            exif = Image.open(p)._getexif()
            if not exif: 
                continue
            gps = {GPSTAGS.get(t, t): v for t, v in exif[list(TAGS.keys())[list(TAGS.values()).index('GPSInfo')]].items()}
            def to_d(v): return float(v[0]) + float(v[1])/60.0 + float(v[2])/3600.0
            lat, lon = to_d(gps['GPSLatitude']), to_d(gps['GPSLongitude'])
            if gps['GPSLatitudeRef'] != 'N': 
                lat = -lat
            if gps['GPSLongitudeRef'] != 'E': 
                lon = -lon
            paths.append(p) 
            coords.append(transformer.transform(lon, lat))
        except: 
            continue
    return (paths, KDTree(np.array(coords))) if coords else (None, None)

def repair_openmvs_ply_colors(ply_path):
    with open(ply_path, 'rb') as f: 
        header_chunk = f.read(2000)
    if b'diffuse_red' in header_chunk:
        fixed_path = ply_path.replace('.ply', '_color_fixed.ply')
        with open(ply_path, 'rb') as f: 
            content = f.read()
        content = content.replace(b'property uchar diffuse_red', b'property uchar red        ')
        content = content.replace(b'property uchar diffuse_green', b'property uchar green      ')
        content = content.replace(b'property uchar diffuse_blue', b'property uchar blue       ')
        with open(fixed_path, 'wb') as f: 
            f.write(content)
        return fixed_path
    return ply_path

# --- 2. EXTRACTION & CROPPING HELPERS ---
def calculate_alpha_shape(points_2d, alpha=1.2):
    if len(points_2d) < 4: 
        return None
    try:
        tri = Delaunay(points_2d)
        edges = np.array([points_2d[tri.simplices[:, [0, 1]]], points_2d[tri.simplices[:, [1, 2]]], points_2d[tri.simplices[:, [2, 0]]]])
        lengths = np.sqrt(np.sum((edges[:, :, 0, :] - edges[:, :, 1, :])**2, axis=2))
        circum_r = lengths[0,:]*lengths[1,:]*lengths[2,:] / (np.sqrt((lengths[0,:]+lengths[1,:]+lengths[2,:])*(-lengths[0,:]+lengths[1,:]+lengths[2,:])*(lengths[0,:]-lengths[1,:]+lengths[2,:])*(lengths[0,:]+lengths[1,:]-lengths[2,:])))
        valid = tri.simplices[np.nan_to_num(circum_r) < alpha]
        if len(valid) == 0: 
            return None
        hull_pts = np.concatenate([points_2d, np.zeros((len(points_2d), 1))], axis=1)
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(hull_pts), o3d.utility.Vector3iVector(valid))
        return o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=0.3), valid, hull_pts
    except: 
        return None

def apply_house_cookie_cutter(ag_pts, ag_colors, footprint_data):
    voxel_grid, _, _ = footprint_data
    contain = voxel_grid.check_if_included(o3d.utility.Vector3dVector(ag_pts * [1., 1., 0.]))
    f_pts, f_cols = ag_pts[contain], ag_colors[contain]
    if len(f_pts) < 100: 
        return None
    
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
    else: 
        return None

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(combined_pts))
    pcd.colors = o3d.utility.Vector3dVector(combined_cols)
    return pcd, area, vol

def load_opensfm_camera(reconstruction_file, image_filename):
    with open(reconstruction_file, 'r') as f:
        reconstruction = json.load(f)[0] 
        
    if image_filename not in reconstruction['shots']:
        raise ValueError(f"Image '{image_filename}' not found in reconstruction data.")
        
    shot = reconstruction['shots'][image_filename]
    camera_id = shot['camera']
    camera = reconstruction['cameras'][camera_id]
    
    rvec = np.array(shot['rotation'], dtype=float)
    tvec = np.array(shot['translation'], dtype=float)
    
    w = camera['width']
    h = camera['height']
    max_size = max(w, h)
    
    focal = camera.get('focal', 0.8) 
    fx = focal * max_size
    fy = focal * max_size
    
    c_x = camera.get('c_x', 0.0)
    c_y = camera.get('c_y', 0.0)
    cx = (w / 2.0) + (c_x * max_size)
    cy = (h / 2.0) + (c_y * max_size)
    
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=float)
    
    dist_coeffs = np.array([
        camera.get('k1', 0.0),
        camera.get('k2', 0.0),
        camera.get('p1', 0.0),
        camera.get('p2', 0.0),
        camera.get('k3', 0.0)
    ], dtype=float)
    
    return rvec, tvec, camera_matrix, dist_coeffs

def get_3d_bounding_box(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    aabb = pcd.get_axis_aligned_bounding_box()
    return np.asarray(aabb.get_box_points())

# --- 3. PARALLEL WORKER ---
def worker_extraction(args):
    idx, seeds, local_pts, local_cols, gz, rot, off_xy, g_off, img_data, out_dir = args
    local_mask = local_pts[:, 2] > 0.20
    local_pts, local_cols = local_pts[local_mask], local_cols[local_mask]
    if len(local_pts) < 100: 
        return []
    
    valid_roof_pts = []
    rem = seeds.copy()
    while len(rem) > 100:
        tmp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rem))
        _, inliers = tmp.segment_plane(0.25, 3, 250)
        if len(inliers) < 100: 
            break
        valid_roof_pts.append(rem[inliers])
        rem = np.delete(rem, inliers, axis=0)
    
    if not valid_roof_pts: 
        return []
    pure = np.vstack(valid_roof_pts)
    b_labels = DBSCAN(eps=2.0, min_samples=15).fit(pure[:, :2]).labels_
    
    houses = []
    for b_id in range(b_labels.max() + 1):
        footprint = calculate_alpha_shape(pure[b_labels == b_id][:, :2])
        if not footprint: 
            continue
        res = apply_house_cookie_cutter(local_pts, local_cols, footprint)
        if res:
            p, a, v = res
            p.translate((0, 0, gz))
            p.translate((off_xy[0], off_xy[1], 0))
            p.rotate(rot.T, center=(0, 0, 0))
            
            z_vals = np.asarray(p.points)[:, 2]
            height_m = z_vals.max() - z_vals.min()
            height_ft = height_m * 3.28084
            area_sqft = a * 10.76
            vol_cuft = v * 35.31
            
            ratio = area_sqft / height_ft if height_ft > 0 else 0
            
            cent = np.mean(np.asarray(p.points), axis=0)
            gx, gy = cent[0] + g_off[0], cent[1] + g_off[1]
            
            temp_uid = f"H_{abs(gx):.3f}_{abs(gy):.3f}".replace('.', 'd')
            
            orig_img_name = "N/A"
            if img_data[1]:
                _, i_idx = img_data[1].query([gx, gy], k=1)
                img_p = img_data[0][i_idx]
                orig_img_name = os.path.basename(img_p) 
                shutil.copy2(img_p, os.path.join(out_dir, "best_images", f"{temp_uid}.jpg"))
                
            o3d.io.write_point_cloud(os.path.join(out_dir, "individual_houses", f"{temp_uid}.ply"), p)
            houses.append({
                "temp_ID": temp_uid, 
                "Area_sqft": round(area_sqft, 2), 
                "Volume_cuft": round(vol_cuft, 2), 
                "Height_ft": round(height_ft, 2),
                "Ratio_Area_to_Height": round(ratio, 2),
                "X_coord": gx,
                "Y_coord": gy,
                "Original_Image": orig_img_name 
            })
    return houses

# --- 4. MASTER PIPELINE ---
def process_reconstruction_v4_2(project_path):
    global_start_time = time.time()
    project_name = os.path.basename(os.path.normpath(project_path))
    out = os.path.join(project_path, f"analysis_{project_name}_v4_2")
    
    for d in ["individual_houses", "best_images", "best_images_cropped", "diagnostics"]: 
        os.makedirs(os.path.join(out, d), exist_ok=True)

    print("[1/9] Syncing Global Offsets...")
    g_off = auto_detect_offsets(project_path)
    img_data = build_spatial_image_index(os.path.join(project_path, 'images'), Transformer.from_crs("EPSG:4326", EPSG_CODE, always_xy=True))

    print("[2/9] Loading and cleaning PLY...")
    pcd = o3d.io.read_point_cloud(repair_openmvs_ply_colors(os.path.join(project_path, 'scene_dense.ply')))
    pcd = pcd.voxel_down_sample(0.05)
    pcd, _ = pcd.remove_statistical_outlier(20, 2.2)

    print("[3/9] Leveling scene geometry...")
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

    print("[4/9] Multi-Tier Ground Cut & SOR...")
    _, inliers = pcd.segment_plane(0.35, 3, 1000)
    gz = np.median(np.asarray(pcd.select_by_index(inliers).points)[:, 2])
    ag_pcd = pcd.select_by_index(inliers, invert=True)
    ag_pcd, _ = ag_pcd.remove_statistical_outlier(12, 1.2)
    ag_pcd.translate((0, 0, -gz))
    ag_pts, ag_colors = np.asarray(ag_pcd.points), np.asarray(ag_pcd.colors)

    print("[5/9] Standard Geometric Consistency...")
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

    print("[6/9] Morphological Erosion & Fragment Reconciliation...")
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

    # --- NEW: Pre-Extraction Agglomerative Reconciliation ---
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    
    label_sizes = {}
    label_coords = {}
    trees = {}
    
    # Pre-calculate sizes (2D area proxy) and build spatial trees
    for lbl in unique_labels:
        lbl_pts = seeds[labels == lbl]
        lbl_c = np.floor((lbl_pts[:, :2] - min_b) / res).astype(int)
        area_sqft = len(np.unique(lbl_c, axis=0)) * (res ** 2) * 10.764
        label_sizes[lbl] = area_sqft
        label_coords[lbl] = lbl_pts[:, :2]
        trees[lbl] = KDTree(label_coords[lbl])
        
    # Proper Union-Find map
    parent = {lbl: lbl for lbl in unique_labels}
    
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]
        
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            # Attach smaller to larger
            if label_sizes[root_i] < label_sizes[root_j]:
                parent[root_i] = root_j
                label_sizes[root_j] += label_sizes[root_i]
            else:
                parent[root_j] = root_i
                label_sizes[root_i] += label_sizes[root_j]
    
    # Evaluate pairwise intersections for fragments
    for lbl_a in unique_labels:
        for lbl_b in unique_labels:
            if lbl_a >= lbl_b: continue
            
            root_a = find(lbl_a)
            root_b = find(lbl_b)
            if root_a == root_b: continue
            
            size_a = label_sizes[root_a]
            size_b = label_sizes[root_b]
            
            # Condition: One must be an absolute fragment (<150 sqft) OR a relative fragment (<5%)
            is_a_frag = size_a < 150 or size_a < (0.05 * size_b)
            is_b_frag = size_b < 150 or size_b < (0.05 * size_a)
            
            if not (is_a_frag or is_b_frag):
                continue
                
            # Proximity check: ~2 feet overlap. If closest points are within 0.75m, they intersect.
            dists_ab, _ = trees[lbl_b].query(label_coords[lbl_a], k=1, distance_upper_bound=0.75)
            if np.any(dists_ab < 0.75):
                union(lbl_a, lbl_b)
                
    # Apply resolved labels back to the points
    for i in range(len(labels)):
        if labels[i] > 0:
            labels[i] = find(labels[i])
            
    final_unique_labels = np.unique(labels)
    final_unique_labels = final_unique_labels[final_unique_labels > 0]
    # --------------------------------------------------------

    worker_args = []
    for i in final_unique_labels:
        idx = np.where(labels == i)[0]
        blob = seeds[idx]
        if len(idx) < 150: continue
        m, M = blob.min(axis=0), blob.max(axis=0)
        mask = (ag_pts[:, 0] >= m[0]-5.0) & (ag_pts[:, 0] <= M[0]+5.0) & (ag_pts[:, 1] >= m[1]-5.0) & (ag_pts[:, 1] <= M[1]+5.0)
        worker_args.append((i, blob, ag_pts[mask], ag_colors[mask], gz, rot, off_xy, g_off, img_data, out))

    print(f"[7/9] Parallel Extraction ({len(worker_args)} candidates)...")
    raw_res = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(NUM_CORES, 32)) as ex:
        for res_list in ex.map(worker_extraction, worker_args):
            if res_list: raw_res.extend(res_list)

    print("[8/9] Artifact Purge & Sequential Georeferencing...")
    final_measurements = []
    lookup_table = []
    house_counter = 1
    
    reverse_trans = Transformer.from_crs(EPSG_CODE, "EPSG:4326", always_xy=True)
    
    if len(raw_res) > 0:
        ratios = np.array([r["Ratio_Area_to_Height"] for r in raw_res])
        q1, q3 = np.percentile(ratios, [25, 75])
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        
        eval_ratios = ratios[ratios <= upper_bound]
        
        if len(eval_ratios) >= 2:
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(eval_ratios.reshape(-1, 1))
            natural_break = np.mean(kmeans.cluster_centers_.flatten())
            threshold = min(natural_break, 20.0) 
        else:
            threshold = 10.0
            
        for r in raw_res:
            old_ply_path = os.path.join(out, "individual_houses", f"{r['temp_ID']}.ply")
            old_img_path = os.path.join(out, "best_images", f"{r['temp_ID']}.jpg")
            
            if r["Ratio_Area_to_Height"] > upper_bound or r["Ratio_Area_to_Height"] >= threshold:
                new_uid = f"{project_name}_{house_counter}"
                house_counter += 1
                
                new_ply_path = os.path.join(out, "individual_houses", f"{new_uid}.ply")
                new_img_path = os.path.join(out, "best_images", f"{new_uid}.jpg")
                
                if os.path.exists(old_ply_path): os.rename(old_ply_path, new_ply_path)
                if os.path.exists(old_img_path): os.rename(old_img_path, new_img_path)
                
                lon, lat = reverse_trans.transform(r['X_coord'], r['Y_coord'])
                
                final_measurements.append({
                    "house_ID": new_uid,
                    "Area_sqft": r["Area_sqft"],
                    "Volume_cuft": r["Volume_cuft"],
                    "Height_ft": r["Height_ft"],
                    "Ratio_Area_to_Height": r["Ratio_Area_to_Height"],
                    "Best_Image": new_img_path if os.path.exists(new_img_path) else "N/A"
                })
                
                lookup_table.append({
                    "house_ID": new_uid,
                    "X_UTM": round(r['X_coord'], 3),
                    "Y_UTM": round(r['Y_coord'], 3),
                    "Latitude": round(lat,6),
                    "Longitude": round(lon,6),
                    "Original_Image": r.get("Original_Image", "N/A") 
                })
            else:
                if os.path.exists(old_ply_path): os.remove(old_ply_path)
                if os.path.exists(old_img_path): os.remove(old_img_path)

    # --- Phase 9: Projection and Cropping ---
    print("[9/9] Precision Cropping Images...")
    reconstruction_file = os.path.join(project_path, 'reconstruction.json')
    
    for record in lookup_table:
        house_id = record["house_ID"]
        orig_name = record["Original_Image"]
        
        if orig_name == "N/A":
            continue
            
        ply_path = os.path.join(out, "individual_houses", f"{house_id}.ply")
        img_path = os.path.join(out, "best_images", f"{house_id}.jpg")
        crop_out_path = os.path.join(out, "best_images_cropped", f"{house_id}.jpg")
        
        if os.path.exists(ply_path) and os.path.exists(img_path) and os.path.exists(reconstruction_file):
            try:
                rvec, tvec, K, dist = load_opensfm_camera(reconstruction_file, orig_name)
                corners_3d = get_3d_bounding_box(ply_path)
                corners_2d, _ = cv2.projectPoints(corners_3d, rvec, tvec, K, dist)
                corners_2d = corners_2d.reshape(-1, 2)
                
                x_min = int(np.floor(np.min(corners_2d[:, 0])))
                x_max = int(np.ceil(np.max(corners_2d[:, 0])))
                y_min = int(np.floor(np.min(corners_2d[:, 1])))
                y_max = int(np.ceil(np.max(corners_2d[:, 1])))
                
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    padding = 60
                    
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)
                    
                    cropped_img = img[y_min:y_max, x_min:x_max]
                    cv2.imwrite(crop_out_path, cropped_img)
            except Exception as e:
                print(f"      [!] Failed to crop {house_id}: {e}")

    # Output Measurements
    with open(os.path.join(out, "measurements.csv"), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["house_ID", "Area_sqft", "Volume_cuft", "Height_ft", "Ratio_Area_to_Height", "Best_Image"])
        writer.writeheader()
        writer.writerows(final_measurements)
        
    # Output Location Lookup Table
    with open(os.path.join(out, "location_lookup.csv"), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["house_ID", "X_UTM", "Y_UTM", "Latitude", "Longitude", "Original_Image"])
        writer.writeheader()
        writer.writerows(lookup_table)
        
    print(f"SUCCESS. Validated and cropped {len(final_measurements)} buildings in {time.time()-global_start_time:.2f}s")

if __name__ == "__main__":
    if len(sys.argv) < 2: 
        sys.exit(1)
    process_reconstruction_v4_2(os.path.abspath(sys.argv[1]))