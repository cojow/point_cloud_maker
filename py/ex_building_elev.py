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
import argparse
import requests
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import Delaunay, KDTree
from scipy.ndimage import binary_erosion, label
from scipy.interpolate import LinearNDInterpolator
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

# --- 1. TOPOGRAPHY ENGINE ---
class GroundElevationModel:
    """Handles absolute ground truth elevation data based on user mode selection."""
    def __init__(self, mode, pts=None, dem_path=None, reverse_transformer=None):
        self.mode = mode
        self.dem_path = dem_path
        self.ransac_z = 0.0
        self.interpolator = None
        
        if mode == 'ransac' and pts is not None:
            print("      -> Initializing Global RANSAC Plane...")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            _, inliers = pcd.segment_plane(0.35, 3, 1000)
            self.ransac_z = np.median(pts[inliers, 2])
            print(f"      -> Global Median Ground Elevation set at: {self.ransac_z:.2f}m")
            
        elif mode == 'usgs' and pts is not None and reverse_transformer is not None:
            print("      -> Querying USGS National Map EPQS via Sparse Grid...")
            self._build_api_interpolator(pts, reverse_transformer)
            
        elif mode == 'local':
            if not dem_path or not os.path.exists(dem_path):
                raise FileNotFoundError(f"Local DEM file not found at: {dem_path}")
            print(f"      -> Loaded Local DEM: {os.path.basename(dem_path)}")

    def _build_api_interpolator(self, pts, reverse_transformer):
        """Builds a local continuous DEM surface by batch querying an elevation API."""
        min_b, max_b = pts[:, :2].min(axis=0), pts[:, :2].max(axis=0)
        # Generate a 15m grid over the project area
        xs = np.arange(min_b[0] - 15, max_b[0] + 30, 15)
        ys = np.arange(min_b[1] - 15, max_b[1] + 30, 15)
        xx, yy = np.meshgrid(xs, ys)
        
        flat_x, flat_y = xx.flatten(), yy.flatten()
        lons, lats = reverse_transformer.transform(flat_x, flat_y)
        
        # Batch Open-Elevation Request (Faster than individual USGS pings for bulk)
        payload = {"locations": [{"latitude": lat, "longitude": lon} for lat, lon in zip(lats, lons)]}
        print(f"      -> Fetching {len(lats)} control points...")
        try:
            response = requests.post("https://api.open-elevation.com/api/v1/lookup", json=payload, timeout=30).json()
            z_vals = np.array([res['elevation'] for res in response['results']])
            self.interpolator = LinearNDInterpolator(list(zip(flat_x, flat_y)), z_vals)
            print("      -> Dynamic terrain surface interpolated successfully.")
        except Exception as e:
            print(f"      [!] API Fetch failed ({e}). Falling back to global RANSAC.")
            self.mode = 'ransac'
            self.ransac_z = np.percentile(pts[:, 2], 5)

    def get_z(self, x, y):
        """Returns the true ground elevation for any given array of X, Y coordinates."""
        if self.mode == 'ransac':
            return np.full_like(x, self.ransac_z, dtype=float)
            
        elif self.mode == 'local':
            import rasterio
            with rasterio.open(self.dem_path) as src:
                # Sample the GeoTIFF at the given XY coordinates
                coords = [(xi, yi) for xi, yi in zip(x, y)]
                return np.array([val[0] for val in src.sample(coords)])
                
        elif self.mode == 'usgs':
            return self.interpolator((x, y))

# --- 2. AUTOMATION & EXTRACTION HELPERS ---
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

def apply_house_cookie_cutter(ag_pts, ag_colors, footprint_data, true_ground_elevation):
    voxel_grid, _, _ = footprint_data
    contain = voxel_grid.check_if_included(o3d.utility.Vector3dVector(ag_pts * [1., 1., 0.]))
    f_pts, f_cols = ag_pts[contain], ag_colors[contain]
    if len(f_pts) < 100: 
        return None
    
    mins, maxs = voxel_grid.get_min_bound(), voxel_grid.get_max_bound()
    res, cell_a = 0.6, 0.36
    vol, area = 0.0, 0.0
    
    # Generate the 2D footprint grid at Z=0 for intersection math
    grid = np.array([[x, y, 0.0] for x in np.arange(mins[0], maxs[0], res) for y in np.arange(mins[1], maxs[1], res)])
    
    if len(grid) > 0:
        f_mask = voxel_grid.check_if_included(o3d.utility.Vector3dVector(grid))
        valid_grid = grid[f_mask]
        area = len(valid_grid) * cell_a
        
        floor_pts = []
        for g_pt in valid_grid:
            col_m = (f_pts[:, 0] >= g_pt[0]) & (f_pts[:, 0] < g_pt[0]+res) & (f_pts[:, 1] >= g_pt[1]) & (f_pts[:, 1] < g_pt[1]+res)
            
            # Place the perfectly flat floor at the DEM's true elevation for this specific terrace
            floor_pts.append([g_pt[0], g_pt[1], true_ground_elevation])
            
            if np.any(col_m):
                min_z = f_pts[col_m, 2].min()
                cell_height = f_pts[col_m, 2].mean() - true_ground_elevation
                vol += cell_a * max(0, cell_height) 
                
                # Foundation Extrusion: Seal the gap up to the physical walls
                if min_z > true_ground_elevation + 0.3:
                    z_steps = np.arange(true_ground_elevation + 0.3, min_z, 0.3)
                    for z in z_steps:
                        floor_pts.append([g_pt[0], g_pt[1], z])
            else:
                vol += cell_a * max(0, f_pts[:, 2].mean() - true_ground_elevation)
                
        floor_pts = np.array(floor_pts)
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
    idx, seeds, local_pts, local_cols, house_gz, rot, off_xy, g_off, img_data, out_dir = args
    
    # Standard 20cm noise filter above the true DEM elevation
    local_mask = local_pts[:, 2] > (house_gz + 0.20)
    
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
        res = apply_house_cookie_cutter(local_pts, local_cols, footprint, house_gz)
        if res:
            p, a, v = res
            # Points are maintained at true global Z. Only shift XY back.
            p.translate((off_xy[0], off_xy[1], 0))
            
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
def process_reconstruction_v5(args):
    global_start_time = time.time()
    project_path = os.path.abspath(args.project_path)
    project_name = os.path.basename(os.path.normpath(project_path))
    out = os.path.join(project_path, f"analysis_{project_name}_v5")
    
    for d in ["individual_houses", "best_images", "best_images_cropped", "diagnostics"]: 
        os.makedirs(os.path.join(out, d), exist_ok=True)

    print("[1/9] Syncing Global Offsets...")
    g_off = auto_detect_offsets(project_path)
    reverse_trans = Transformer.from_crs(EPSG_CODE, "EPSG:4326", always_xy=True)
    img_data = build_spatial_image_index(os.path.join(project_path, 'images'), Transformer.from_crs("EPSG:4326", EPSG_CODE, always_xy=True))

    print("[2/9] Loading and cleaning PLY...")
    pcd = o3d.io.read_point_cloud(repair_openmvs_ply_colors(os.path.join(project_path, 'scene_dense.ply')))
    pcd = pcd.voxel_down_sample(0.05)
    pcd, _ = pcd.remove_statistical_outlier(20, 2.2)

    print("[3/9] Centering XY geometry (Maintaining True Global Z)...")
    pts = np.asarray(pcd.points)
    grnd = pts[pts[:, 2] < np.percentile(pts[:, 2], 30)]
    off_xy = np.mean(grnd, axis=0)[:2]
    pcd.translate((-off_xy[0], -off_xy[1], 0))
    rot = np.eye(3)

    print(f"[4/9] Executing Topography Filter (Mode: {args.ground_mode.upper()})...")
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    
    elevation_model = GroundElevationModel(
        mode=args.ground_mode, 
        pts=pts, 
        dem_path=args.dem_file, 
        reverse_transformer=reverse_trans
    )
    
    # Query true ground elevation for every point in the cloud
    ground_z_array = elevation_model.get_z(pts[:, 0], pts[:, 1])
    
    # Strict AGL cut: keep points 0.35m (~1.1ft) above true ground
    ag_mask = pts[:, 2] > (ground_z_array + 0.35)
    ag_pts = pts[ag_mask]
    ag_colors = cols[ag_mask]

    ag_pcd = o3d.geometry.PointCloud()
    ag_pcd.points = o3d.utility.Vector3dVector(ag_pts)
    ag_pcd.colors = o3d.utility.Vector3dVector(ag_colors)
    ag_pcd, _ = ag_pcd.remove_statistical_outlier(12, 1.2)
    
    ag_pts, ag_colors = np.asarray(ag_pcd.points), np.asarray(ag_pcd.colors)

    print("[5/9] Standard Geometric Consistency...")
    r, g, b = ag_colors[:, 0], ag_colors[:, 1], ag_colors[:, 2]
    exg = (2 * g) - r - b
    cand_idx = np.where(exg > 0.07)[0]
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
    
    # Query ground below surviving points to place seeds perfectly on roofs
    local_ground_z = elevation_model.get_z(final_pts[:, 0], final_pts[:, 1])
    seeds = final_pts[final_pts[:, 2] > (local_ground_z + 1.85)]
    
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
        print("      [!] Error: Erosion removed all structures.")
        return
        
    core_tree = KDTree(core_pts[:, :2])
    dists, nearest_core_idx = core_tree.query(seeds[:, :2], k=1, workers=-1)
    
    labels = core_labels[nearest_core_idx]
    labels[dists > 2.0] = 0 

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    
    label_sizes = {}
    label_coords = {}
    trees = {}
    
    for lbl in unique_labels:
        lbl_pts = seeds[labels == lbl]
        lbl_c = np.floor((lbl_pts[:, :2] - min_b) / res).astype(int)
        area_sqft = len(np.unique(lbl_c, axis=0)) * (res ** 2) * 10.764
        label_sizes[lbl] = area_sqft
        label_coords[lbl] = lbl_pts[:, :2]
        trees[lbl] = KDTree(label_coords[lbl])
        
    parent = {lbl: lbl for lbl in unique_labels}
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]
        
    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            if label_sizes[root_i] < label_sizes[root_j]:
                parent[root_i] = root_j
                label_sizes[root_j] += label_sizes[root_i]
            else:
                parent[root_j] = root_i
                label_sizes[root_i] += label_sizes[root_j]
    
    for lbl_a in unique_labels:
        for lbl_b in unique_labels:
            if lbl_a >= lbl_b: continue
            root_a, root_b = find(lbl_a), find(lbl_b)
            if root_a == root_b: continue
            
            size_a, size_b = label_sizes[root_a], label_sizes[root_b]
            is_a_frag = size_a < 150 or size_a < (0.05 * size_b)
            is_b_frag = size_b < 150 or size_b < (0.05 * size_a)
            
            if not (is_a_frag or is_b_frag): continue
                
            dists_ab, _ = trees[lbl_b].query(label_coords[lbl_a], k=1, distance_upper_bound=0.75)
            if np.any(dists_ab < 0.75):
                union(lbl_a, lbl_b)
                
    for i in range(len(labels)):
        if labels[i] > 0:
            labels[i] = find(labels[i])
            
    final_unique_labels = np.unique(labels)
    final_unique_labels = final_unique_labels[final_unique_labels > 0]

    worker_args = []
    for i in final_unique_labels:
        idx = np.where(labels == i)[0]
        blob = seeds[idx]
        if len(idx) < 150: continue
        
        # Calculate the exact DEM elevation for the centroid of this specific house
        centroid_x, centroid_y = np.mean(blob[:, 0]), np.mean(blob[:, 1])
        house_gz = elevation_model.get_z(np.array([centroid_x]), np.array([centroid_y]))[0]
        
        m, M = blob.min(axis=0), blob.max(axis=0)
        mask = (ag_pts[:, 0] >= m[0]-5.0) & (ag_pts[:, 0] <= M[0]+5.0) & (ag_pts[:, 1] >= m[1]-5.0) & (ag_pts[:, 1] <= M[1]+5.0)
        
        # Pass the exact house_gz to the worker so it doesn't have to guess
        worker_args.append((i, blob, ag_pts[mask], ag_colors[mask], house_gz, rot, off_xy, g_off, img_data, out))

    print(f"[7/9] Parallel Extraction ({len(worker_args)} candidates)...")
    raw_res = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(NUM_CORES, 32)) as ex:
        for res_list in ex.map(worker_extraction, worker_args):
            if res_list: raw_res.extend(res_list)

    print("[8/9] Artifact Purge & Sequential Georeferencing...")
    final_measurements = []
    lookup_table = []
    house_counter = 1
    
    if len(raw_res) > 0:
        valid_res = []
        for r in raw_res:
            # HARD GEOMETRIC BOUNDS: Instantly purge cars, roads, and floating junk
            if r["Height_ft"] < 6.5 or r["Area_sqft"] < 250:
                old_ply_path = os.path.join(out, "individual_houses", f"{r['temp_ID']}.ply")
                old_img_path = os.path.join(out, "best_images", f"{r['temp_ID']}.jpg")
                if os.path.exists(old_ply_path): os.remove(old_ply_path)
                if os.path.exists(old_img_path): os.remove(old_img_path)
                continue
                
            valid_res.append(r)
            
        if len(valid_res) > 0:
            ratios = np.array([r["Ratio_Area_to_Height"] for r in valid_res])
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
                
            for r in valid_res:
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

    print("[9/9] Precision Cropping Images...")
    reconstruction_file = os.path.join(project_path, 'reconstruction.json')
    
    for record in lookup_table:
        house_id = record["house_ID"]
        orig_name = record["Original_Image"]
        if orig_name == "N/A": continue
            
        ply_path = os.path.join(out, "individual_houses", f"{house_id}.ply")
        img_path = os.path.join(out, "best_images", f"{house_id}.jpg")
        crop_out_path = os.path.join(out, "best_images_cropped", f"{house_id}.jpg")
        
        if os.path.exists(ply_path) and os.path.exists(img_path) and os.path.exists(reconstruction_file):
            try:
                rvec, tvec, K, dist = load_opensfm_camera(reconstruction_file, orig_name)
                corners_3d = get_3d_bounding_box(ply_path)
                corners_2d, _ = cv2.projectPoints(corners_3d, rvec, tvec, K, dist)
                corners_2d = corners_2d.reshape(-1, 2)
                
                x_min, x_max = int(np.floor(np.min(corners_2d[:, 0]))), int(np.ceil(np.max(corners_2d[:, 0])))
                y_min, y_max = int(np.floor(np.min(corners_2d[:, 1]))), int(np.ceil(np.max(corners_2d[:, 1])))
                
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    padding = 60
                    x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
                    x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)
                    
                    cv2.imwrite(crop_out_path, img[y_min:y_max, x_min:x_max])
            except Exception as e:
                print(f"      [!] Failed to crop {house_id}: {e}")

    with open(os.path.join(out, "measurements.csv"), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["house_ID", "Area_sqft", "Volume_cuft", "Height_ft", "Ratio_Area_to_Height", "Best_Image"])
        writer.writeheader()
        writer.writerows(final_measurements)
        
    with open(os.path.join(out, "location_lookup.csv"), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["house_ID", "X_UTM", "Y_UTM", "Latitude", "Longitude", "Original_Image"])
        writer.writeheader()
        writer.writerows(lookup_table)
        
    print(f"SUCCESS. Validated and cropped {len(final_measurements)} buildings in {time.time()-global_start_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract buildings from point cloud data.")
    parser.add_argument("project_path", type=str, help="Path to the OpenDroneMap project directory.")
    parser.add_argument("--ground-mode", choices=['ransac', 'usgs', 'local'], default='ransac', 
                        help="Select the topography engine. Default is global ransac.")
    parser.add_argument("--dem-file", type=str, default=None, 
                        help="Path to the local .tif DEM file (Required if --ground-mode is 'local')")
    
    args = parser.parse_args()
    process_reconstruction_v5(args)