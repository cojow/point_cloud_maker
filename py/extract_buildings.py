import os

# --- PREVENT THREAD COLLISION ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from sklearn.cluster import DBSCAN
import numpy as np
import open3d as o3d
import csv
import sys
import time
import shutil
import glob
import gc
from scipy.spatial import Delaunay, KDTree
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS
from pyproj import Transformer
import concurrent.futures

# --- CONVENTIONS & SETTINGS ---
EPSG_CODE = "EPSG:32612"  
GLOBAL_PLY_OFFSET_X = 0 
GLOBAL_PLY_OFFSET_Y = 0

def get_exif_gps(image_path):
    try:
        image = Image.open(image_path)
        exif = image._getexif()
        if not exif: return None
        gps_info = {}
        for key, val in exif.items():
            tag = TAGS.get(key)
            if tag == 'GPSInfo':
                for t, v in val.items(): gps_info[GPSTAGS.get(t, t)] = v
        if 'GPSLatitude' not in gps_info or 'GPSLongitude' not in gps_info: return None
        def convert_to_degrees(value):
            return float(value[0]) + (float(value[1]) / 60.0) + (float(value[2]) / 3600.0)
        lat, lon = convert_to_degrees(gps_info['GPSLatitude']), convert_to_degrees(gps_info['GPSLongitude'])
        if gps_info['GPSLatitudeRef'] != 'N': lat = -lat
        if gps_info['GPSLongitudeRef'] != 'E': lon = -lon
        return lat, lon
    except Exception:
        return None

def build_image_catalog(image_dir, transformer):
    catalog = []
    if not image_dir or not os.path.exists(image_dir): return catalog
    image_files = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.JPG"))
    for img_path in image_files:
        coords = get_exif_gps(img_path)
        if coords:
            x_proj, y_proj = transformer.transform(coords[0], coords[1])
            catalog.append({"path": img_path, "x": x_proj, "y": y_proj})
    return catalog

def calculate_alpha_shape(points_2d, alpha):
    if len(points_2d) < 4:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate([points_2d, np.zeros((len(points_2d), 1))], axis=1)))
        return pcd.compute_convex_hull()[0]
    tri = Delaunay(points_2d)
    edges = np.array([points_2d[tri.simplices[:, [0, 1]]], points_2d[tri.simplices[:, [1, 2]]], points_2d[tri.simplices[:, [2, 0]]]])
    lengths = np.sqrt(np.sum((edges[:, :, 0, :] - edges[:, :, 1, :])**2, axis=2))
    with np.errstate(divide='ignore', invalid='ignore'):
        circum_r = lengths[0, :] * lengths[1, :] * lengths[2, :] / (np.sqrt((lengths[0, :] + lengths[1, :] + lengths[2, :]) * (-lengths[0, :] + lengths[1, :] + lengths[2, :]) * (lengths[0, :] - lengths[1, :] + lengths[2, :]) * (lengths[0, :] + lengths[1, :] - lengths[2, :])))
    final_simplices = tri.simplices[np.nan_to_num(circum_r) < alpha]
    if len(final_simplices) == 0: return None
    hull_pts_3d = np.concatenate([points_2d, np.zeros((len(points_2d), 1))], axis=1)
    unique_indices = np.unique(final_simplices)
    index_map = {old: new for new, old in enumerate(unique_indices)}
    mapped_simplices = np.vectorize(index_map.get)(final_simplices)
    final_hull_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(hull_pts_3d[unique_indices]))
    hull_mesh = o3d.geometry.TriangleMesh(final_hull_pcd.points, o3d.utility.Vector3iVector(mapped_simplices))
    return o3d.geometry.VoxelGrid.create_from_triangle_mesh(hull_mesh, voxel_size=0.3)

def apply_house_cookie_cutter(ag_pts, concave_voxel_grid, mins, maxs):
    box_mask = (ag_pts[:, 0] >= mins[0]) & (ag_pts[:, 0] <= maxs[0]) & (ag_pts[:, 1] >= mins[1]) & (ag_pts[:, 1] <= maxs[1])
    box_pts = ag_pts[box_mask]
    if len(box_pts) == 0: return None
    contain_mask = concave_voxel_grid.check_if_included(o3d.utility.Vector3dVector(box_pts * [1., 1., 0.]))
    final_house_pts = box_pts[contain_mask]
    if len(final_house_pts) == 0: return None
    final_house_pts = final_house_pts[final_house_pts[:, 2] >= 0.15]
    if len(final_house_pts) == 0: return None
    squished_pts = np.copy(final_house_pts)
    squished_pts[:, 2] *= 0.1  
    labels = np.array(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(squished_pts)).cluster_dbscan(eps=0.75, min_points=15))
    if labels.max() >= 0:
        valid_mask = np.zeros(len(final_house_pts), dtype=bool)
        for label_id in range(labels.max() + 1):
            cluster_mask = (labels == label_id)
            if np.sum(cluster_mask) > 300: 
                cluster_pts = final_house_pts[cluster_mask]
                if len(cluster_pts) > 0 and (cluster_pts[:, 2].max() - cluster_pts[:, 2].min()) > 1.5: 
                    valid_mask = valid_mask | cluster_mask
        final_house_pts = final_house_pts[valid_mask]
    else: return None
    if len(final_house_pts) == 0: return None
    final_footprint = calculate_alpha_shape(final_house_pts[:, :2], alpha=0.6)
    if final_footprint is None: return None
    new_mins, new_maxs = final_footprint.get_min_bound(), final_footprint.get_max_bound()
    max_h, min_h = final_house_pts[:, 2].max(), final_house_pts[:, 2].min()
    if max_h > 2.5 and (max_h - min_h) > 1.5:
        grid_size, cell_area = 0.6, 0.36
        total_volume = 0.0
        grid_points = [[x, y, 0.0] for x in np.arange(new_mins[0], new_maxs[0], grid_size) for y in np.arange(new_mins[1], new_maxs[1], grid_size)]
        grid_pts_array = np.array(grid_points)
        floor_pcd = o3d.geometry.PointCloud()
        exact_area_m2 = 0.0
        if len(grid_pts_array) > 0:
            floor_mask = final_footprint.check_if_included(o3d.utility.Vector3dVector(grid_pts_array))
            valid_floor = grid_pts_array[floor_mask]
            if len(valid_floor) > 0:
                exact_area_m2 = len(valid_floor) * cell_area
                floor_pcd.points = o3d.utility.Vector3dVector(valid_floor)
                floor_pcd.paint_uniform_color([0.5, 0.5, 0.5])
                global_avg_h = final_house_pts[:, 2].mean()
                for floor_pt in valid_floor:
                    x, y, _ = floor_pt
                    col_mask = (final_house_pts[:, 0] >= x) & (final_house_pts[:, 0] < x + grid_size) & (final_house_pts[:, 1] >= y) & (final_house_pts[:, 1] < y + grid_size)
                    local_roof_pts = final_house_pts[col_mask]
                    total_volume += (cell_area * local_roof_pts[:, 2].mean()) if len(local_roof_pts) > 0 else (cell_area * global_avg_h)
        house_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(final_house_pts))
        return house_pcd + floor_pcd, exact_area_m2, total_volume
    return None

def worker_process_cluster(args):
    """Worker function with strictly bounded timestamps."""
    # T1: Worker boots up and receives payload
    t_enter = time.time()
    
    cluster_idx, current_blob_roof_pts, local_ag_pts, original_offset_xy, ground_z, rotation_matrix, image_catalog, debug_dir, images_out_dir = args
    houses_found = []
    pts_count = len(current_blob_roof_pts)
    pid = os.getpid()
    
    points_2d = current_blob_roof_pts[:, :2]
    if len(points_2d) < 3: 
        t_exit = time.time()
        return houses_found, f"[PID:{pid}] Cluster {cluster_idx} ({pts_count} pts) -> Skipped (<3 pts)", t_enter, t_exit
    
    voxel_footprint = calculate_alpha_shape(points_2d, alpha=0.6)
    if voxel_footprint is None: 
        t_exit = time.time()
        return houses_found, f"[PID:{pid}] Cluster {cluster_idx} ({pts_count} pts) -> Skipped (No footprint)", t_enter, t_exit
    
    mins, maxs = voxel_footprint.get_min_bound(), voxel_footprint.get_max_bound()
    area = (maxs[0] - mins[0]) * (maxs[1] - mins[1]) 

    if area <= 35:
        t_exit = time.time()
        return houses_found, f"[PID:{pid}] Cluster {cluster_idx} ({pts_count} pts) -> Skipped (Area {area:.1f} < 35)", t_enter, t_exit

    sub_clusters = [current_blob_roof_pts]
    if area > 350:
        sub_squished_pts = np.copy(current_blob_roof_pts)
        sub_squished_pts[:, 2] *= 0.1 
        sub_labels = np.array(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sub_squished_pts)).cluster_dbscan(eps=1.2, min_points=8))
        if sub_labels.max() >= 0:
            sub_clusters = [current_blob_roof_pts[np.where(sub_labels == k)[0]] for k in range(sub_labels.max() + 1)]

    for sc_pts in sub_clusters:
        valid_roof_pts = []
        remaining_pts = sc_pts.copy()
        
        while len(remaining_pts) > 500:
            pcd_temp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(remaining_pts))
            plane_model, inliers = pcd_temp.segment_plane(distance_threshold=0.15, ransac_n=3, num_iterations=250)
            if len(inliers) < 500: break 
            plane_pts = remaining_pts[inliers]
            plane_labels = np.array(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(plane_pts)).cluster_dbscan(eps=0.5, min_points=15))
            if plane_labels.max() >= 0:
                largest_cluster_idx = np.bincount(plane_labels[plane_labels >= 0]).argmax()
                contiguous_inliers = np.where(plane_labels == largest_cluster_idx)[0]
                if len(contiguous_inliers) > 400:
                    normal = np.array(plane_model[:3])
                    normal = normal / np.linalg.norm(normal)
                    if abs(normal[2]) > 0.5: valid_roof_pts.append(plane_pts[contiguous_inliers])
            mask = np.ones(len(remaining_pts), dtype=bool)
            mask[inliers] = False
            remaining_pts = remaining_pts[mask]
            
        if len(valid_roof_pts) == 0: continue 
        pure_roof_pts = np.vstack(valid_roof_pts)
        sc_2d = pure_roof_pts[:, :2]
        if len(sc_2d) < 3: continue
        sc_footprint = calculate_alpha_shape(sc_2d, alpha=0.6)
        if sc_footprint is None: continue
        
        mins, maxs = sc_footprint.get_min_bound(), sc_footprint.get_max_bound()
        sc_area = (maxs[0] - mins[0]) * (maxs[1] - mins[1])
        
        if 20 < sc_area < 2500:
            result = apply_house_cookie_cutter(local_ag_pts, sc_footprint, mins, maxs)
            if result:
                final_pcd_with_floor, area_m2, vol_m3 = result
                if area_m2 >= 20: 
                    final_pcd_with_floor.translate((0, 0, ground_z))
                    final_pcd_with_floor.translate((original_offset_xy[0], original_offset_xy[1], 0))
                    final_pcd_with_floor.rotate(rotation_matrix.T, center=(0, 0, 0))
                    centroid = np.mean(np.asarray(final_pcd_with_floor.points), axis=0)
                    global_x, global_y = centroid[0] + GLOBAL_PLY_OFFSET_X, centroid[1] + GLOBAL_PLY_OFFSET_Y
                    unique_id = f"H_{abs(global_x):.5f}_{abs(global_y):.5f}".replace('.', 'd')
                    best_image = None
                    if image_catalog:
                        distances = [np.sqrt((global_x - img['x'])**2 + (global_y - img['y'])**2) for img in image_catalog]
                        best_image = image_catalog[np.argmin(distances)]['path']
                        shutil.copy2(best_image, os.path.join(images_out_dir, f"{unique_id}.jpg"))
                    o3d.io.write_point_cloud(os.path.join(debug_dir, f"{unique_id}.ply"), final_pcd_with_floor)
                    houses_found.append({"house_ID": unique_id, "Area_sqft": round(area_m2 * 10.76391, 2), "Volume_cuft": round(vol_m3 * 35.31467, 2), "Best_Image": best_image if best_image else "N/A"})
    
    # T2: Math is fully complete
    t_exit = time.time()
    profile_str = f"[PID:{pid}] Cluster {cluster_idx:>2} | Pts: {pts_count:>6} | Found: {len(houses_found)} | Math: {t_exit - t_enter:>5.2f}s"
    
    return houses_found, profile_str, t_enter, t_exit

def process_reconstruction_v22(ply_path, raw_images_dir=None):
    global_start_time = time.time()
    
    output_dir = os.path.join(os.path.dirname(ply_path), "house_analysis_v22_7")
    debug_dir, images_out_dir = os.path.join(output_dir, "individual_houses"), os.path.join(output_dir, "best_images")
    for d in [output_dir, debug_dir, images_out_dir]: os.makedirs(d, exist_ok=True)

    transformer = Transformer.from_crs("EPSG:4326", EPSG_CODE, always_xy=True)
    
    print("[1/7] Building Image GPS Catalog...")
    t0 = time.time()
    image_catalog = build_image_catalog(raw_images_dir, transformer)
    print(f"      Mapped {len(image_catalog)} images in {time.time() - t0:.2f}s")
        
    print(f"\n[2/7] Loading point cloud from {ply_path}...")
    t0 = time.time()
    pcd = o3d.io.read_point_cloud(ply_path)
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    print(f"      Done in {time.time() - t0:.2f}s")

    print("[3/7] Cleaning and leveling...")
    t0 = time.time()
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.2)
    pts_clean = np.asarray(pcd.points)

    ground_pts = pts_clean[pts_clean[:, 2] < np.percentile(pts_clean[:, 2], 30)]
    centroid = np.mean(ground_pts, axis=0)
    evals, evecs = np.linalg.eigh(np.cov(ground_pts.T))
    plane_norm = evecs[:, 0] if evecs[2, 0] > 0 else -evecs[:, 0] 

    v = np.cross(plane_norm, np.array([0, 0, 1]))
    s, c_val = np.linalg.norm(v), np.dot(plane_norm, np.array([0, 0, 1]))
    if s > 1e-6:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c_val) / (s**2))
        pcd.rotate(rotation_matrix, center=(0,0,0))
    else:
        rotation_matrix = np.eye(3)
    original_offset_xy = centroid[:2]
    pcd.translate((-original_offset_xy[0], -original_offset_xy[1], 0))
    print(f"      Done in {time.time() - t0:.2f}s")

    print("[4/7] Applying RANSAC (Ground Removal)...")
    t0 = time.time()
    _, inliers = pcd.segment_plane(distance_threshold=0.5, ransac_n=3, num_iterations=1000)
    ground_z = np.median(np.asarray(pcd.select_by_index(inliers).points)[:, 2])
    above_ground_pcd = pcd.select_by_index(inliers, invert=True)
    above_ground_pcd.translate((0, 0, -ground_z))
    ag_pts = np.asarray(above_ground_pcd.points)
    print(f"      Done in {time.time() - t0:.2f}s")

    print("[5/7] Isolating potential roofs...")
    t0 = time.time()
    high_idx = np.where(ag_pts[:, 2] > 2.2)[0]
    high_pts = ag_pts[high_idx]
    if len(high_pts) < 10: return print("      Insufficient high points. Exiting.")
    _, neighbor_indices = KDTree(high_pts).query(high_pts, k=10) 
    planar_mask = np.std(high_pts[neighbor_indices][:, :, 2], axis=1) < 0.20
    roof_pcd = above_ground_pcd.select_by_index(high_idx[planar_mask])
    real_roof_pts = np.asarray(roof_pcd.points)
    print(f"      Done in {time.time() - t0:.2f}s")
    
    print("[6/7] Broad clustering & Parallel Extraction (TIMELINE TRACER)...")
    step_start = time.time()
    roof_pts_squished = np.copy(real_roof_pts)
    roof_pts_squished[:, 2] *= 0.1 
    
    # Establish max workers early so sklearn can use them
    slurm_cores = os.environ.get('SLURM_CPUS_PER_TASK')
    max_workers = int(slurm_cores) if slurm_cores and slurm_cores.isdigit() else min(6, os.cpu_count() or 1)
    
    print(f"      -> Running global DBScan using {max_workers} threads...")
    t_db_start = time.time()
    
    # Use scikit-learn's DBSCAN. By passing n_jobs=max_workers, it bypasses the 
    # OpenMP 1-thread limit and utilizes all your SLURM cores for this massive step.
    # Note: sklearn uses 'min_samples' instead of Open3D's 'min_points'
    db = DBSCAN(eps=2.5, min_samples=15, n_jobs=max_workers).fit(roof_pts_squished)
    labels = db.labels_
    
    print(f"      -> Global DBScan finished in {time.time() - t_db_start:.2f}s")
    
    if labels.size == 0 or labels.max() < 0: return print("      No clusters found. Exiting.")

    cluster_args = []
    for i in range(labels.max() + 1):
        current_blob = real_roof_pts[np.where(labels == i)[0]]
        mins, maxs = current_blob.min(axis=0) - 5.0, current_blob.max(axis=0) + 5.0
        local_mask = (ag_pts[:, 0] >= mins[0]) & (ag_pts[:, 0] <= maxs[0]) & (ag_pts[:, 1] >= mins[1]) & (ag_pts[:, 1] <= maxs[1])
        cluster_args.append((i, current_blob, ag_pts[local_mask], original_offset_xy, ground_z, rotation_matrix, image_catalog, debug_dir, images_out_dir))

    house_list = []
    slurm_cores = os.environ.get('SLURM_CPUS_PER_TASK')
    max_workers = int(slurm_cores) if slurm_cores and slurm_cores.isdigit() else min(6, os.cpu_count() or 1)
    
    print(f"\n      -> Initiating ProcessPoolExecutor with {max_workers} workers.")
    print(f"      -> Total Clusters to evaluate: {len(cluster_args)}\n")

    # --- THE MICROSCOPIC TRACER ---
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        t_submit_start = time.time()
        
        # T0: Submitting to pipe
        future_to_idx = {}
        for arg in cluster_args:
            future = executor.submit(worker_process_cluster, arg)
            future_to_idx[future] = arg[0]
            
        t_submit_end = time.time()
        print(f"      [SYSTEM] Pipeline serialization (pickling) took: {t_submit_end - t_submit_start:.2f}s")

        last_received_time = t_submit_end

        for future in concurrent.futures.as_completed(future_to_idx):
            # T3: Main thread receives the package
            t_received = time.time()
            cluster_idx = future_to_idx[future]
            try:
                houses, profile_str, t_enter, t_exit = future.result()
                if houses: house_list.extend(houses)
                
                dispatch_lag = t_enter - t_submit_end
                math_duration = t_exit - t_enter
                return_lag = t_received - t_exit
                
                print(f"      {profile_str}")
                print(f"         > Dispatch Lag: {dispatch_lag:>5.2f}s | Math: {math_duration:>5.2f}s | Return Lag: {return_lag:>5.2f}s")
                last_received_time = t_received
            except Exception as e:
                print(f"\n      [WARNING] Cluster {cluster_idx} failed with error: {e}")

    # T4: Pool officially closed
    t_pool_closed = time.time()
    teardown_time = t_pool_closed - last_received_time
    print(f"\n      [SYSTEM] Pool Teardown (Garbage Collection) took: {teardown_time:.2f}s")
    print(f"      Done in {time.time() - step_start:.2f}s")

    print("\n[7/7] Finalizing CSV Output...")
    csv_path = os.path.join(output_dir, "house_measurements.csv")
    if house_list:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["house_ID", "Area_sqft", "Volume_cuft", "Best_Image"])
            writer.writeheader()
            writer.writerows(house_list)
            
    print("\n" + "="*40)
    print("--- Extraction Complete ---")
    print(f"Total Processing Time:   {time.time() - global_start_time:.2f} seconds")
    print(f"Unique Buildings Found:  {len(house_list)}")
    print("="*40 + "\n")

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        sys.exit(1)
        
    project_dir = os.path.abspath(sys.argv[1])
    test_file, raw_images_directory = os.path.join(project_dir, 'scene_dense.ply'), os.path.join(project_dir, 'images')
    if not os.path.exists(test_file): sys.exit(1)
        
    process_reconstruction_v22(test_file, raw_images_directory)
