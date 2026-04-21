import numpy as np
import open3d as o3d
import os
import csv
import sys
import time
import shutil
import glob
from scipy.spatial import Delaunay, KDTree
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS
from pyproj import Transformer
from tqdm import tqdm
import concurrent.futures
from sklearn.cluster import DBSCAN  # --- THE MEMORY FIX ---

# --- CONVENTIONS & SETTINGS ---
EPSG_CODE = "EPSG:32612"  

# If your PLY file has a 'Global Shift' (e.g. from CloudCompare or Pix4D), 
# enter those large numbers here. If not, leave as 0.
GLOBAL_PLY_OFFSET_X = 0 
GLOBAL_PLY_OFFSET_Y = 0
# ------------------------------

def get_exif_gps(image_path):
    try:
        image = Image.open(image_path)
        exif = image._getexif()
        if not exif: 
            return None

        gps_info = {}
        for key, val in exif.items():
            tag = TAGS.get(key)
            if tag == 'GPSInfo':
                for t, v in val.items():
                    gps_tag = GPSTAGS.get(t, t)
                    gps_info[gps_tag] = v

        if 'GPSLatitude' not in gps_info or 'GPSLongitude' not in gps_info:
            return None

        def convert_to_degrees(value):
            d = float(value[0])
            m = float(value[1])
            s = float(value[2])
            return d + (m / 60.0) + (s / 3600.0)

        lat = convert_to_degrees(gps_info['GPSLatitude'])
        lon = convert_to_degrees(gps_info['GPSLongitude'])

        if gps_info['GPSLatitudeRef'] != 'N': 
            lat = -lat
        if gps_info['GPSLongitudeRef'] != 'E': 
            lon = -lon
        
        return lat, lon
    except Exception:
        return None

def build_image_catalog(image_dir, transformer):
    catalog = []
    if not image_dir or not os.path.exists(image_dir):
        return catalog
        
    image_files = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.JPG"))
    
    for img_path in image_files:
        coords = get_exif_gps(img_path)
        if coords:
            lat, lon = coords
            x_proj, y_proj = transformer.transform(lat, lon)
            catalog.append({
                "path": img_path,
                "x": x_proj,
                "y": y_proj
            })
    return catalog

def calculate_alpha_shape(points_2d, alpha):
    if len(points_2d) < 4:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate([points_2d, np.zeros((len(points_2d), 1))], axis=1)))
        hull, _ = pcd.compute_convex_hull()
        return hull
        
    tri = Delaunay(points_2d)
    
    edges = np.array([
        points_2d[tri.simplices[:, [0, 1]]],
        points_2d[tri.simplices[:, [1, 2]]],
        points_2d[tri.simplices[:, [2, 0]]]
    ])
    
    lengths = np.sqrt(np.sum((edges[:, :, 0, :] - edges[:, :, 1, :])**2, axis=2))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        circum_r = lengths[0, :] * lengths[1, :] * lengths[2, :] / (
            np.sqrt((lengths[0, :] + lengths[1, :] + lengths[2, :]) *
                    (-lengths[0, :] + lengths[1, :] + lengths[2, :]) *
                    (lengths[0, :] - lengths[1, :] + lengths[2, :]) *
                    (lengths[0, :] + lengths[1, :] - lengths[2, :]))
        )
    
    final_simplices = tri.simplices[np.nan_to_num(circum_r) < alpha]
    if len(final_simplices) == 0: 
        return None
    
    hull_pts_3d = np.concatenate([points_2d, np.zeros((len(points_2d), 1))], axis=1)
    unique_indices = np.unique(final_simplices)
    index_map = {old: new for new, old in enumerate(unique_indices)}
    mapped_simplices = np.vectorize(index_map.get)(final_simplices)
    
    final_hull_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(hull_pts_3d[unique_indices]))
    hull_mesh = o3d.geometry.TriangleMesh(final_hull_pcd.points, o3d.utility.Vector3iVector(mapped_simplices))
    
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(hull_mesh, voxel_size=0.3)
    return voxel_grid

def apply_house_cookie_cutter(ag_pts, concave_voxel_grid, mins, maxs):
    box_mask = (ag_pts[:, 0] >= mins[0]) & (ag_pts[:, 0] <= maxs[0]) & \
               (ag_pts[:, 1] >= mins[1]) & (ag_pts[:, 1] <= maxs[1])
    box_pts = ag_pts[box_mask]
    
    if len(box_pts) == 0: 
        return None
    
    contain_mask = concave_voxel_grid.check_if_included(o3d.utility.Vector3dVector(box_pts * [1., 1., 0.]))
    final_house_pts = box_pts[contain_mask]
    if len(final_house_pts) == 0: 
        return None
    
    final_house_pts = final_house_pts[final_house_pts[:, 2] >= 0.15]
    if len(final_house_pts) == 0: 
        return None

    squished_pts = np.copy(final_house_pts)
    squished_pts[:, 2] *= 0.1  
    
    # Sklearn DBSCAN avoids Open3D memory leak
    labels = DBSCAN(eps=0.75, min_samples=15, n_jobs=1).fit(squished_pts).labels_
    
    if labels.max() >= 0:
        valid_mask = np.zeros(len(final_house_pts), dtype=bool)
        
        for label_id in range(labels.max() + 1):
            cluster_mask = (labels == label_id)
            
            if np.sum(cluster_mask) > 300: 
                cluster_pts = final_house_pts[cluster_mask]
                if len(cluster_pts) > 0:
                    cluster_height = cluster_pts[:, 2].max() - cluster_pts[:, 2].min()
                    if cluster_height > 1.5: 
                        valid_mask = valid_mask | cluster_mask
                        
        final_house_pts = final_house_pts[valid_mask]
    else:
        return None

    if len(final_house_pts) == 0: 
        return None

    final_points_2d = final_house_pts[:, :2]
    final_footprint = calculate_alpha_shape(final_points_2d, alpha=0.6)
    
    if final_footprint is None: 
        return None
    
    new_mins = final_footprint.get_min_bound()
    new_maxs = final_footprint.get_max_bound()
    
    max_h = final_house_pts[:, 2].max()
    min_h = final_house_pts[:, 2].min()
    
    if max_h > 2.5 and (max_h - min_h) > 1.5:
        grid_size = 0.6
        cell_area = grid_size * grid_size  
        total_volume = 0.0
        
        grid_points = [[x, y, 0.0] for x in np.arange(new_mins[0], new_maxs[0], grid_size) 
                                   for y in np.arange(new_mins[1], new_maxs[1], grid_size)]
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
                    
                    col_mask = (final_house_pts[:, 0] >= x) & (final_house_pts[:, 0] < x + grid_size) & \
                               (final_house_pts[:, 1] >= y) & (final_house_pts[:, 1] < y + grid_size)
                    
                    local_roof_pts = final_house_pts[col_mask]
                    
                    if len(local_roof_pts) > 0:
                        local_h = local_roof_pts[:, 2].mean()
                        total_volume += (cell_area * local_h)
                    else:
                        total_volume += (cell_area * global_avg_h)

        house_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(final_house_pts))
        
        return house_pcd + floor_pcd, exact_area_m2, total_volume
        
    return None

def worker_process_cluster(args):
    current_blob_roof_pts, local_ag_pts, original_offset_xy, ground_z, rotation_matrix, image_catalog, debug_dir, images_out_dir = args
    houses_found = []
    
    points_2d = current_blob_roof_pts[:, :2]
    if len(points_2d) < 3: 
        return houses_found
    
    voxel_footprint = calculate_alpha_shape(points_2d, alpha=0.6)
    if voxel_footprint is None: 
        return houses_found
    
    mins, maxs = voxel_footprint.get_min_bound(), voxel_footprint.get_max_bound()
    area = (maxs[0] - mins[0]) * (maxs[1] - mins[1]) 

    if area > 35:
        if area > 350:
            sub_squished_pts = np.copy(current_blob_roof_pts)
            sub_squished_pts[:, 2] *= 0.1 
            # Sklearn DBSCAN substitution
            sub_labels = DBSCAN(eps=1.2, min_samples=8, n_jobs=1).fit(sub_squished_pts).labels_
                        
            sub_clusters = []
            if sub_labels.max() >= 0:
                for k in range(sub_labels.max() + 1):
                    sub_idx = np.where(sub_labels == k)[0]
                    sub_clusters.append(current_blob_roof_pts[sub_idx])
            else:
                sub_clusters = [current_blob_roof_pts]
        else:
            sub_clusters = [current_blob_roof_pts]

        for sc_pts in sub_clusters:
            valid_roof_pts = []
            remaining_pts = sc_pts.copy()
            
            while len(remaining_pts) > 500:
                pcd_temp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(remaining_pts))
                plane_model, inliers = pcd_temp.segment_plane(distance_threshold=0.15, ransac_n=3, num_iterations=250)
                
                if len(inliers) < 500: 
                    break 
                    
                plane_pts = remaining_pts[inliers]
                # Sklearn DBSCAN substitution
                plane_labels = DBSCAN(eps=0.5, min_samples=15, n_jobs=1).fit(plane_pts).labels_
                
                if plane_labels.max() >= 0:
                    largest_cluster_idx = np.bincount(plane_labels[plane_labels >= 0]).argmax()
                    contiguous_inliers = np.where(plane_labels == largest_cluster_idx)[0]
                    
                    if len(contiguous_inliers) > 400:
                        normal = np.array(plane_model[:3])
                        normal = normal / np.linalg.norm(normal)
                        if abs(normal[2]) > 0.5: 
                            valid_roof_pts.append(plane_pts[contiguous_inliers])
                
                mask = np.ones(len(remaining_pts), dtype=bool)
                mask[inliers] = False
                remaining_pts = remaining_pts[mask]
                
            if len(valid_roof_pts) == 0:
                continue 
                
            pure_roof_pts = np.vstack(valid_roof_pts)
            sc_2d = pure_roof_pts[:, :2]
            if len(sc_2d) < 3: 
                continue
            
            sc_footprint = calculate_alpha_shape(sc_2d, alpha=0.6)
            if sc_footprint is None: 
                continue
            
            mins, maxs = sc_footprint.get_min_bound(), sc_footprint.get_max_bound()
            
            result = apply_house_cookie_cutter(local_ag_pts, sc_footprint, mins, maxs)
            
            if result:
                final_pcd_with_floor, area_m2, vol_m3 = result
                
                if area_m2 < 20: 
                    continue
                
                final_pcd_with_floor.translate((0, 0, ground_z))
                final_pcd_with_floor.translate((original_offset_xy[0], original_offset_xy[1], 0))
                final_pcd_with_floor.rotate(rotation_matrix.T, center=(0, 0, 0))
                
                centroid = np.mean(np.asarray(final_pcd_with_floor.points), axis=0)
                global_x = centroid[0] + GLOBAL_PLY_OFFSET_X
                global_y = centroid[1] + GLOBAL_PLY_OFFSET_Y
                
                unique_id = f"H_{abs(global_x):.5f}_{abs(global_y):.5f}".replace('.', 'd')
                
                best_image = None
                if image_catalog:
                    distances = [np.sqrt((global_x - img['x'])**2 + (global_y - img['y'])**2) for img in image_catalog]
                    min_idx = np.argmin(distances)
                    best_image = image_catalog[min_idx]['path']
                    
                    dest_img_path = os.path.join(images_out_dir, f"{unique_id}.jpg")
                    shutil.copy2(best_image, dest_img_path)

                o3d.io.write_point_cloud(os.path.join(debug_dir, f"{unique_id}.ply"), final_pcd_with_floor)
                
                area_sqft = area_m2 * 10.76391
                vol_cuft = vol_m3 * 35.31467
                
                houses_found.append({
                    "house_ID": unique_id, 
                    "Area_sqft": round(area_sqft, 2), 
                    "Volume_cuft": round(vol_cuft, 2),
                    "Best_Image": best_image if best_image else "N/A"
                })
                
    return houses_found

def process_reconstruction_v22(ply_path, raw_images_dir=None):
    global_start_time = time.time()
    
    output_dir = os.path.join(os.path.dirname(ply_path), "house_analysis_v22_7")
    debug_dir = os.path.join(output_dir, "individual_houses")
    images_out_dir = os.path.join(output_dir, "best_images")
    
    for d in [output_dir, debug_dir, images_out_dir]:
        if not os.path.exists(d): 
            os.makedirs(d)

    transformer = Transformer.from_crs("EPSG:4326", EPSG_CODE, always_xy=True)
    
    print("[1/7] Building Image GPS Catalog...")
    step_start = time.time()
    image_catalog = build_image_catalog(raw_images_dir, transformer)
    print(f"      Mapped {len(image_catalog)} images in {time.time() - step_start:.2f}s")
        
    print(f"\n[2/7] Loading point cloud from {ply_path}...")
    step_start = time.time()
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"      Done in {time.time() - step_start:.2f}s")

    print("[3/7] Cleaning and leveling...")
    step_start = time.time()
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.2)
    pts_clean = np.asarray(pcd.points)

    z_vals = pts_clean[:, 2]
    ground_thresh = np.percentile(z_vals, 30)
    ground_pts = pts_clean[z_vals < ground_thresh]

    centroid = np.mean(ground_pts, axis=0)
    cov = np.cov(ground_pts.T)
    evals, evecs = np.linalg.eigh(cov)
    plane_norm = evecs[:, 0] 
    if plane_norm[2] < 0: 
        plane_norm = -plane_norm

    target_norm = np.array([0, 0, 1])
    v = np.cross(plane_norm, target_norm)
    s = np.linalg.norm(v)
    c_val = np.dot(plane_norm, target_norm)
    
    if s > 1e-6:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c_val) / (s**2))
        pcd.rotate(rotation_matrix, center=(0,0,0))
    else:
        rotation_matrix = np.eye(3)
        
    original_offset_xy = centroid[:2]
    pcd.translate((-original_offset_xy[0], -original_offset_xy[1], 0))
    print(f"      Done in {time.time() - step_start:.2f}s")

    print("[4/7] Applying RANSAC (Ground Removal)...")
    step_start = time.time()
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.5, ransac_n=3, num_iterations=1000)
    ground_pcd = pcd.select_by_index(inliers)
    ground_z = np.median(np.asarray(ground_pcd.points)[:, 2])
    
    above_ground_pcd = pcd.select_by_index(inliers, invert=True)
    above_ground_pcd.translate((0, 0, -ground_z))
    ag_pts = np.asarray(above_ground_pcd.points)
    print(f"      Done in {time.time() - step_start:.2f}s")

    print("[5/7] Isolating potential roofs...")
    step_start = time.time()
    high_idx = np.where(ag_pts[:, 2] > 2.2)[0]
    high_pts = ag_pts[high_idx]
    
    if len(high_pts) < 10: 
        print("      Insufficient high points found. Exiting.")
        return

    tree = KDTree(high_pts)
    _, neighbor_indices = tree.query(high_pts, k=10, workers=1) 
    z_vals_high = high_pts[:, 2]
    neighbor_z = z_vals_high[neighbor_indices]
    neighborhood_stdev = np.std(neighbor_z, axis=1)
    
    planar_mask = neighborhood_stdev < 0.20
    roof_idx = high_idx[planar_mask]
    roof_pcd = above_ground_pcd.select_by_index(roof_idx)
    real_roof_pts = np.asarray(roof_pcd.points)
    print(f"      Done in {time.time() - step_start:.2f}s")
    
    print("[6/7] Broad clustering & Parallel Extraction...")
    step_start = time.time()
    roof_pts_squished = np.copy(real_roof_pts)
    roof_pts_squished[:, 2] *= 0.1 
    
    # Sklearn DBSCAN substitution
    labels = DBSCAN(eps=2.5, min_samples=15, n_jobs=1).fit(roof_pts_squished).labels_
    
    if labels.size == 0 or labels.max() < 0: 
        print("      No clusters found. Exiting.")
        return

    # Windows Grouping Optimization
    valid_mask = labels >= 0
    valid_labels = labels[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    sort_order = np.argsort(valid_labels)
    sorted_labels = valid_labels[sort_order]
    sorted_indices = valid_indices[sort_order]

    split_points = np.where(np.diff(sorted_labels) != 0)[0] + 1
    grouped_indices = np.split(sorted_indices, split_points)

    cluster_args = []
    for broad_idx in grouped_indices:
        current_blob_roof_pts = real_roof_pts[broad_idx]
        
        mins = current_blob_roof_pts.min(axis=0) - 5.0
        maxs = current_blob_roof_pts.max(axis=0) + 5.0
        
        local_mask = (ag_pts[:, 0] >= mins[0]) & (ag_pts[:, 0] <= maxs[0]) & \
                     (ag_pts[:, 1] >= mins[1]) & (ag_pts[:, 1] <= maxs[1])
                     
        local_ag_pts = ag_pts[local_mask]
        
        cluster_args.append((
            current_blob_roof_pts,
            local_ag_pts,
            original_offset_xy,
            ground_z,
            rotation_matrix,
            image_catalog,
            debug_dir,
            images_out_dir
        ))

    # --- MULTIPROCESSING EXECUTION ---
    house_list = []
    
    # Restrict to a safe number of cores (max 3) to prevent Mac memory overloading
    total_cores = os.cpu_count() or 1
    max_workers = min(3, total_cores) 
    
    print(f"      Spinning up {max_workers} processes for {len(cluster_args)} clusters...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker_process_cluster, arg): i for i, arg in enumerate(cluster_args)}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="      Extracting Buildings"):
            try:
                res = future.result()
                if res:
                    house_list.extend(res)
            except concurrent.futures.process.BrokenProcessPool:
                print(f"\n      [FATAL WARNING] A worker process crashed (likely ran out of RAM).")
            except Exception as e:
                cluster_id = futures[future]
                print(f"\n      [WARNING] Cluster {cluster_id} failed with error: {e}")

    print(f"      Done in {time.time() - step_start:.2f}s")

    print("[7/7] Finalizing CSV Output...")
    csv_path = os.path.join(output_dir, "house_measurements.csv")
    if house_list:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["house_ID", "Area_sqft", "Volume_cuft", "Best_Image"])
            writer.writeheader()
            writer.writerows(house_list)
            
    total_elapsed = time.time() - global_start_time
    
    print("\n" + "="*40)
    print("--- Extraction Complete ---")
    print(f"Total Processing Time:   {total_elapsed:.2f} seconds")
    print(f"Unique Buildings Found:  {len(house_list)}")
    print("="*40 + "\n")

if __name__ == "__main__":
    import sys
    if 'ipykernel' in sys.modules or 'IPython' in sys.modules:
        if not hasattr(sys.modules['__main__'], '__spec__'):
            sys.modules['__main__'].__spec__ = None

    test_file = 'data/ElmA60H90P12/scene_dense.ply'
    raw_images_directory = 'data/ElmA60H90P12/images' 
     
    if not os.path.exists(test_file):
        print("Error: Could not find the dense point cloud.")
        sys.exit(1)
        
    process_reconstruction_v22(test_file, raw_images_directory)