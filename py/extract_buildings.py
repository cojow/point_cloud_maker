import numpy as np
import open3d as o3d
import os
import csv
from shapely.geometry import MultiPoint, Polygon
from scipy.spatial import KDTree
from matplotlib.path import Path
import sys

def generate_architectural_footprint(points_2d, shrink_dist=1.5):
    """
    Implements Morphological Regularization (Erosion & Dilation) 
    to sever irregular organic shapes from solid architectural blocks.
    """
    # 1. Create a tight wrapper around all the points
    raw_points = MultiPoint(points_2d)
    base_footprint = raw_points.buffer(0.4) 
    
    # 2. EROSION: Shrink the footprint inward to sever the tree connections
    eroded = base_footprint.buffer(-shrink_dist)
    
    if eroded.is_empty:
        return None
        
    # 3. SELECTION: If the tree disconnected, keep only the largest block (the house)
    if eroded.geom_type == 'MultiPolygon':
        largest_poly = max(eroded.geoms, key=lambda a: a.area)
    else:
        largest_poly = eroded
        
    # 4. DILATION: Expand the house back to its true original size
    dilated = largest_poly.buffer(shrink_dist)
    
    # 5. SIMPLIFICATION: Force the organic edges to snap into straight architectural lines
    architectural_footprint = dilated.simplify(0.5, preserve_topology=True)
    
    return architectural_footprint

def apply_rectilinear_cookie_cutter(ag_pts, master_footprint, area_m2, avg_h):
    min_x, min_y, max_x, max_y = master_footprint.bounds
    
    box_mask = (ag_pts[:, 0] >= min_x) & (ag_pts[:, 0] <= max_x) & \
               (ag_pts[:, 1] >= min_y) & (ag_pts[:, 1] <= max_y)
    box_pts = ag_pts[box_mask]
    
    if len(box_pts) == 0: return None
    
    if master_footprint.geom_type == 'Polygon':
        polys = [master_footprint]
    else:
        polys = list(master_footprint.geoms)
        
    final_mask = np.zeros(len(box_pts), dtype=bool)
    points_2d = box_pts[:, :2]
    
    for poly in polys:
        path = Path(np.asarray(poly.exterior.coords))
        final_mask |= path.contains_points(points_2d)
        
    final_house_pts = box_pts[final_mask]
    
    if len(final_house_pts) == 0: return None
    
    max_h = final_house_pts[:, 2].max()
    min_h = final_house_pts[:, 2].min()
    
    if max_h > 2.5 and (max_h - min_h) > 1.5:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(final_house_pts))
        volume = area_m2 * avg_h
        return pcd, round(area_m2, 2), round(volume, 2)
    return None

def process_reconstruction_v24(ply_path):
    output_dir = os.path.join(os.path.dirname(ply_path), "house_analysis_v24")
    debug_dir = os.path.join(output_dir, "individual_houses")
    for d in [output_dir, debug_dir]:
        if not os.path.exists(d): os.makedirs(d)
        
    print(f"--- Loading point cloud from {ply_path} ---")
    pcd = o3d.io.read_point_cloud(ply_path)

    print("Cleaning statistical outliers...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.2)
    pts_clean = np.asarray(pcd.points)

    print("Leveling point cloud via PCA...")
    z_vals = pts_clean[:, 2]
    ground_thresh = np.percentile(z_vals, 30)
    ground_pts = pts_clean[z_vals < ground_thresh]

    centroid = np.mean(ground_pts, axis=0)
    cov = np.cov(ground_pts.T)
    evals, evecs = np.linalg.eigh(cov)
    plane_norm = evecs[:, 0] 
    if plane_norm[2] < 0: plane_norm = -plane_norm

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

    print("Applying RANSAC to drop the ground plane...")
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.5, ransac_n=3, num_iterations=1000)
    
    ground_pcd = pcd.select_by_index(inliers)
    ground_z = np.median(np.asarray(ground_pcd.points)[:, 2])
    
    above_ground_pcd = pcd.select_by_index(inliers, invert=True)
    above_ground_pcd.translate((0, 0, -ground_z))
    ag_pts = np.asarray(above_ground_pcd.points)

    print("Isolating potential roofs...")
    high_idx = np.where(ag_pts[:, 2] > 2.2)[0]
    high_pts = ag_pts[high_idx]
    
    if len(high_pts) < 10: return

    # Pure standalone tree killer (Z-Variance)
    tree = KDTree(high_pts)
    _, neighbor_indices = tree.query(high_pts, k=10) 
    high_neighborhoods = high_pts[neighbor_indices]
    neighborhood_stdev = np.std(high_neighborhoods[:, :, 2], axis=1)
    planar_mask = neighborhood_stdev < 0.20
    
    roof_idx = high_idx[planar_mask]
    roof_pcd = above_ground_pcd.select_by_index(roof_idx)
    real_roof_pts = np.asarray(roof_pcd.points)
    
    print("Performing broad clustering pass (Z-squish)...")
    roof_pts_squished = np.copy(real_roof_pts)
    roof_pts_squished[:, 2] *= 0.1 
    clustering_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(roof_pts_squished))
    
    labels = np.array(clustering_pcd.cluster_dbscan(eps=2.5, min_points=15))
    
    if labels.size == 0 or labels.max() < 0: return

    house_list = []
    
    for i in range(labels.max() + 1):
        broad_idx = np.where(labels == i)[0]
        current_blob_roof_pts = real_roof_pts[broad_idx]
        
        points_2d = current_blob_roof_pts[:, :2]
        if len(points_2d) < 3: continue
        
        # --- FIX: YOUR POST-PROCESSING GEOMETRIC FILTER ---
        # Instead of guessing the shape, we mathematically regularize it
        master_footprint = generate_architectural_footprint(points_2d, shrink_dist=1.5)
        
        if master_footprint is None or master_footprint.area < 20: 
            continue
            
        sc_area = master_footprint.area
        avg_h = current_blob_roof_pts[:, 2].mean()
        
        result = apply_rectilinear_cookie_cutter(ag_pts, master_footprint, sc_area, avg_h)
        
        if result:
            final_pcd, area_meas, vol_meas = result
            centroid = np.mean(final_pcd.points, axis=0)
            global_x = centroid[0] + original_offset_xy[0]
            global_y = centroid[1] + original_offset_xy[1]
            unique_id = f"H_{abs(global_x):.5f}_{abs(global_y):.5f}".replace('.', 'd')
            
            # Build floor perfectly matching the regularized architectural footprint
            min_x, min_y, max_x, max_y = master_footprint.bounds
            grid_points = [[x, y, 0.0] for x in np.arange(min_x, max_x, 0.6) 
                           for y in np.arange(min_y, max_y, 0.6)]
            
            floor_pcd = o3d.geometry.PointCloud()
            grid_pts_array = np.array(grid_points)
            
            if len(grid_pts_array) > 0:
                if master_footprint.geom_type == 'Polygon':
                    polys = [master_footprint]
                else:
                    polys = list(master_footprint.geoms)
                    
                floor_mask = np.zeros(len(grid_pts_array), dtype=bool)
                grid_2d = grid_pts_array[:, :2]
                
                for poly in polys:
                    path = Path(np.asarray(poly.exterior.coords))
                    floor_mask |= path.contains_points(grid_2d)
                    
                valid_floor = grid_pts_array[floor_mask]
                
                if len(valid_floor) > 0:
                    floor_pcd.points = o3d.utility.Vector3dVector(valid_floor)
                    floor_pcd.paint_uniform_color([0.5, 0.5, 0.5])
            
            o3d.io.write_point_cloud(os.path.join(debug_dir, f"{unique_id}.ply"), final_pcd + floor_pcd)
            
            house_list.append({
                "house_ID": unique_id, 
                "Area_m2": area_meas, 
                "volume_m3": vol_meas
            })

    csv_path = os.path.join(output_dir, "house_measurements.csv")
    if house_list:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["house_ID", "Area_m2", "volume_m3"])
            writer.writeheader()
            writer.writerows(house_list)
    
    print("--- Top-Down Extraction complete ---")
    print(f"Deterministically identified {len(house_list)} unique buildings.")

if __name__ == "__main__":
    test_file = 'data/ElmA60H90P12/scene_dense.ply'
    if not os.path.exists(test_file):
        print(f"Error: Could not find the dense point cloud.")
        sys.exit(1)
        
    process_reconstruction_v24(test_file)