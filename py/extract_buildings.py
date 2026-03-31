import numpy as np
import open3d as o3d
import os
import csv
from scipy.spatial import Delaunay, KDTree
import sys

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
    circum_r = lengths[0, :] * lengths[1, :] * lengths[2, :] / (
        np.sqrt((lengths[0, :] + lengths[1, :] + lengths[2, :]) *
                (-lengths[0, :] + lengths[1, :] + lengths[2, :]) *
                (lengths[0, :] - lengths[1, :] + lengths[2, :]) *
                (lengths[0, :] + lengths[1, :] - lengths[2, :]))
    )
    
    final_simplices = tri.simplices[circum_r < alpha]
    if len(final_simplices) == 0: return None
    
    hull_pts_3d = np.concatenate([points_2d, np.zeros((len(points_2d), 1))], axis=1)
    unique_indices = np.unique(final_simplices)
    index_map = {old: new for new, old in enumerate(unique_indices)}
    mapped_simplices = np.vectorize(index_map.get)(final_simplices)
    
    final_hull_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(hull_pts_3d[unique_indices]))
    hull_mesh = o3d.geometry.TriangleMesh(final_hull_pcd.points, o3d.utility.Vector3iVector(mapped_simplices))
    
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(hull_mesh, voxel_size=0.3)
    return voxel_grid

def apply_house_cookie_cutter(ag_pts, concave_voxel_grid, area_m2, avg_h):
    min_bound = concave_voxel_grid.get_min_bound()
    max_bound = concave_voxel_grid.get_max_bound()
    
    box_mask = (ag_pts[:, 0] >= min_bound[0]) & (ag_pts[:, 0] <= max_bound[0]) & \
               (ag_pts[:, 1] >= min_bound[1]) & (ag_pts[:, 1] <= max_bound[1])
    box_pts = ag_pts[box_mask]
    
    contain_mask = concave_voxel_grid.check_if_included(o3d.utility.Vector3dVector(box_pts * [1., 1., 0.]))
    final_house_pts = box_pts[contain_mask]
    if len(final_house_pts) == 0: return None
    
    max_h = final_house_pts[:, 2].max()
    min_h = final_house_pts[:, 2].min()
    
    if max_h > 2.5 and (max_h - min_h) > 1.5:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(final_house_pts))
        volume = area_m2 * avg_h
        return pcd, round(area_m2, 2), round(volume, 2)
    return None

def process_reconstruction_v22(ply_path):
    output_dir = os.path.join(os.path.dirname(ply_path), "house_analysis_v22")
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
        
        voxel_footprint = calculate_alpha_shape(points_2d, alpha=0.6)
        if voxel_footprint is None: continue
        
        mins, maxs = voxel_footprint.get_min_bound(), voxel_footprint.get_max_bound()
        area = (maxs[0] - mins[0]) * (maxs[1] - mins[1]) 

        if area > 35:
            if area > 350:
                blob_un_squished = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(current_blob_roof_pts))
                sub_labels = np.array(blob_un_squished.cluster_dbscan(eps=1.6, min_points=8))
                
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
                
                # --- NEW FIX: The Massive Threshold Bump ---
                # We now demand at least 500 points to even consider running RANSAC
                while len(remaining_pts) > 500:
                    pcd_temp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(remaining_pts))
                    plane_model, inliers = pcd_temp.segment_plane(distance_threshold=0.15, ransac_n=3, num_iterations=250)
                    
                    # RANSAC must find at least 500 perfectly flat points
                    if len(inliers) < 500: 
                        break 
                        
                    plane_pts = remaining_pts[inliers]
                    
                    plane_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(plane_pts))
                    plane_labels = np.array(plane_pcd.cluster_dbscan(eps=0.5, min_points=15))
                    
                    if plane_labels.max() >= 0:
                        largest_cluster_idx = np.bincount(plane_labels[plane_labels >= 0]).argmax()
                        contiguous_inliers = np.where(plane_labels == largest_cluster_idx)[0]
                        
                        # DBSCAN must confirm at least 400 of those flat points are physically touching
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
                if len(sc_2d) < 3: continue
                
                sc_footprint = calculate_alpha_shape(sc_2d, alpha=0.6)
                if sc_footprint is None: continue
                
                mins, maxs = sc_footprint.get_min_bound(), sc_footprint.get_max_bound()
                sc_area = (maxs[0] - mins[0]) * (maxs[1] - mins[1])
                avg_h = pure_roof_pts[:, 2].mean()
                
                if sc_area > 20:
                    result = apply_house_cookie_cutter(ag_pts, sc_footprint, sc_area, avg_h)
                    
                    if result:
                        final_pcd, area_meas, vol_meas = result
                        centroid = np.mean(final_pcd.points, axis=0)
                        global_x = centroid[0] + original_offset_xy[0]
                        global_y = centroid[1] + original_offset_xy[1]
                        unique_id = f"H_{abs(global_x):.5f}_{abs(global_y):.5f}".replace('.', 'd')
                        
                        grid_points = [[x, y, 0.0] for x in np.arange(mins[0], maxs[0], 0.6) 
                                       for y in np.arange(mins[1], maxs[1], 0.6)]
                        
                        floor_pcd = o3d.geometry.PointCloud()
                        grid_pts_array = np.array(grid_points)
                        
                        if len(grid_pts_array) > 0:
                            floor_mask = sc_footprint.check_if_included(o3d.utility.Vector3dVector(grid_pts_array))
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
        print("Error: Could not find the dense point cloud.")
        sys.exit(1)
        
    process_reconstruction_v22(test_file)