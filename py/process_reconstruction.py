import json
import numpy as np
import open3d as o3d
import os
import csv
import shutil
from shapely.geometry import MultiPoint, Point

def process_reconstruction_v16(json_path):
    json_dir = os.path.dirname(os.path.abspath(json_path))
    output_dir = os.path.join(json_dir, "house_analysis_v16")
    debug_dir = os.path.join(output_dir, "individual_houses")
    img_output_dir = os.path.join(output_dir, "representative_images")
    src_images_dir = os.path.join(json_dir, "images")
    
    for d in [output_dir, debug_dir, img_output_dir]:
        if not os.path.exists(d): os.makedirs(d)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    recon = data[0]
    points_dict = recon['points']
    shots = recon['shots']
    
    pts = np.array([v['coordinates'] for v in points_dict.values()])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # 1. Clean outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.2)
    pts_clean = np.asarray(pcd.points)

    # 2. 100% DETERMINISTIC LEVELING (PCA)
    # Grab the lowest 30% of points to mathematically define the ground
    z_vals = pts_clean[:, 2]
    ground_thresh = np.percentile(z_vals, 30)
    ground_pts = pts_clean[z_vals < ground_thresh]

    centroid = np.mean(ground_pts, axis=0)
    cov = np.cov(ground_pts.T)
    evals, evecs = np.linalg.eigh(cov)
    plane_norm = evecs[:, 0] # The normal vector of the ground
    if plane_norm[2] < 0: plane_norm = -plane_norm

    target_norm = np.array([0, 0, 1])
    v = np.cross(plane_norm, target_norm)
    s = np.linalg.norm(v)
    c_val = np.dot(plane_norm, target_norm)
    
    # Rotate to flat
    if s > 1e-6:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c_val) / (s**2))
        pcd.rotate(rotation_matrix, center=(0,0,0))
    else:
        rotation_matrix = np.eye(3)

    # Shift XY to the centroid anchor (Consistent IDs)
    original_offset_xy = centroid[:2]
    pcd.translate((-original_offset_xy[0], -original_offset_xy[1], 0))

    # Shift Z so ground is 0
    pts_rotated = np.asarray(pcd.points)
    z_rotated = pts_rotated[:, 2]
    new_ground_thresh = np.percentile(z_rotated, 30)
    ground_z = np.median(pts_rotated[z_rotated < new_ground_thresh][:, 2])
    pcd.translate((0, 0, -ground_z))

# 3. TOP-DOWN EXTRACTION
    pts_leveled = np.asarray(pcd.points)
    
    # Extract ALL points above ground for the final house model
    above_ground_pcd = pcd.select_by_index(np.where(pts_leveled[:, 2] > 0.5)[0])
    ag_pts = np.asarray(above_ground_pcd.points)

    # Extract ONLY roofs for clustering (> 2.2m tall)
    # We use the raw, un-filtered points so sparse roofs survive!
    roof_pcd = pcd.select_by_index(np.where(pts_leveled[:, 2] > 2.2)[0])
    roof_pts = np.asarray(roof_pcd.points).copy()
    
    # Squish Z to keep garages attached
    roof_pts[:, 2] *= 0.1 
    clustering_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(roof_pts))
    
    # PASS 1: Broad Clustering (The wide 2.5m net)
    labels = np.array(clustering_pcd.cluster_dbscan(eps=2.5, min_points=15))
    
    house_list = []
    
    if labels.size > 0 and labels.max() >= 0:
        for i in range(labels.max() + 1):
            r_cluster_idx = np.where(labels == i)[0]
            real_roof_pts = np.asarray(roof_pcd.points)[r_cluster_idx]
            
            points_2d = real_roof_pts[:, :2]
            if len(points_2d) < 3: continue
            hull = MultiPoint(points_2d).convex_hull
            area = hull.area

            # Only process if it's at least the size of a small structure
            if area > 35:
                # --- TWO-PASS OVERSIZED FILTER ---
                if area > 400:
                    blob_pts = np.copy(real_roof_pts)
                    blob_pts[:, 2] *= 0.1 
                    blob_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(blob_pts))
                    
                    sub_labels = np.array(blob_pcd.cluster_dbscan(eps=2.0, min_points=5))
                    
                    sub_clusters = []
                    if sub_labels.max() >= 0:
                        for k in range(sub_labels.max() + 1):
                            sub_idx = np.where(sub_labels == k)[0]
                            sub_clusters.append(real_roof_pts[sub_idx])
                else:
                    sub_clusters = [real_roof_pts]

                # Process the separated footprints
                for sc_pts in sub_clusters:
                    sc_2d = sc_pts[:, :2]
                    if len(sc_2d) < 3: continue
                    sc_hull = MultiPoint(sc_2d).convex_hull
                    sc_area = sc_hull.area
                    
                    # Area check for the final footprint
                    if sc_area > 20:
                        
                        # --- THE GEOMETRIC TREE FILTER ---
                        perimeter = sc_hull.length
                        if perimeter == 0: continue
                        
                        # Circularity Math: Circle ≈ 1.0, Square ≈ 0.78, Rectangle < 0.7
                        circularity = (4 * np.pi * sc_area) / (perimeter ** 2)
                        
                        # Pine trees form near-perfect circles. Houses have corners.
                        # If the footprint is highly circular, we skip it.
                        if circularity > 0.82:
                            continue # It's a tree, throw it out!
                        
# 4. THE COOKIE CUTTER
                        min_x, min_y, max_x, max_y = sc_hull.bounds
                        
                        box_mask = (ag_pts[:, 0] >= min_x) & (ag_pts[:, 0] <= max_x) & \
                                   (ag_pts[:, 1] >= min_y) & (ag_pts[:, 1] <= max_y)
                        box_pts = ag_pts[box_mask]
                        
                        raw_house_pts = np.array([p for p in box_pts if sc_hull.contains(Point(p[0], p[1]))])
                        if len(raw_house_pts) == 0: continue

                        # --- THE NEW FIX: The Vertical Ceiling Slice ---
                        # We chop off the extreme top 0.5% of the Z-axis to kill floaters
                        # without destroying the sparse horizontal layout of the roof.
                        z_vals = raw_house_pts[:, 2]
                        ceiling_h = np.percentile(z_vals, 99.5) 
                        
                        # Keep points below the ceiling (plus a 10cm safety buffer)
                        final_house_pts = raw_house_pts[z_vals <= (ceiling_h + 0.1)]
                        
                        if len(final_house_pts) == 0: continue

                        # Go back to standard max/min now that the floaters are deleted
                        max_h = final_house_pts[:, 2].max()
                        min_h = final_house_pts[:, 2].min()
                        avg_h = final_house_pts[:, 2].mean()

                        # Re-applied the strict Height Rules to keep tall bushes and fences out
                        if max_h > 2.5 and (max_h - min_h) > 1.5:

                            # ID Generation
                            local_centroid = np.mean(final_house_pts[:, :2], axis=0)
                            global_x = local_centroid[0] + original_offset_xy[0]
                            global_y = local_centroid[1] + original_offset_xy[1]
                            unique_id = f"H_{abs(global_x):.5f}_{abs(global_y):.5f}".replace('.', 'd')
                            
                            # --- BEST IMAGE SELECTION ---
                            best_image = None
                            min_dist = float('inf')
                            for shot_id, shot_data in shots.items():
                                rot_vec = np.array(shot_data['rotation'])
                                trans_vec = np.array(shot_data['translation'])
                                
                                theta = np.linalg.norm(rot_vec)
                                if theta > 1e-10:
                                    axis = rot_vec / theta
                                    K = np.array([
                                        [0, -axis[2], axis[1]],
                                        [axis[2], 0, -axis[0]],
                                        [-axis[1], axis[0], 0]
                                    ])
                                    R_cam = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
                                else:
                                    R_cam = np.eye(3)
                                    
                                true_cam_world = -np.dot(R_cam.T, trans_vec)
                                
                                if s > 1e-6: 
                                    cam_leveled = rotation_matrix.dot(true_cam_world)
                                else:
                                    cam_leveled = true_cam_world
                                    
                                cam_final = cam_leveled[:2] - original_offset_xy
                                
                                dist = np.linalg.norm(local_centroid - cam_final)
                                if dist < min_dist:
                                    min_dist = dist
                                    best_image = shot_id

                            if best_image:
                                src_img = os.path.join(src_images_dir, best_image)
                                dst_img = os.path.join(img_output_dir, f"{unique_id}.jpg")
                                if os.path.exists(src_img):
                                    shutil.copy2(src_img, dst_img)
                            
                            # Generate Synthetic Floor & Export
                            grid_points = [[x, y, 0.0] for x in np.arange(min_x, max_x, 0.6) 
                                           for y in np.arange(min_y, max_y, 0.6) if sc_hull.contains(Point(x, y))]
                            
                            final_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(final_house_pts))
                            floor_pcd = o3d.geometry.PointCloud()
                            if grid_points:
                                floor_pcd.points = o3d.utility.Vector3dVector(np.array(grid_points))
                                floor_pcd.paint_uniform_color([0.5, 0.5, 0.5])
                            
                            o3d.io.write_point_cloud(os.path.join(debug_dir, f"{unique_id}.ply"), final_pcd + floor_pcd)
                            house_list.append({
                                "house_ID": unique_id, 
                                "Area_m2": round(sc_area, 2), 
                                "volume_m3": round(sc_area * avg_h, 2),
                                "representative_image": f"{unique_id}.jpg"
                            })

    # 5. Save Results
    csv_path = os.path.join(output_dir, "house_measurements.csv")
    if house_list:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["house_ID", "Area_m2", "volume_m3", "representative_image"])
            writer.writeheader()
            writer.writerows(house_list)
    
    print("--- Top-Down Extraction complete ---")
    print(f"Identified {len(house_list)} unique buildings.")

# Execute the function
process_reconstruction_v16('data/ElmBaseLine/reconstruction.json')