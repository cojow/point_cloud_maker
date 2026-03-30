import json
import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt

def run_clustering_diagnostic(json_path):
    json_dir = os.path.dirname(os.path.abspath(json_path))
    out_dir = os.path.join(json_dir, "clustering_diagnostics")
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    with open(json_path, 'r') as f:
        data = json.load(f)
    pts = np.array([v['coordinates'] for v in data[0]['points'].values()])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # 1. Leveling (Using the solid PCA math from V13/V14)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.3, ransac_n=3, num_iterations=5000)
    [a, b, c, d_val] = plane_model
    target_norm = np.array([0, 0, 1])
    plane_norm = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    v = np.cross(plane_norm, target_norm)
    s = np.linalg.norm(v)
    c_val = np.dot(plane_norm, target_norm)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    pcd.rotate(np.eye(3) + vx + np.dot(vx, vx) * ((1 - c_val) / (s**2)), center=(0,0,0))
    pcd.translate((0, 0, -np.median(np.asarray(pcd.points)[inliers, 2])))
    if np.median(np.asarray(pcd.points)[:, 2]) < 0:
        pcd.rotate(np.array([[1,0,0],[0,1,0],[0,0,-1]]), center=(0,0,0))

    # Get points > 0.5m (The "Above Ground" set)
    obj_idx = np.where(np.asarray(pcd.points)[:, 2] > 0.5)[0]
    objects_pcd = pcd.select_by_index(obj_idx)

    # Helper function to colorize clusters
    def colorize_clusters(pcd_obj, eps, min_points, squish_z=False):
        pts_copy = np.asarray(pcd_obj.points).copy()
        if squish_z:
            pts_copy[:, 2] *= 0.1
            
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(pts_copy)
        labels = np.array(temp_pcd.cluster_dbscan(eps=eps, min_points=min_points))
        
        max_label = labels.max()
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0 # Noise is black
        
        # Apply colors to the ORIGINAL pcd, not the squished one
        colored_pcd = o3d.geometry.PointCloud()
        colored_pcd.points = pcd_obj.points
        colored_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        return colored_pcd

    # --- DIAGNOSTIC 1: Raw Above-Ground Data ---
    # Are the roof points even there to begin with?
    o3d.io.write_point_cloud(os.path.join(out_dir, "01_raw_above_ground.ply"), objects_pcd)

    # --- DIAGNOSTIC 2: The V12 Method (True 3D, Strict) ---
    v12_pcd = colorize_clusters(objects_pcd, eps=3.5, min_points=40, squish_z=False)
    o3d.io.write_point_cloud(os.path.join(out_dir, "02_v12_strict_3D.ply"), v12_pcd)

    # --- DIAGNOSTIC 3: The V14 Method (2.5D Squished, Relaxed) ---
    v14_pcd = colorize_clusters(objects_pcd, eps=3.5, min_points=15, squish_z=True)
    o3d.io.write_point_cloud(os.path.join(out_dir, "03_v14_squished_2.5D.ply"), v14_pcd)

    print(f"Diagnostics saved to {out_dir}")

run_clustering_diagnostic('data/ElmA60H90-24/reconstruction.json')