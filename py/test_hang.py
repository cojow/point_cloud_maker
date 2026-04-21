import open3d as o3d
import numpy as np
import concurrent.futures
import time

def worker_task(task_id):
    # Simulate exactly what your script does: load Open3D, do math, drop memory
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(5000, 3))
    labels = np.array(pcd.cluster_dbscan(eps=0.1, min_points=10))
    return f"Task {task_id} completed."

if __name__ == "__main__":
    print("Starting fast Open3D multiprocessing diagnostic...")
    start_time = time.time()
    
    # Spin up 4 workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker_task, i) for i in range(10)]
        for f in concurrent.futures.as_completed(futures):
            print(f.result())
            
    # If the script hangs before printing this next line, we have proven the Open3D destructor bug.
    print(f"SUCCESS: Pool closed and exited cleanly in {time.time() - start_time:.2f} seconds!")