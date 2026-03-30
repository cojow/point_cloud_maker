import os
import itertools
import subprocess
import sys

def create_combination_workspace(workspace_dir, folder1_path, folder2_path):
    """Creates a workspace with an 'images' folder containing prefixed symlinks to the source images."""
    images_dir = os.path.join(workspace_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Bundle the folders with a unique prefix identifier
    folders_to_process = [
        ('F1', folder1_path),
        ('F2', folder2_path)
    ]
    
    for prefix, source_folder in folders_to_process:
        # Symlink all images with the new prefix
        for filename in os.listdir(source_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.tif', '.tiff')):
                src_file = os.path.join(source_folder, filename)
                # Prepend the prefix (e.g., F1_DJI_..._0001_D.JPG)
                dst_link = os.path.join(images_dir, f"{prefix}_{filename}") 
                
                if not os.path.exists(dst_link):
                    os.symlink(src_file, dst_link)
                    
        # Copy the .MRK files into the root of the workspace with the new prefix
        import shutil
        for filename in os.listdir(source_folder):
            if filename.upper().endswith('.MRK'):
                src_mrk = os.path.join(source_folder, filename)
                dst_mrk = os.path.join(workspace_dir, f"{prefix}_{filename}")
                if not os.path.exists(dst_mrk):
                    shutil.copy2(src_mrk, dst_mrk)

def main():
    # --- CONFIGURATION ---
    # Update this list with the exact paths to your 5 main flight folders
    parent_folders = [
        os.path.abspath("data/ElmA80H150P15"),
        os.path.abspath("data/ElmA70H150P14"),
        os.path.abspath("data/ElmA70H120P18"),
        os.path.abspath("data/ElmA80H120P20"),
        os.path.abspath("data/ElmA60H90P22")
    ]
    
    # Where the combined OpenSfM workspaces will be created
    output_base_dir = os.path.abspath("data/Combinations")
    
    # The script we want to run on each workspace
    auto_reconstruct_script = os.path.abspath("py/auto_reconstruct.py")
    
    # The specific bands we are testing right now
    target_bands = ['RGB', 'Red']
    # ---------------------

    # Generate all unique pairs of the parent folders (10 combinations)
    flight_pairs = list(itertools.combinations(parent_folders, 2))
    
    total_runs = len(flight_pairs) * len(target_bands)
    print(f"Starting pipeline: {total_runs} total runs queued.\n")

    run_counter = 1
    for flight1, flight2 in flight_pairs:
        flight1_name = os.path.basename(flight1)
        flight2_name = os.path.basename(flight2)
        
        for band in target_bands:
            # Construct paths to the specific band subfolders
            band_dir1 = os.path.join(flight1, band)
            band_dir2 = os.path.join(flight2, band)
            
            # Skip if the folders don't exist yet
            if not os.path.exists(band_dir1) or not os.path.exists(band_dir2):
                print(f"Skipping {flight1_name} + {flight2_name} ({band}) - Folders not found.")
                continue

            # Create a unique workspace name: e.g., "Combo_Flight1_Flight2_RGB"
            workspace_name = f"Combo_{flight1_name}_{flight2_name}_{band}"
            workspace_dir = os.path.join(output_base_dir, workspace_name)
            
            print(f"--- Run {run_counter}/{total_runs}: Building {workspace_name} ---")
            create_combination_workspace(workspace_dir, band_dir1, band_dir2)
            
            # Execute your auto_reconstruct.py script on this new workspace
            command = [sys.executable, auto_reconstruct_script, workspace_dir]
            print(f"Executing: {' '.join(command)}")
            
            result = subprocess.run(command)
            if result.returncode != 0:
                print(f"WARNING: OpenSfM failed on {workspace_name}")
            
            run_counter += 1

if __name__ == "__main__":
    main()