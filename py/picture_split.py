import os
import shutil

def organize_mavic3m_data(source_dir, dest_dir=None):
    """
    Organizes DJI Mavic 3M files into separate folders by band.
    Moves image files and copies/removes .MRK files to distribute them to all folders.
    """
    if dest_dir is None:
        dest_dir = source_dir

    bands = {
        'RGB': '_D.JPG',
        'Green': '_MS_G.TIF',
        'NIR': '_MS_NIR.TIF',
        'Red': '_MS_R.TIF',
        'RedEdge': '_MS_RE.TIF'
    }

    # Create destination folders
    for folder in bands.keys():
        os.makedirs(os.path.join(dest_dir, folder), exist_ok=True)

    # Process each file in the source directory
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)

        # Skip subdirectories
        if os.path.isdir(file_path):
            continue

        filename_upper = filename.upper()

        # Handle .MRK files: Copy to all 5 folders, then delete the original
        if filename_upper.endswith('.MRK'):
            for folder in bands.keys():
                dest_path = os.path.join(dest_dir, folder, filename)
                shutil.copy2(file_path, dest_path)
            
            os.remove(file_path) # Remove original after successful copies
            print(f"Distributed and removed original MRK file: {filename}")
            continue

        # Handle image files: Move to their respective folder
        for folder, suffix in bands.items():
            if filename_upper.endswith(suffix):
                dest_path = os.path.join(dest_dir, folder, filename)
                shutil.move(file_path, dest_path) 
                print(f"Moved: {filename} -> {folder}/")
                break

if __name__ == "__main__":
    # Update with your directory path
    SOURCE_DIRECTORY = r"/Users/connor/Desktop/ElmA80H120P20"
    OUTPUT_DIRECTORY = None 

    organize_mavic3m_data(SOURCE_DIRECTORY, OUTPUT_DIRECTORY)
    print("Organization complete.")