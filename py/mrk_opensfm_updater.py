import os
import json
import glob
import re

def update_opensfm_with_mrk(project_path, mrk_file_path):
    exif_path = os.path.join(project_path, 'exif')
    
    # 1. Parse your specific MRK format
    mrk_data = {}
    with open(mrk_file_path, 'r') as f:
        for line in f:
            # Your file uses tabs to separate the main chunks
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
                
            try:
                # Part 0: Index (e.g., "1")
                idx = parts[0].strip()
                
                # Part 4: Contains Lat, Lon, Alt, etc. separated by commas
                # Format: "40.25316878,Lat	-111.64008739,Lon	1456.818,Ellh"
                # Note: Depending on tabs vs spaces, we'll search specifically for the patterns
                
                lat_match = re.search(r"([-+]?\d*\.\d+|\d+),Lat", line)
                lon_match = re.search(r"([-+]?\d*\.\d+|\d+),Lon", line)
                alt_match = re.search(r"([-+]?\d*\.\d+|\d+),Ellh", line)
                
                if lat_match and lon_match and alt_match:
                    mrk_data[idx] = {
                        "lat": float(lat_match.group(1)),
                        "lon": float(lon_match.group(1)),
                        "alt": float(alt_match.group(1))
                    }
            except (ValueError, IndexError):
                continue

    # 2. Match with OpenSfM EXIF JSONs
    # We assume images are named or sorted in the order they appear in the MRK
    json_files = sorted(glob.glob(os.path.join(exif_path, "*.json")))

    for i, json_file in enumerate(json_files, start=1):
        idx_str = str(i)
        if idx_str in mrk_data:
            with open(json_file, 'r') as f:
                data = json.load(f)

            data['gps']['latitude'] = mrk_data[idx_str]['lat']
            data['gps']['longitude'] = mrk_data[idx_str]['lon']
            data['gps']['altitude'] = mrk_data[idx_str]['alt']
            
            # Since your file shows "1,Q" (Fixed RTK), we can trust this deeply.
            data['gps']['dop'] = 0.01 

            with open(json_file, 'w') as f:
                json.dump(data, f, indent=4)
            
            print(f"Updated {os.path.basename(json_file)}: Lat {mrk_data[idx_str]['lat']}")

# Usage: 
update_opensfm_with_mrk("/Users/willicon/Desktop/OpenSfM/data/ElmA60H90", 
                        '/Users/willicon/Desktop/OpenSfM/data/ElmA60H90/DJI_202602281053_016_ElmA60H90_Timestamp.MRK')