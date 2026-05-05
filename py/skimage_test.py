import os
import cv2
import glob
import numpy as np
from skimage import filters, morphology

# --- THE NUMPY 1.x BRIDGE ---
if not hasattr(np, 'bool'):
    np.bool = np.bool_

def batch_process_inverse_masking(input_dir, output_dir):
    # 1. Setup Directories
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    extensions = ['*.JPG', '*.jpg', '*.jpeg']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images. Starting inverse masking...")

    for img_path in image_paths:
        # 2. Load Image
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        img = cv2.imread(img_path)
        
        if img is None:
            continue

        # Convert to float for spectral math
        b = img[:, :, 0].astype(float)
        g = img[:, :, 1].astype(float)
        r = img[:, :, 2].astype(float)

        # 3. Calculate Excess Green Index (ExG)
        # Formula: $$ExG = 2G - R - B$$
        exg = (2 * g) - r - b

        # 4. Generate and Invert the Mask
        try:
            thresh = filters.threshold_otsu(exg)
            
            # This identifies the vegetation
            veg_mask = exg > thresh
            
            # INVERSION: We flip the mask so that Vegetation = False (0)
            # and Background = True (1)
            inverse_mask = ~veg_mask
            
            # Morphological cleaning on the inverse to smooth the edges
            # of the "holes" we are cutting out
            inverse_mask = morphology.binary_closing(inverse_mask, morphology.disk(2))
            inverse_mask = morphology.remove_small_holes(inverse_mask, area_threshold=500)
            
            # Convert to uint8 for OpenCV
            mask_uint8 = (inverse_mask * 255).astype(np.uint8)

            # 5. Apply Inverted Mask to Image
            # This keeps everything EXCEPT the trees
            result_image = cv2.bitwise_and(img, img, mask=mask_uint8)

            # 6. Save with requested naming convention: "name" + "mask"
            output_filename = f"{name}mask{ext}"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, result_image)
            
            print(f"Processed: {output_filename}")

        except ValueError:
            print(f"Skipping {filename}: Could not determine threshold.")

    print(f"\nBatch complete. Outputs saved to: {output_dir}")

# --- EXECUTION ---
input_folder = "data/test"
output_folder = "data/output_mask"

batch_process_inverse_masking(input_folder, output_folder)