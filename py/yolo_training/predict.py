from ultralytics import YOLO
import pandas as pd
import os

def run_yolo_prediction(model_path, source_path, project_name, conf_threshold=None, save_csv=True, save_image=True):
    """
    Runs YOLO prediction. 
    """
    # Load the model
    model = YOLO(os.path.abspath(model_path)) 

    # Predict arguments
    predict_args = {
        "source": source_path,
        "name": project_name,
        "show": False,
        "save": save_image,      
        "line_width": 1,
        "save_crop": False,
        "save_txt": True,        # We usually need this for Step 3 (matching), so keep True
        "show_labels": True,
        "show_conf": True,
        "classes": [0,1,2,3,4,5]
    }
    
    # Only apply confidence threshold if one is provided
    if conf_threshold is not None:
        predict_args["conf"] = conf_threshold

    # Run prediction
    results = model.predict(**predict_args)

    # Determine paths where YOLO saved the output
    save_dir = results[0].save_dir 
    labels_dir = os.path.join(save_dir, "labels")
    
    csv_output_path = None

    if save_csv:
        # 2. Extract Data
        all_detection_data = []
        class_names = model.names 

        for result in results:
            if result.path:
                source_file = os.path.basename(result.path) 
            else:
                source_file = "N/A" 
            
            # Use xywhn (Normalized Center-XY, Width, Height)
            if result.boxes:
                boxes = result.boxes.xywhn.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for i in range(len(boxes)):
                    # Extract normalized coordinates
                    x_c, y_c, w, h = [round(float(coord), 6) for coord in boxes[i]]
                    
                    confidence = round(float(confidences[i]), 4)
                    class_id = class_ids[i]
                    class_name = class_names.get(class_id, f"Class_{class_id}")
                    
                    all_detection_data.append({
                        'Source_File': source_file,
                        'Class_ID': class_id,
                        'Class_Name': class_name,
                        'Confidence': confidence,
                        'Normalized_X_Center': x_c,
                        'Normalized_Y_Center': y_c,
                        'Normalized_Width': w,
                        'Normalized_Height': h
                    })

        # 3. Create DataFrame and Save
        csv_filename = f"{project_name}.csv"
        csv_output_path = os.path.join(save_dir, csv_filename)

        if all_detection_data:
            df = pd.DataFrame(all_detection_data)
            df.to_csv(csv_output_path, index=False)
            print(f"\n✅ Successfully saved {len(df)} detections to: {csv_output_path}")
        else:
            print("\n⚠️ No detections found. Creating empty CSV.")
            # Create empty CSV to prevent errors in downstream scripts
            df = pd.DataFrame(columns=['Source_File', 'Class_ID', 'Class_Name', 'Confidence', 
                                     'Normalized_X_Center', 'Normalized_Y_Center', 'Normalized_Width', 'Normalized_Height'])
            df.to_csv(csv_output_path, index=False)

    return csv_output_path, labels_dir

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    # --- CONFIGURATION ---
    MODEL_PATH = "/Users/willicon/Desktop/dronemodeling_Logan/train_orem/weights/best.pt" 
    SOURCE_PATH = "/Users/willicon/Desktop/orem_windsor_seed15/images" 
    PROJECT_NAME = "orem_windsor"
    SAVE_CSV = True 
    
    # Run the function using the hard-coded values above
    run_yolo_prediction(
        model_path=MODEL_PATH, 
        source_path=SOURCE_PATH, 
        project_name=PROJECT_NAME,
        save_csv=SAVE_CSV,
        save_image= False # Default to True when running manually
    )