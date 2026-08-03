import pandas as pd
import os
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- 1. PARAMETERS ---
input_files = [
    'output_data/orem_caschade_orchard_seed15_2026_02_04_083646/neighbor calculations/spatial_features_k5.csv',
    'output_data/orem_geneva_suncrest_seed15_2026_02_04_094210/neighbor calculations/spatial_features_k5.csv',
    'output_data/orem_sharon_riverbottoms_seed15_2026_02_04_094415/neighbor calculations/spatial_features_k5.csv',
    'output_data/orem_timpview_seed15_2026_02_04_094301/neighbor calculations/spatial_features_k5.csv',
    'output_data/orem_windsor_seed15_2026_02_04_094605/neighbor calculations/spatial_features_k5.csv',
    'output_data/vinyard_seed15_2026_02_04_094507/neighbor calculations/spatial_features_k5.csv'
    # Add other file paths here
]
random_state = 42 # Used to create the random string of variables when creating the random forrest.
test_size = 0.4 # The lower the test size, the more of the buildings used in training. 
                  #Keep at 0.001 if the data used to train will not be the same data used to perdict.

name = 'orem_combined' #Name for model

# Variables are what the model will train itself on. Give as a list of columns.
# Product is what column is being solved for.
variables = ['Area_sqm', 'Perimeter_m', 'Confidence','neighbor_mean_conf', 'neighbor_majority_class','yolo_pred']
product = 'annotation_type'

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

# --- 2. MODEL NAMING ---
# Define the subfolder
subfolder = "rf_models"
full_output_dir = os.path.join(script_dir, subfolder)

# CRITICAL: Create the folder if it doesn't exist
if not os.path.exists(full_output_dir):
    os.makedirs(full_output_dir)

# Create the full path
model_filename = f"{name}_rf_model.joblib"
model_output_path = os.path.join(full_output_dir, model_filename)

# --- 3. LOAD, CLEAN & MERGE MULTIPLE FILES ---
dataframes_list = []

print(f"--- Loading {len(input_files)} datasets ---")

for file_path in input_files:
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: File not found, skipping: {file_path}")
        continue
        
    temp_df = pd.read_csv(file_path)
    
    # --- NEW LOGIC START ---
    # 1. Get the Grandparent Folder Name (Source Name)
    dataset_source_name = Path(file_path).parent.parent.name
    
    # 2. Create a NEW column with this name
    temp_df['Dataset_Source'] = dataset_source_name
    # --- NEW LOGIC END ---

    # 3. Check for missing columns (excluding the new source column)
    missing_cols = [col for col in variables + [product] if col not in temp_df.columns]
    if missing_cols:
        print(f"⚠️ Warning: Skipping {os.path.basename(file_path)} - Missing columns: {missing_cols}")
        continue

    # 4. Remove NaNs (Cleaning)
    initial_count = len(temp_df)
    temp_df = temp_df.dropna(subset=[product])
    dropped_count = initial_count - len(temp_df)
    
    if dropped_count > 0:
        print(f"   - {dataset_source_name}: Removed {dropped_count} rows (NaN in target).")
    
    # 5. Define which columns to keep
    # We keep the variables, the product, AND our new Source column.
    # We also keep 'ID' if it exists, just for your reference, but we don't train on it.
    cols_to_keep = variables + [product, 'Dataset_Source']
    
    if 'ID' in temp_df.columns:
        cols_to_keep.append('ID')

    # Filter the dataframe to only necessary columns
    temp_df = temp_df[cols_to_keep]
    
    dataframes_list.append(temp_df)

# Combine
if not dataframes_list:
    raise ValueError("❌ No valid data loaded.")

df_combined = pd.concat(dataframes_list, ignore_index=True)
print(f"✅ Successfully merged data. Total Training Rows: {len(df_combined)}")

# --- 4. PREPARE TRAINING DATA ---
# X only gets the variables (Physical features)
# It does NOT get 'ID' or 'Dataset_Source'
X = df_combined[variables]
y = df_combined[product]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

print(f"Training on {len(X_train)} samples...")

# --- 5. TRAIN MODEL ---
rf_model = RandomForestClassifier(random_state=random_state)
rf_model.fit(X_train, y_train)

# --- 6. SAVE MODEL ---
joblib.dump(rf_model, model_output_path)

print(f"✅ Model trained and saved to: {model_output_path}")