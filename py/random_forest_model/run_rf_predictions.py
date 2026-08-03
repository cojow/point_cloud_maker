import pandas as pd
import os
import joblib
import matplotlib
from pathlib import Path

matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- 1. PARAMETERS ---
input_file = 'output_data/orem_caschade_orchard_seed15_2026_02_04_083646/neighbor calculations/spatial_features_k5.csv'
model_path = 'py/random_forest_model/rf_models/orem_combined_rf_model.joblib'
name_adden = 'orem_combined'

#These variables should reflect the variables that the model was trained on.  'neighbor_mean_perimeter','neighbor_mean_area'
variables = ['Area_sqm', 'Perimeter_m', 'Confidence','neighbor_mean_conf', 'neighbor_majority_class','yolo_pred' ]
product = 'annotation_type'


try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

# --- 2. OUTPUT DIRECTORY LOGIC ---
target_folder_name = f"random forest prediction{name_adden}"
input_path_obj = Path(input_file)

if input_path_obj.exists():
    grandparent_dir = input_path_obj.parent.parent
    if grandparent_dir.exists():
        output_dir = os.path.join(grandparent_dir, target_folder_name)
    else:
        output_dir = os.path.join(script_dir, target_folder_name)
else:
    output_dir = os.path.join(script_dir, target_folder_name)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

filename = os.path.basename(input_file)
short_name = os.path.splitext(filename)[0].replace('_final', '') 
output_csv = os.path.join(output_dir, f'{short_name}_predictions.csv')

# --- 3. LOAD DATA & MODEL ---
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at: {model_path}")

print(f"Loading model from: {model_path}")
rf_model = joblib.load(model_path)

print(f"Loading data from: {input_file}")
df = pd.read_csv(input_file)

# Remove rows where the target is NaN (Empty/Null)
initial_count = len(df)
df = df.dropna(subset=[product])
print(f"Removed {initial_count - len(df)} rows with missing '{product}'. Remaining: {len(df)}")

#These variables should reflect the variables that the model was trained on.  'neighbor_mean_perimeter','neighbor_mean_area'
feature_cols = variables
X = df[feature_cols]
y = df[product]

# --- 4. PREDICT ON WHOLE DATASET ---
print("Running predictions on the entire dataset...")
# Predict on 100% of the data
y_pred = rf_model.predict(X)

# Save to CSV
df['RF_Prediction'] = y_pred
df.to_csv(output_csv, index=False)
print(f"Predictions saved to: {output_csv}")

# --- 5. VALIDATION (Whole Dataset) ---
print(f"Model Accuracy (Whole Dataset): {accuracy_score(y, y_pred):.4f}")
print(classification_report(y, y_pred, zero_division=0))

# --- 6. PLOTS ---

# A. Confusion Matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), bbox_inches='tight')
plt.close() 

# A-2. Normalized Confusion Matrix
cm_norm = confusion_matrix(y, y_pred, normalize='true')

plt.figure(figsize=(6, 5))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues') # fmt='.2f' for 0.85, fmt='.1%' for 85.0%
plt.title('Normalized Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix_normalized.png'), bbox_inches='tight')
plt.close()

# B. Feature Importance
if hasattr(rf_model, 'feature_importances_'):
    importances = rf_model.feature_importances_
    plt.figure(figsize=(10, 6)) # Increased width slightly for long feature names
    sns.barplot(x=importances, y=feature_cols)
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), bbox_inches='tight')
    plt.close()

# C. Correlation Matrix
plt.figure(figsize=(10, 8)) # Increased size slightly

# Create a subset with just features + target + prediction
cols_to_plot = feature_cols + ['yolo_pred', 'RF_Prediction']
correlation_subset = df[cols_to_plot]

# Plot correlation of just that subset
sns.heatmap(correlation_subset.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), bbox_inches='tight')
plt.close()

print("All plots saved successfully.")