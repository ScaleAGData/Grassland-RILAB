# -*- coding: utf-8 -*-
"""
PROJECT: ScaleAgData - Work Package 5 (SpainSite Grassland)
DESCRIPTION:
    This script trains a Multi-Layer Perceptron (MLP) Neural Network to estimate 
    Gross Primary Production (GPP) using Sentinel-2 satellite reflectance data 
    and in-situ Eddy Covariance measurements (IFAPA).

WORKFLOW:
    1. Feature Engineering: Calculates NDVI and extracts Day of Year (DOY) for seasonality.
    2. Data Cleaning: Filters out noisy atmospheric bands (B01, B09, B10).
    3. Normalization: Scales inputs and targets using MinMaxScaler.
    4. Optimization: Runs a multi-seed loop to find the best weight initialization 
       based on Validation Set performance (R-squared).
    5. Evaluation: Assesses the final model on an independent Test Set.
    6. Export: Saves the model, scalers, and a detailed Excel report with metrics.

@author: Paolo Cosmo Silvestro
Updated on: April 20, 2026
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib

# --- Helper Functions for Statistical Metrics ---

def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def relative_root_mean_squared_error(true, pred):
    """RRMSE: Normalized version of RMSE."""
    n = len(true)
    num = np.sum(np.square(true - pred)) / n
    den = np.sum(np.square(pred))
    return np.sqrt(num/den) if den != 0 else 0

def relative_absolute_error(true, pred):
    """RAE: Ratio of absolute error relative to the mean of true values."""
    true_mean = np.mean(true)
    num = np.sum(np.abs(true - pred))
    den = np.sum(np.abs(true - true_mean))
    return num / den if den != 0 else 0

### 1. CONFIGURATION & PATHS ###

filename_ref = 'IFAPA_insitu_Sentinel2_Test1'
# Automatically set base folder to the script location
base_folder = os.path.dirname(os.path.abspath(__file__))

# Create a unique output directory using a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = os.path.join(base_folder, f"Run_GPP_Cleaned_{timestamp}")
os.makedirs(output_folder, exist_ok=True)

# ANN Hyperparameters
hidden_layer_sizes = (12, 6)  # Architecture: 2 hidden layers
activation = 'relu'           # Rectified Linear Unit function
solver = 'lbfgs'              # Optimizer: quasi-Newton method (stable for smaller datasets)
max_iter = 5000               # High limit for convergence
target_r2_val = 0.90          # Stop early if this Validation R2 is met
max_attempts = 10000          # Number of random initializations to test

### 2. DATA LOADING & FEATURE ENGINEERING ###

excelfile = os.path.join(base_folder, filename_ref + '.xlsx')
df = pd.read_excel(excelfile)

# Convert 'Date' to Day of Year (DOY) to help the model capture seasonality
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df['DOY'] = df['Date'].dt.dayofyear
else:
    print("Warning: 'Date' column not found. DOY feature skipped.")

# Calculate NDVI (Standardized Difference Vegetation Index)
if 'B08' in df.columns and 'B04' in df.columns:
    df['NDVI'] = (df['B08'] - df['B04']) / (df['B08'] + df['B04'] + 1e-8)

# Select valid spectral bands and engineered features
# We exclude B01 (Aerosols), B09 (Water Vapor), and B10 (Cirrus) as they introduce noise
noise_bands = ['B01', 'B09', 'B10']
input_cols = [c for c in df.columns if (c.startswith('B') and c not in noise_bands) or c in ['NDVI', 'DOY']]
target_col = 'GPP_DT_uStar'

X_raw = df[input_cols]
y_raw = df[[target_col]] 

### 3. DATA SCALING ###

# Neural Networks perform significantly better with normalized data (0 to 1 range)
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_norm = scaler_x.fit_transform(X_raw)
y_norm = scaler_y.fit_transform(y_raw).ravel()

### 4. DATA SPLITTING ###

# Use quantiles for 'stratification' to ensure Train/Val/Test have similar target distributions
df['strat_group'] = pd.qcut(df[target_col], q=5, labels=False)

# Split: 70% Training, 30% Temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X_norm, y_norm, train_size=0.7, random_state=42, stratify=df['strat_group']
)
# Split Temporary: 15% Validation (for selection) and 15% Test (final check)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, train_size=0.5, random_state=42
)

### 5. OPTIMIZATION LOOP ###

best_val_r2 = -np.inf
best_model = None

print(f"Starting optimization loop (Target R2: {target_r2_val}, Max attempts: {max_attempts})...")

for attempt in range(1, max_attempts + 1):
    # Changing the random_state here changes the initial weights of the neurons
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        random_state=attempt
    )

    try:
        model.fit(X_train, y_train)
        
        # Monitor performance using the Validation Set
        y_pred_val = model.predict(X_val)
        cur_r2_val = r2_score(y_val, y_pred_val)
        
        if cur_r2_val > best_val_r2:
            best_val_r2 = cur_r2_val
            best_model = model
            print(f"Attempt {attempt}: New best Val R2 = {best_val_r2:.4f}")

        # Break loop if the model meets the quality threshold
        if best_val_r2 >= target_r2_val:
            break
    except Exception:
        continue

### 6. FINAL EVALUATION & ARTIFACT EXPORT ###

if best_model is not None:
    # Final evaluation on the UNSEEN Test Set
    y_pred_test_norm = best_model.predict(X_test)
    
    # Denormalize values to original units (gC/m²/day)
    y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_real = scaler_y.inverse_transform(y_pred_test_norm.reshape(-1, 1)).flatten()
    
    # Compute Final Statistics
    test_r2 = r2_score(y_test_real, y_pred_real)
    test_rmse = rmse(y_test_real, y_pred_real)
    test_rrmse = relative_root_mean_squared_error(y_test_real, y_pred_real)
    test_rae = relative_absolute_error(y_test_real, y_pred_real)

    # Save trained objects for future inference scripts
    joblib.dump(best_model, os.path.join(output_folder, 'gpp_model.pkl'))
    joblib.dump(scaler_x, os.path.join(output_folder, 'scaler_x.pkl'))
    joblib.dump(scaler_y, os.path.join(output_folder, 'scaler_y.pkl'))
    joblib.dump(input_cols, os.path.join(output_folder, 'features_list.pkl'))
    
    # Export Results to Excel
    excel_path = os.path.join(output_folder, f"{filename_ref}_BEST_GPP_Detailed.xlsx")
    
    df_predictions = pd.DataFrame({
        'Observed_GPP': y_test_real,
        'Predicted_GPP': y_pred_real
    })
    
    df_metrics = pd.DataFrame([
        {'Metric': 'Validation R-Squared (R2)', 'Value': best_val_r2},
        {'Metric': 'Test R-Squared (R2)', 'Value': test_r2},
        {'Metric': 'Test RMSE', 'Value': test_rmse},
        {'Metric': 'Test Relative RMSE (RRMSE)', 'Value': test_rrmse},
        {'Metric': 'Test Relative Absolute Error (RAE)', 'Value': test_rae}
    ])

    with pd.ExcelWriter(excel_path) as writer:
        df_predictions.to_excel(writer, sheet_name='PREDICTIONS', index=False)
        df_metrics.to_excel(writer, sheet_name='METRICS', index=False)

    print("-" * 30)
    print("OPTIMIZATION FINISHED.")
    print(f"Final Model Selected | Val R2: {best_val_r2:.4f} | Test R2: {test_r2:.4f}")
    print(f"Results saved in: {output_folder}")
else:
    print("No model reached convergence or was found.")