# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:07:27 2026

@author: pcss
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 4 17:26:21 2025
Modified on 15/04/2026 11:10
@project ScaleAgData
@author: Paolo Cosmo Silvestro
"""

"""
This script trains a Multilayer Perceptron (MLP) to predict EC-GPP
using Sentinel-1 features, with an optional Day Of Year (DOY) input.

Input variables can be:
- If use_DOY = True:
    1. Day Of Year (DOY), derived from the acquisition date (ac_date)
    2. vh_av (normalized)
    3. vv_av (normalized)
- If use_DOY = False:
    1. vh_av (normalized)
    2. vv_av (normalized)

In this version:
- Input features are normalized with StandardScaler.
- The target variable (EC - GPP) is normalized with MinMaxScaler in [0, 1].
- The MLP output layer uses a ReLU activation, so predictions are >= 0.

Main features of this script:
- Loads an Excel file containing Sentinel-1 and EC-GPP data.
- Filters rows where required columns are not missing.
- Optionally creates and uses the DOY feature from the acquisition date.
- Splits the cleaned dataset into:
    - 70% training
    - 15% validation
    - 15% test
  using a time-ordered, cyclic split so that all sets are well distributed
  along the whole time series.
- Normalizes inputs and target using only the training set statistics.
- Trains an MLP with a user-defined number of hidden layers.
- Uses EarlyStopping and ModelCheckpoint to keep the best model.
- Evaluates the model on the test set (R², RMSE, MAE) using EC-GPP in the original scale.
- Saves:
    - A CSV with the full original rows used in the validation set.
    - A CSV with the full original rows used in the test set.
    - A text file logging:
        - user parameters (model configuration and paths)
        - whether DOY was used or not
        - dataset sizes
        - number of epochs actually run and best val_loss
        - evaluation metrics.
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 4 17:26:21 2025
Modified on 21/04/2026 15:00
@project ScaleAgData
@author: Paolo Cosmo Silvestro & AI Assistant
"""

"""
This script trains a Multilayer Perceptron (MLP) to predict EC-GPP using Sentinel-1 
features (VH, VV) and Day Of Year (DOY). It features an outer optimization loop 
that reruns the training multiple times to find the best random weight initialization 
based on Validation R² performance.
"""

import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Helper Functions for Statistical Metrics ---

def relative_root_mean_squared_error(true, pred):
    """
    Calculates the RRMSE by normalizing the RMSE by the mean of the observed values.
    """
    n = len(true)
    num = np.sum(np.square(true - pred)) / n
    return np.sqrt(num) / np.mean(true) if np.mean(true) != 0 else 0

def relative_absolute_error(true, pred):
    """
    Calculates the RAE: the ratio of the absolute error relative to the 
    mean of the true values.
    """
    true_mean = np.mean(true)
    num = np.sum(np.abs(true - pred))
    den = np.sum(np.abs(true - true_mean))
    return num / den if den != 0 else 0

# ======================================================================
# User Configuration & Parameters
# ======================================================================

# Input data source
input_excel = "IFAPA_merged_S1_IFAPAData2024_input.xlsx"

# MLP Architecture and Training Hyperparameters
num_hidden_layers = 2
epochs = 1000
batch_size = 16
use_DOY = True

# Outer Optimization Loop Settings
num_iterations = 10000      # Max number of training attempts
patience_iterations = 1000   # Stop loop if Validation R2 doesn't improve for X iterations

# --- Dynamic File Naming Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Generate a unique timestamp for this session
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define output paths using the unique timestamp
validation_csv_path = os.path.join(script_dir, f"validation_set_{timestamp_str}.csv")
test_csv_path       = os.path.join(script_dir, f"test_set_{timestamp_str}.csv")
results_log_path    = os.path.join(script_dir, f"ECGPP_results_{timestamp_str}.txt")
best_model_path     = os.path.join(script_dir, f"ECGPP_best_model_{timestamp_str}.keras")
output_excel_path   = os.path.join(script_dir, f"ECGPP_Final_Predictions_{timestamp_str}.xlsx")
scaler_X_path       = os.path.join(script_dir, f"scaler_X_{timestamp_str}.pkl")
scaler_y_path       = os.path.join(script_dir, f"scaler_y_{timestamp_str}.pkl")
temp_model_path     = os.path.join(script_dir, f"temp_iteration_{timestamp_str}.keras")

# ======================================================================
# 1. Data Loading and Pre-processing
# ======================================================================

# Load Excel dataset
df_original = pd.read_excel(input_excel)
df_original["ac_date"] = pd.to_datetime(df_original["ac_date"])

# Filter rows: keep only records where all required features and target exist
required_cols = ["ac_date", "vh_av", "vv_av", "EC - GPP"]
mask_valid = df_original[required_cols].notna().all(axis=1)
df_clean = df_original.loc[mask_valid].copy()
df_clean["orig_index"] = df_clean.index
df_clean = df_clean.sort_values("ac_date").reset_index(drop=True)

# Feature Engineering: Extract Day of Year
df_clean["DOY"] = df_clean["ac_date"].dt.dayofyear
feature_cols = ["DOY", "vh_av", "vv_av"] if use_DOY else ["vh_av", "vv_av"]

X_all = df_clean[feature_cols].values
y_all = df_clean["EC - GPP"].values

# Dataset Splitting Strategy: Cyclic Split
# 70% Train (0-13), 15% Val (14-16), 15% Test (17-19) in blocks of 20
indices = np.arange(X_all.shape[0])
train_indices, val_indices, test_indices = [], [], []
for i in indices:
    m = i % 20
    if m < 14: train_indices.append(i)
    elif m < 17: val_indices.append(i)
    else: test_indices.append(i)

X_train, y_train = X_all[train_indices], y_all[train_indices]
X_val, y_val     = X_all[val_indices], y_all[val_indices]
X_test, y_test   = X_all[test_indices], y_all[test_indices]

# Keep original scale targets for evaluation
y_val_orig, y_test_orig = y_val.copy(), y_test.copy()

# Export split subsets for reproducibility/tracking
df_original.loc[df_clean.loc[val_indices, "orig_index"].values].to_csv(validation_csv_path, index=False)
df_original.loc[df_clean.loc[test_indices, "orig_index"].values].to_csv(test_csv_path, index=False)

# --- Normalization ---
# Standardize inputs (Zero mean, unit variance)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled   = scaler_X.transform(X_val)
X_test_scaled  = scaler_X.transform(X_test)

# Scale target to [0, 1] range
scaler_y = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_val_scaled   = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

# Save scalers for future inference
joblib.dump(scaler_X, scaler_X_path)
joblib.dump(scaler_y, scaler_y_path)

# ======================================================================
# 2. MLP Training Loop (Initialization Optimization)
# ======================================================================

best_r2_val = -np.inf
best_iteration_info = {}
no_improvement_counter = 0

def build_mlp_model(input_dim, num_layers):
    """Factory function to build a Keras MLP model."""
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    for _ in range(num_layers - 1):
        model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='relu')) # Output layer with ReLU for non-negative GPP
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

print(f"--- Training Session Started: {timestamp_str} ---")
print(f"Configuration: Max Iterations={num_iterations}, Patience={patience_iterations}")

for run in range(1, num_iterations + 1):
    # Clear session to prevent memory leaks over 1000 iterations
    tf.keras.backend.clear_session()
    
    model = build_mlp_model(X_train_scaled.shape[1], num_hidden_layers)
    
    # Internal Keras EarlyStopping (stops epoch training if val_loss plateaus)
    early_stop_keras = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath=temp_model_path, monitor='val_loss', save_best_only=True, verbose=0)

    # Train model
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=epochs, batch_size=batch_size,
        callbacks=[early_stop_keras, checkpoint], verbose=0
    )

    # Performance check on Validation Set using R²
    if os.path.exists(temp_model_path):
        temp_model = load_model(temp_model_path)
        y_val_pred_scaled = temp_model.predict(X_val_scaled, verbose=0).ravel()
        y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).ravel()
        current_r2_val = r2_score(y_val_orig, y_val_pred)
        
        # Determine if this iteration is the best so far
        if current_r2_val > best_r2_val:
            best_r2_val = current_r2_val
            temp_model.save(best_model_path) # Permanently save the best weight init
            best_iteration_info = {
                'iteration': run,
                'epochs_run': len(history.history['loss']),
                'best_val_loss': min(history.history['val_loss']),
                'val_r2': current_r2_val
            }
            no_improvement_counter = 0
            print(f"Iter {run}: New Global Best R²_val = {best_r2_val:.4f}")
        else:
            no_improvement_counter += 1
            if run % 5 == 0:
                print(f"Iter {run}: No improvement ({no_improvement_counter}/{patience_iterations})")

    # Early Exit for the outer loop
    if no_improvement_counter >= patience_iterations:
        print(f"\n--- STOP: No improvement for {patience_iterations} iterations. ---")
        break

# ======================================================================
# 3. Final Evaluation and Reporting
# ======================================================================

if os.path.exists(best_model_path):
    # Load the absolute best version of the model found during iterations
    final_model = load_model(best_model_path)
    y_pred_scaled = final_model.predict(X_test_scaled, verbose=0).ravel()
    y_pred_scaled = np.clip(y_pred_scaled, 0.0, 1.0)
    y_pred_test = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # Calculate Test Metrics
    r2_test = r2_score(y_test_orig, y_pred_test)
    rmse    = mean_squared_error(y_test_orig, y_pred_test, squared=False)
    rrmse   = relative_root_mean_squared_error(y_test_orig, y_pred_test)
    rae     = relative_absolute_error(y_test_orig, y_pred_test)
    r2_val  = best_iteration_info['val_r2']

    # --- Excel Output Generation ---
    # Sheet 1: Direct comparison of observed vs predicted GPP
    df_predictions = pd.DataFrame({'GPP_observed': y_test_orig, 'GPP_predicted': y_pred_test})
    # Sheet 2: Summary of statistical performance
    df_metrics = pd.DataFrame({
        'Metric': ['R2 Validation', 'R2 Test', 'RMSE', 'RRMSE', 'RAE'],
        'Value': [r2_val, r2_test, rmse, rrmse, rae]
    })

    with pd.ExcelWriter(output_excel_path) as writer:
        df_predictions.to_excel(writer, sheet_name='PREDICTIONS', index=False)
        df_metrics.to_excel(writer, sheet_name='METRICS', index=False)

    # --- Text Log Generation ---
    now_full = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(results_log_path, "w", encoding="utf-8") as f:
        f.write(f"Session Timestamp ID: {timestamp_str}\n")
        f.write(f"Completion Date: {now_full}\n")
        f.write(f"Total Loop Iterations: {run}\n")
        f.write(f"Best iteration index: {best_iteration_info['iteration']}\n\n")
        f.write(f"--- MODEL PERFORMANCE ---\n")
        f.write(f"R2 Validation: {r2_val:.6f}\n")
        f.write(f"R2 Test:       {r2_test:.6f}\n")
        f.write(f"RMSE:          {rmse:.6f}\n")
        f.write(f"RRMSE:         {rrmse:.6f}\n")
        f.write(f"RAE:           {rae:.6f}\n")

    # Clean up the iteration-specific model file
    if os.path.exists(temp_model_path): os.remove(temp_model_path)
    
    print(f"\n--- PROCESS COMPLETED SUCCESSFULLY ---")
    print(f"Final Model: {best_model_path}")
    print(f"Excel Report: {output_excel_path}")
    print(f"Log File:     {results_log_path}")
else:
    print("Error: The best model was not saved correctly during the training loop.")
