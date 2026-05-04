# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:36:29 2026

@author: pcss
"""

# -*- coding: utf-8 -*-
"""
Updated on Thu Apr 23 17:10:19 2026
Created on Mon Mar  9 2026
@project ScaleAgData - WP5 Rilab Grassland
Description: Training pipeline for Random Forest and XGBoost to estimate GPP.
Saves the best independent models and exports comprehensive metrics to Excel.
@author: Paolo Cosmo Silvestro

ESTIMATION OF GROSS PRIMARY PRODUCTION (GPP) - WP5 Rilab Grassland
==================================================================
Project: ScaleAgData initiative
Task: Machine Learning Pipeline for GPP Regression

DESCRIPTION:
This script implements a high-iteration training pipeline to estimate GPP 
using three primary biophysical variables. It compares two state-of-the-art 
ensemble algorithms—Random Forest (RF) and XGBoost (XGB)—to identify the 
most accurate model for grassland productivity monitoring.

ALGORITHM WORKFLOW:
1. Data Cleaning: Loads in-situ/satellite data and filters relevant features.
2. Iterative Optimization: Executes up to 1000 random data splits to find the 
   weight initialization that minimizes error and maximizes stability.
3. Independent Tracking: Monitors R2 and RMSE independently for both RF and XGB, 
   saving the "Best-in-Class" version of each model.
4. Relative Metrics: Calculates advanced error statistics including RRMSE 
   (Relative Root Mean Squared Error) and RAE (Relative Absolute Error).

INPUTS:
- Features: FCOVER, NDVI Average, FAPAR Average.
- Target: GPP (Gross Primary Production).

OUTPUTS:
- Model Artifacts: 'best_model_RandomForest.pkl' and 'best_model_XGBoost.pkl'.
- Excel Report: Detailed predictions and statistical metrics (R2, RMSE, RRMSE, RAE).
- Visualizations: Scatter plots of measured vs. predicted GPP and Feature Importance charts.
"""

# -*- coding: utf-8 -*-
"""
Updated on Thu Apr 23 2026
@project: ScaleAgData - WP5 Rilab Grassland
@task: GPP Estimation Retrieval from Sentinel-2
@description: Optimized pipeline with Feature Engineering and European Date Parsing
@author: Paolo Cosmo Silvestro
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import json

# --- CONFIGURATION ---
FILE_PATH = r'C:/Users/pcss/OneDrive - Indra/Projects/ScaleAgData/RI LAB grassaland/RiLabGrassland_Shared/Data/Tests/SpainSite/ANNmodelGPP_Cropsar2D/Utils/Input_IFAPA_GPP_cropsar2D_3vars.csv'
ITERATIONS = 1000
# Added the new engineered features to the list for the model to use
FEATURES = ['FCOVER', 'NDVI Average', 'FAPAR Average', 'NDVI_FAPAR_Interact', 'DOY_sin', 'DOY_cos']
TARGET = 'GPP'
SEED = 42 

# 1. Create Output Directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f'GPP_Optimized_Estimation_{timestamp}'
os.makedirs(output_folder, exist_ok=True)

def calculate_relative_metrics(y_true, y_pred):
    """Calculates Relative Root Mean Squared Error (RRMSE) and Relative Absolute Error (RAE)."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rrmse = (rmse / np.mean(y_true)) * 100 
    rae = (np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))) * 100
    return rrmse, rae

def main():
    try:
        df = pd.read_csv(FILE_PATH)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Clean column names (remove leading/trailing spaces)
    df.columns = [c.strip() for c in df.columns]

    # --- FEATURE ENGINEERING ---
    print("Applying Feature Engineering...")
    
    # 1. Interaction Feature: Helps model learn Light Use Efficiency logic
    df['NDVI_FAPAR_Interact'] = df['NDVI Average'] * df['FAPAR Average']
    
    # 2. Cyclic Seasonal Encoding (DOY)
    # Automatically finds the date column and handles European format DD/MM/YYYY
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if date_col:
        print(f"Date column detected: '{date_col}'. Parsing European format...")
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        
        # Remove rows where the date could not be parsed
        df = df.dropna(subset=[date_col])
        
        doy = df[date_col].dt.dayofyear
        df['DOY_sin'] = np.sin(2 * np.pi * doy / 365.25)
        df['DOY_cos'] = np.cos(2 * np.pi * doy / 365.25)
    else:
        print("Warning: No date column found. Seasonal features initialized to 0.")
        df['DOY_sin'] = 0
        df['DOY_cos'] = 0

    # Ensure all required features and target are present and clean
    df = df[FEATURES + [TARGET]].dropna()
    X, y = df[FEATURES], df[TARGET]

    # --- STRATIFIED SPLIT (Uniform GPP Distribution) ---
    # We bin the GPP into 5 levels to ensure the split is representative
    y_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y_bins
    )
    
    # Split the temporary set equally into Validation and Test
    y_temp_bins = pd.qcut(y_temp, q=5, labels=False, duplicates='drop')
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp_bins
    )

    print(f"Dataset Split Completed (SEED {SEED})")
    print(f"Samples -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print("-" * 75)

    # Initialize tracking
    best_rf = {'val_r2': -np.inf, 'model': None}
    best_xgb = {'val_r2': -np.inf, 'model': None}

    print(f"Starting {ITERATIONS} iterations to optimize stochastic weights...")

    for i in range(1, ITERATIONS + 1):
        # --- RANDOM FOREST ---
        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=i)
        rf.fit(X_train, y_train)
        rf_val_r2 = r2_score(y_val, rf.predict(X_val))
        
        if rf_val_r2 > best_rf['val_r2']:
            best_rf.update({'val_r2': rf_val_r2, 'model': rf})
            joblib.dump(rf, os.path.join(output_folder, 'best_model_RandomForest.pkl'))

        # --- XGBOOST (Hyper-parameters adjusted for better precision) ---
        xgb = XGBRegressor(
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=6, 
            subsample=0.8,
            random_state=i, 
            verbosity=0
        )
        xgb.fit(X_train, y_train)
        xgb_val_r2 = r2_score(y_val, xgb.predict(X_val))

        if xgb_val_r2 > best_xgb['val_r2']:
            best_xgb.update({'val_r2': xgb_val_r2, 'model': xgb})
            joblib.dump(xgb, os.path.join(output_folder, 'best_model_XGBoost.pkl'))

        if i % 100 == 0:
            print(f"Iteration {i}/{ITERATIONS} | Best Val R2 -> RF: {best_rf['val_r2']:.4f} | XGB: {best_xgb['val_r2']:.4f}")

    # --- FINAL EVALUATION ---
    winner_name = 'Random Forest' if best_rf['val_r2'] > best_xgb['val_r2'] else 'XGBoost'
    winner_model = best_rf['model'] if winner_name == 'Random Forest' else best_xgb['model']
    
    # Final check on the unseen Test Set
    y_pred_test = winner_model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_rrmse, test_rae = calculate_relative_metrics(y_test, y_pred_test)

    # --- EXCEL REPORT ---
    report_path = os.path.join(output_folder, 'GPP_Estimation_Report.xlsx')
    with pd.ExcelWriter(report_path) as writer:
        pd.DataFrame({'Measured_GPP': y_test, 'Estimated_GPP': y_pred_test}).to_excel(writer, sheet_name='Test_Data', index=False)
        pd.DataFrame({
            'Metric': ['Best Algorithm', 'Final Test R2', 'Test RMSE', 'RRMSE (%)', 'RAE (%)', 'Validation R2 (Best)'],
            'Value': [winner_name, test_r2, test_rmse, test_rrmse, test_rae, max(best_rf['val_r2'], best_xgb['val_r2'])]
        }).to_excel(writer, sheet_name='Performance_Summary', index=False)

    # --- VISUALIZATION ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter Plot
    axes[0].scatter(y_test, y_pred_test, alpha=0.6, edgecolors='k', color='#2c3e50')
    axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[0].set_title(f'Estimation Performance: {winner_name}\n(Test R2: {test_r2:.4f})')
    axes[0].set_xlabel('In-Situ Measured GPP')
    axes[0].set_ylabel('Satellite Estimated GPP')
    
    # --- GESTIONE FEATURE IMPORTANCE AGGREGATA ---
    # Creiamo un dizionario temporaneo per mappare importanze e nomi
    feat_imp_dict = dict(zip(FEATURES, winner_model.feature_importances_))
    
    # Sommiamo le importanze di sin e cos in una nuova voce 'DOY'
    doy_total_importance = feat_imp_dict.get('DOY_sin', 0) + feat_imp_dict.get('DOY_cos', 0)
    
    # Creiamo una nuova lista di importanze "pulita"
    plot_data = []
    for feat, imp in feat_imp_dict.items():
        if feat not in ['DOY_sin', 'DOY_cos']:
            plot_data.append({'Feature': feat, 'Importance': imp})
    
    # Aggiungiamo il DOY aggregato
    plot_data.append({'Feature': 'DOY (Seasonal)', 'Importance': doy_total_importance})
    
    # Convertiamo in DataFrame e ordiniamo per importanza
    df_plot = pd.DataFrame(plot_data).sort_values(by='Importance', ascending=False)
    
    # Feature Importance Chart
    sns.barplot(x='Importance', y='Feature', data=df_plot, ax=axes[1], palette='viridis')
    axes[1].set_title(f'Feature Importance ({winner_name})')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'final_performance_plots.png'), dpi=300)
    plt.show()

    # --- INFERENCE BUNDLE PREPARATION ---

    # Define metadata to ensure consistent feature ordering during prediction
    # Note: 'i' represents the last iteration number from the loop
    inference_info = {
        'project': 'ScaleAgData - WP5 Rilab Grassland',
        'algorithm_winner': winner_name,
        'features_required': FEATURES,
        'target_variable': TARGET,
        'training_timestamp': timestamp,
        'best_iteration_index': i, 
        'performance_metrics': {
            'test_r2': float(test_r2),
            'test_rmse': float(test_rmse),
            'test_rrmse': float(test_rrmse),
            'test_rae': float(test_rae)
        }
    }

    # 1. Save the model artifact
    # FIXED: removed ['model'] because winner_model is already the estimator
    model_filename = f'best_model_GPP_{winner_name.replace(" ", "")}.pkl'
    joblib.dump(winner_model, os.path.join(output_folder, model_filename))

    # 2. Save metadata as JSON (Essential for future feature alignment)
    meta_filename = 'inference_metadata.json'
    with open(os.path.join(output_folder, meta_filename), 'w') as f:
        json.dump(inference_info, f, indent=4)

    print("\n" + "="*75)
    print(f"ESTIMATION SUCCESSFUL ({winner_name})")
    print(f"Final Test R2:   {test_r2:.4f}")
    print(f"Final Test RMSE: {test_rmse:.4f}")
    print("-" * 75)
    print(f"[INFO] Inference bundle saved:")
    print(f" - Model file:    {model_filename}")
    print(f" - Metadata file: {meta_filename}")
    print(f" - Results folder: {output_folder}")
    print("="*75)


if __name__ == "__main__":
    main()