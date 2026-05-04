
# -*- coding: utf-8 -*-
"""
Updated on Fri Apr 24 2026
@project: ScaleAgData - WP5 Rilab Grassland
@task: GPP Estimation - Minimal Model (FCOVER + DOY)
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

# Updated Features: Only FCOVER and the cyclic components of DOY
FEATURES = ['FCOVER', 'DOY_sin', 'DOY_cos']
TARGET = 'GPP'
SEED = 42 

# 1. Create Output Directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f'GPP_Minimal_FCOVER_DOY_{timestamp}'
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

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    # --- MINIMAL FEATURE ENGINEERING (DOY ONLY) ---
    print("Applying Minimal Feature Engineering (DOY Cyclic Encoding)...")
    
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if date_col:
        print(f"Date column detected: '{date_col}'.")
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        df = df.dropna(subset=[date_col])
        
        doy = df[date_col].dt.dayofyear
        df['DOY_sin'] = np.sin(2 * np.pi * doy / 365.25)
        df['DOY_cos'] = np.cos(2 * np.pi * doy / 365.25)
    else:
        print("Error: No date column found. DOY features are mandatory for this version.")
        return

    # Ensure required columns are present
    df = df[FEATURES + [TARGET]].dropna()
    X, y = df[FEATURES], df[TARGET]

    # --- STRATIFIED SPLIT ---
    y_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y_bins
    )
    
    y_temp_bins = pd.qcut(y_temp, q=5, labels=False, duplicates='drop')
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp_bins
    )

    print(f"Dataset Split Completed. Samples -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print("-" * 75)

    best_rf = {'val_r2': -np.inf, 'model': None}
    best_xgb = {'val_r2': -np.inf, 'model': None}

    print(f"Starting {ITERATIONS} iterations...")

    for i in range(1, ITERATIONS + 1):
        # --- RANDOM FOREST ---
        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=i)
        rf.fit(X_train, y_train)
        rf_val_r2 = r2_score(y_val, rf.predict(X_val))
        
        if rf_val_r2 > best_rf['val_r2']:
            best_rf.update({'val_r2': rf_val_r2, 'model': rf})

        # --- XGBOOST ---
        xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=i, verbosity=0)
        xgb.fit(X_train, y_train)
        xgb_val_r2 = r2_score(y_val, xgb.predict(X_val))

        if xgb_val_r2 > best_xgb['val_r2']:
            best_xgb.update({'val_r2': xgb_val_r2, 'model': xgb})

        if i % 100 == 0:
            print(f"Iteration {i}/{ITERATIONS} | Best Val R2 -> RF: {best_rf['val_r2']:.4f} | XGB: {best_xgb['val_r2']:.4f}")

    # --- FINAL EVALUATION ---
    winner_name = 'Random Forest' if best_rf['val_r2'] > best_xgb['val_r2'] else 'XGBoost'
    winner_model = best_rf['model'] if winner_name == 'Random Forest' else best_xgb['model']
    
    y_pred_test = winner_model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_rrmse, test_rae = calculate_relative_metrics(y_test, y_pred_test)

    # --- EXCEL REPORT ---
    report_path = os.path.join(output_folder, 'GPP_Minimal_Report.xlsx')
    with pd.ExcelWriter(report_path) as writer:
        pd.DataFrame({'Measured_GPP': y_test, 'Estimated_GPP': y_pred_test}).to_excel(writer, sheet_name='Test_Data', index=False)
        pd.DataFrame({
            'Metric': ['Best Algorithm', 'Final Test R2', 'Test RMSE', 'RRMSE (%)', 'RAE (%)', 'Validation R2 (Best)'],
            'Value': [winner_name, test_r2, test_rmse, test_rrmse, test_rae, max(best_rf['val_r2'], best_xgb['val_r2'])]
        }).to_excel(writer, sheet_name='Performance_Summary', index=False)

    # --- VISUALIZATION ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].scatter(y_test, y_pred_test, alpha=0.6, edgecolors='k', color='#2c3e50')
    axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[0].set_title(f'Performance: {winner_name}\n(Test R2: {test_r2:.4f})')
    axes[0].set_xlabel('Measured GPP')
    axes[0].set_ylabel('Estimated GPP')
    
    # Aggregated Importance for DOY
    feat_imp_dict = dict(zip(FEATURES, winner_model.feature_importances_))
    doy_total_importance = feat_imp_dict.get('DOY_sin', 0) + feat_imp_dict.get('DOY_cos', 0)
    
    plot_data = [{'Feature': 'FCOVER', 'Importance': feat_imp_dict.get('FCOVER', 0)},
                 {'Feature': 'DOY (Seasonal)', 'Importance': doy_total_importance}]
    
    df_plot = pd.DataFrame(plot_data).sort_values(by='Importance', ascending=False)
    sns.barplot(x='Importance', y='Feature', data=df_plot, ax=axes[1], palette='viridis', hue='Feature', legend=False)
    axes[1].set_title(f'Feature Importance ({winner_name})')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'final_performance_plots.png'), dpi=300)
    plt.show()

    # --- INFERENCE BUNDLE ---
    inference_info = {
        'project': 'ScaleAgData - WP5 Rilab Grassland',
        'algorithm_winner': winner_name,
        'features_required': FEATURES,
        'target_variable': TARGET,
        'performance_metrics': {'test_r2': float(test_r2), 'test_rmse': float(test_rmse)}
    }

    model_filename = f'best_model_GPP_Minimal_{winner_name.replace(" ", "")}.pkl'
    joblib.dump(winner_model, os.path.join(output_folder, model_filename))

    with open(os.path.join(output_folder, 'inference_metadata.json'), 'w') as f:
        json.dump(inference_info, f, indent=4)

    print(f"\n[INFO] Inference bundle saved successfully in {output_folder}")

if __name__ == "__main__":
    main()