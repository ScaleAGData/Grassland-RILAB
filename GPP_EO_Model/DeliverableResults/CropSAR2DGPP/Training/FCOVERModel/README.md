# CropSAR2D FCOVER Model Training

## Overview
This repository contains the scripts and configurations for training and validating models to estimate the **Fraction of Vegetation Cover (FCOVER)** using **CropSAR2D** data. 

As part of the **ScaleAgData (WP5)** initiative for the SpainSite Grassland RILAB, this component focuses on optimizing the biophysical inputs required for Gross Primary Production (GPP) modeling. By using gap-filled CropSAR2D data (optical and radar fusion), the pipeline ensures continuous, high-resolution FCOVER estimations regardless of cloud cover.

## Directory Structure
- `Training/`: Scripts for the iterative training and optimization of machine learning regressors.
- `FCOVERModel/`: Specific logic and hyperparameter configurations for the Fraction of Vegetation Cover model.

## Methodology
The training pipeline utilizes a multi-model approach to ensure the highest retrieval accuracy:
1.  **Algorithms:** Supports **Random Forest (RF)** and **XGBoost (XGB)** architectures.
2.  **Iterative Optimization:** Runs multiple training cycles (up to 1,000 iterations) to identify the most stable model based on Validation $R^2$.
3.  **Data Integration:** Uses fused Sentinel-1 (SAR) and Sentinel-2 (Optical) data processed through the CropSAR algorithm to provide gap-free time series.

## Input Requirements
The training script expects a standardized dataset (CSV) containing:
*   **Target:** In-situ or reference FCOVER measurements.
*   **Features:** 
    *   Spectral indices (e.g., NDVI).
    *   SAR-derived features from CropSAR2D.
    *   Cyclical temporal encoding (Sine/Cosine of the Day of Year).

## Outputs
Successful execution of the training pipeline generates a timestamped folder containing:
*   **`best_model_FCOVER.pkl`**: The serialized, top-performing model for deployment.
*   **`performance_metrics.xlsx`**: Detailed statistics including $R^2$, RMSE, and RRMSE.
*   **Visualizations**: Feature importance charts and scatter plots (Observed vs. Predicted).

## Installation & Usage
Ensure you have activated the project's Conda environment:

```bash
conda activate gpp_env
```

To initiate the FCOVER training process:
```bash
python train_fcover_model.py --input path/to/your_data.csv
```
