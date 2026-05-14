# -*- coding: utf-8 -*-
"""
Created Mon May 04 17:10:46 2026


Project: ScaleAgData - WP5 - Grassland Rilab
@author: paolo cosmo silvestro

Description:
"""

import numpy as np
import rasterio
import joblib
import json
import math
from datetime import datetime

def run_gpp_inference(model_path, metadata_path, input_tif, output_tif, acquisition_date):
    # 1. Load the Random Forest model and metadata
    # Verified model: best_model_GPP_Minimal_RandomForest.pkl[cite: 1]
    model = joblib.load(model_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    required_features = metadata["features_required"] # ["FCOVER", "DOY_sin", "DOY_cos"][cite: 2]
    
    # 2. Calculate temporal features (Cyclical Day of Year)
    # Date: 20260404 -> Day 94
    date_obj = datetime.strptime(acquisition_date, "%Y%m%d")
    doy = date_obj.timetuple().tm_yday
    
    doy_sin = math.sin(2 * math.pi * doy / 365.0)
    doy_cos = math.cos(2 * math.pi * doy / 365.0)
    
    # 3. Read the FAPAR (FCOVER) GeoTIFF
    with rasterio.open(input_tif) as src:
        fcover_data = src.read(1)  # Model uses FCOVER as primary feature[cite: 2]
        profile = src.profile.copy()
        nodata_value = src.nodata
        
        # Create a mask for valid pixels (land/vegetation)
        mask = fcover_data != nodata_value
        fcover_valid = fcover_data[mask]
        
    # 4. Construct the feature matrix[cite: 2]
    # Matrix must follow the order: FCOVER, DOY_sin, DOY_cos
    num_valid = fcover_valid.shape[0]
    X = np.zeros((num_valid, len(required_features)))
    X[:, 0] = fcover_valid
    X[:, 1] = doy_sin
    X[:, 2] = doy_cos
    
    # 5. Perform Inference
    print(f"Running inference on {num_valid} pixels...")
    gpp_predictions = model.predict(X)
    
    # 6. Reconstruct the georeferenced raster
    # Initialize with NoData (-9999)
    gpp_raster = np.full(fcover_data.shape, -9999.0, dtype=np.float32)
    gpp_raster[mask] = gpp_predictions
    
    # 7. Update profile and save output
    profile.update({
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": -9999.0,
        "count": 1
    })
    
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(gpp_raster, 1)
        # Spatial metadata (CRS and Transform) are preserved from input[cite: 2]
    
    print(f"Success! GPP map saved as: {output_tif}")

if __name__ == "__main__":
    # Configuration
    MODEL = "best_model_GPP_Minimal_RandomForest.pkl"
    METADATA = "inference_metadata.json"
    INPUT_TIF = "FCOVER_S2A_MSIL2A_20260404T105711_IFAPAaoiTest.tif"
    OUTPUT_TIF = "GPP_IFAPA_20260404.tif"
    DATE_STR = "20260404"
    
    run_gpp_inference(MODEL, METADATA, INPUT_TIF, OUTPUT_TIF, DATE_STR)
