# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:06:56 2026

@author: Paolo Cosmo Silvestro

@project: ScaleAgData


GPP INFERENCE PIPELINE - ScaleAgData Initiative
===============================================
This script performs pixel-wise inference to generate a Gross Primary Production (GPP) 
map using a trained Multilayer Perceptron (MLP) neural network.

PROJECT OVERVIEW:
Developed under the ScaleAgData initiative (Work Package 5), this tool estimates 
EC-GPP by integrating Sentinel-2 L2A satellite imagery with temporal features.

INPUTS:
- GeoTIFF: A 10-band preprocessed Sentinel-2 image (B02, B03, B04, B05, B06, 
  B07, B08, B8A, B11, B12).
- Model Artifacts: Trained MLP model (.pkl), Scalers (.pkl), and Feature List (.pkl).
- Date: A user-defined date used to calculate the Day of Year (DOY).

PROCESS:
1. Feature Engineering: Calculates NDVI from B04 (Red) and B08 (NIR).
2. Normalization: Applies MinMaxScaler transformations used during training.
3. Masking: Detects NoData/masked pixels in the input imagery to ensure they 
   remain null in the output.
4. Spatial Prediction: Runs the MLP regressor on every valid pixel.

OUTPUT:
- A single-band 32-bit Float GeoTIFF where pixel values represent GPP 
  in gC/m^2/day. Spatial metadata (CRS and Transform) is preserved from the input.
"""


import joblib
import rasterio
import numpy as np
import pandas as pd
from datetime import datetime
import os

# --- 1. Configuration ---
# Paths to files generated during training
MODEL_PATH = 'gpp_model.pkl'
SCALER_X_PATH = 'scaler_x.pkl'
SCALER_Y_PATH = 'scaler_y.pkl'
FEATURES_LIST_PATH = 'features_list.pkl'

# Input Sentinel-2 GeoTIFF
INPUT_GEOTIFF_PATH = 'S2A_MSIL2A_20260404T105711_N0512_R094_T30SUH_20260404T175010_S24GPPformat_Clipped.tif'

# Input Date for GPP calculation
INPUT_DATE_STR = '2026-04-20'

# Output GeoTIFF path
OUTPUT_GEOTIFF_PATH = 'GPP_Map_20260420.tif'

# Band mapping from GeoTIFF to model features
# The training script used: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
# We assume the GeoTIFF has these bands in the same order (1-indexed for rasterio).
BAND_MAPPING = {
    'B02': 1, 'B03': 2, 'B04': 3, 'B05': 4,
    'B06': 5, 'B07': 6, 'B08': 7, 'B8A': 8,
    'B11': 9, 'B12': 10
}

def main():
    print("--- GPP Map Inference Script ---")

    # --- 2. Load Model and Preprocessing Files ---
    print("Loading model and scalers...")
    try:
        model = joblib.load(MODEL_PATH)
        scaler_x = joblib.load(SCALER_X_PATH)
        scaler_y = joblib.load(SCALER_Y_PATH)
        trained_features = joblib.load(FEATURES_LIST_PATH)
    except FileNotFoundError as e:
        print(f"Error: Required training file not found: {e.filename}")
        return

    # --- 3. Process Input Date ---
    print(f"Processing input date: {INPUT_DATE_STR}...")
    try:
        input_date = datetime.strptime(INPUT_DATE_STR, '%Y-%m-%d')
        doy = input_date.timetuple().tm_yday
    except ValueError:
        print(f"Error: Invalid date format. Please use YYYY-MM-DD.")
        return

    # --- 4. Read Sentinel-2 GeoTIFF ---
    print(f"Reading input GeoTIFF: {INPUT_GEOTIFF_PATH}...")
    try:
        with rasterio.open(INPUT_GEOTIFF_PATH) as src:
            # Get metadata for the output file
            profile = src.profile
            # Check if all required bands are available
            if src.count < max(BAND_MAPPING.values()):
                print(f"Error: Input GeoTIFF does not have all required bands. "
                      f"Expected at least {max(BAND_MAPPING.values())}, got {src.count}.")
                return

            # Read bands into a dictionary of numpy arrays
            bands_data = {}
            for band_name, band_idx in BAND_MAPPING.items():
                # Read the band, applying the internal mask if present
                bands_data[band_name] = src.read(band_idx, masked=True)

            # Get the overall mask (True for valid data, False for masked/NoData)
            # This logic assumes that if *any* band is masked, the entire pixel is invalid for GPP calculation.
            overall_mask = ~bands_data['B02'].mask # Start with one band's mask, negate to get valid pixels
            for band_data in bands_data.values():
                overall_mask &= ~band_data.mask # Intersection of valid pixels across all bands

            # Store band values as standard NumPy float32 arrays (removing mask for calculations)
            # Masked pixels are filled with a dummy value (like 0), but we'll use the overall_mask to ignore them later.
            for band_name in bands_data:
                bands_data[band_name] = bands_data[band_name].filled(0).astype('float32')

    except rasterio.errors.RasterioIOError:
        print(f"Error: Could not open input GeoTIFF: {INPUT_GEOTIFF_PATH}")
        return

    height = profile['height']
    width = profile['width']
    
    # --- 5. Prepare Feature DataFrame ---
    print("Preparing feature DataFrame...")
    
    # Create NDVI feature
    denominator = bands_data['B08'] + bands_data['B04']
    # Avoid division by zero, especially in masked areas filled with 0
    with np.errstate(invalid='ignore', divide='ignore'):
        ndvi = np.where(denominator != 0, (bands_data['B08'] - bands_data['B04']) / denominator, 0)
    bands_data['NDVI'] = ndvi

    # Create DOY feature
    bands_data['DOY'] = np.full((height, width), doy, dtype='int32')

    # Convert dictionary of arrays to a structure suitable for a DataFrame
    # Reshape arrays to (height * width,) to create columns
    data_for_df = {}
    for feature_name in trained_features:
        if feature_name in bands_data:
            data_for_df[feature_name] = bands_data[feature_name].ravel()
        else:
            print(f"Error: Required feature '{feature_name}' not found in GeoTIFF or derived features.")
            return

    feature_df = pd.DataFrame(data_for_df)
    
    # Check column order against trained features
    if not list(feature_df.columns) == list(trained_features):
         print("Error: Feature DataFrame column order does not match trained features list.")
         print(f"DataFrame order: {list(feature_df.columns)}")
         print(f"Trained order: {trained_features}")
         return

    # --- 6. Scale Features ---
    print("Scaling features...")
    scaled_features = scaler_x.transform(feature_df)

    # --- 7. Perform Inference ---
    print("Running model inference...")
    predicted_gpp_scaled = model.predict(scaled_features)

    # --- 8. Unscale GPP ---
    print("Unscaling GPP predictions...")
    # Reshape scaled prediction for inverse_transform
    predicted_gpp_unscaled = scaler_y.inverse_transform(predicted_gpp_scaled.reshape(-1, 1))

    # --- 9. Reshape and Apply Mask ---
    print("Reshaping and masking GPP array...")
    gpp_array = predicted_gpp_unscaled.reshape(height, width).astype('float32')
    
    # Apply the mask: set masked pixels (False in overall_mask) to 0 or another suitable NoData value
    # We use 0 as specified, but will also set it as the NoData value in the metadata.
    # Note: 0 GPP is a physically plausible value for a grassland pixel. Consider using a 
    # more distinct value like -9999 for true "NoData" if 0 GPP is meaningful.
    # However, since the user specified "0 or no data", and many masked areas might already be 0 in input,
    # 0 is a reasonable choice. We'll set the NoData value in the profile to 0.
    final_gpp_array = np.where(overall_mask, gpp_array, 0)

    # --- 10. Save Output GeoTIFF ---
    print(f"Saving GPP map to: {OUTPUT_GEOTIFF_PATH}...")
    
    # Update profile for output file: 1 band, float32 dtype
    profile.update({
        'count': 1,
        'dtype': 'float32',
        'nodata': 0, # Set the NoData value in the output file
        'driver': 'GTiff' # Ensure driver is explicit
    })

    try:
        with rasterio.open(OUTPUT_GEOTIFF_PATH, 'w', **profile) as dst:
            dst.write(final_gpp_array, 1)
        print("Success: GPP map generated successfully.")
    except Exception as e:
        print(f"Error: Could not save output GeoTIFF. {e}")

if __name__ == '__main__':
    main()