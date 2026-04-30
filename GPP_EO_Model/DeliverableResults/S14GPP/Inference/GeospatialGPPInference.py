# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:32:24 2026


@project ScaleAgData
@author: Paolo Cosmo Silvestro

Script Overview: Geospatial GPP Inference
The script provides an automated pipeline for generating a Gross Primary Production (GPP) map by applying a pre-trained machine learning model to a multi-band geospatial raster (GeoTIFF).

Key Functional Steps
Geospatial Data Ingestion:
The script uses the rasterio library to open the source TIFF file. It extracts not only the pixel values but also the geospatial metadata (Coordinate Reference System, transform, and dimensions). This ensures the output GPP map is correctly georeferenced and aligns perfectly with the input data.

Dimensionality Transformation (Flattening):
Satellite imagery is stored in 3D arrays (Bands × Height × Width). Since machine learning models typically require a 2D input (Samples × Features), the script flattens the spatial grid. Each pixel is treated as an individual observation, where the different spectral bands or environmental variables serve as the input features.

NoData Masking & Optimization:
To ensure scientific accuracy and computational efficiency, the script identifies NoData pixels (empty or background areas). It restricts the model's inference only to valid pixels, preventing the generation of "noise" or artifacts in areas where data is missing.

Model Inference:
The pre-trained model (loaded via joblib) processes the valid pixel data to predict GPP values. This step translates raw environmental inputs—such as vegetation indices, temperature, or radiation—into biological carbon flux estimates.

Raster Reconstruction and Export:
The resulting 1D array of GPP predictions is reshaped back into a 2D spatial grid. The script then initializes a new GeoTIFF file, mapping the calculated GPP values into a single-band raster optimized for analysis in GIS software like QGIS or ArcGIS.

Technical Specifications
Input: Multi-band GeoTIFF (Predictors) + Model File (.pkl).

Output: Single-band GeoTIFF (GPP Values).

Data Type: 32-bit Floating Point (to maintain precision in carbon flux units).

Handling: Built-in support for spatial projections and coordinate alignment.


"""

import rasterio
import numpy as np
import os
import tensorflow as tf
import joblib
from datetime import datetime

def generate_gpp_map(input_tiff, output_tiff, model_path, scaler_x_path, scaler_y_path, ac_date='2025-02-03'):
    """
    Generates a GPP map ensuring NoData areas from input are preserved in the output.
    """
    
    # 1. Load Model and Scalers
    if not all(os.path.exists(p) for p in [model_path, scaler_x_path, scaler_y_path]):
        raise FileNotFoundError("Check if model or scaler files exist in the specified paths.")
        
    model = tf.keras.models.load_model(model_path)
    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    
    # 2. Convert Date to Day of Year (DOY)
    date_obj = datetime.strptime(ac_date, '%Y-%m-%d')
    doy_value = date_obj.timetuple().tm_yday
    
    with rasterio.open(input_tiff) as src:
        # Prepare Metadata - Impostiamo esplicitamente il valore nodata a -9999
        meta = src.meta.copy()
        nodata_val = -9999
        meta.update({
            "driver": "GTiff", 
            "count": 1, 
            "dtype": "float32", 
            "nodata": nodata_val
        })
        
        # Read S1 Bands
        data = src.read().astype(np.float32)
        n_bands, height, width = data.shape
        
        # --- GESTIONE NODATA ---
        # Creiamo una maschera booleana: True dove il pixel è VALIDO
        # Un pixel è valido se NON è uguale al valore nodata del file E NON è 0 (se 0 indica assenza dato)
        input_nodata = src.nodata if src.nodata is not None else 0
        
        # La maschera è True solo se TUTTE le bande hanno dati validi
        mask = np.all(data != input_nodata, axis=0) & np.all(data != 0, axis=0)
        mask_flat = mask.flatten()
        
        # Flatten dei dati per il modello
        pixels = data.reshape(n_bands, -1).T
        
        # Inizializziamo il risultato con il valore NoData (-9999)
        gpp_flat = np.full(pixels.shape[0], nodata_val, dtype=np.float32)
        
        if np.any(mask_flat):
            # Prendiamo solo i pixel dove la maschera è True
            valid_pixels = pixels[mask_flat] 
            
            # 3. Feature Construction: [DOY, vh_av, vv_av]
            doy_col = np.full((valid_pixels.shape[0], 1), doy_value)
            model_input_raw = np.hstack([doy_col, valid_pixels])
            
            # 4. Input Normalization
            model_input_scaled = scaler_X.transform(model_input_raw)
            
            # 5. Prediction
            print(f"Running inference on {np.sum(mask_flat)} valid pixels...")
            pred_scaled = model.predict(model_input_scaled, batch_size=4096).ravel()
            
            # 6. Output Denormalization
            gpp_real = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            
            # Assegniamo i risultati solo ai pixel validi
            gpp_flat[mask_flat] = gpp_real

        # 7. Reshape and Export
        gpp_map = gpp_flat.reshape(height, width)
        
        with rasterio.open(output_tiff, 'w', **meta) as dst:
            dst.write(gpp_map, 1)
            
    print(f"Success! GPP Map saved as: {output_tiff}")

# ======================================================================
# EXECUTION
# ======================================================================
# Otteniamo la cartella dove si trova lo script
base_path = os.path.dirname(os.path.abspath(__file__))

generate_gpp_map(
    # Usiamo os.path.join per evitare errori di slash mancanti
    input_tiff    = os.path.join(base_path, 'IFAPA_s1_testarea.tiff'),
    output_tiff   = os.path.join(base_path, 'gpp_IFAPA_Test_Corrected.tif'),
    model_path    = os.path.join(base_path, 'ECGPP_MLP_best_model_5.keras'),
    scaler_x_path = os.path.join(base_path, 'scaler_X.pkl'),
    scaler_y_path = os.path.join(base_path, 'scaler_y.pkl'),
    ac_date       = '2025-02-03'
)