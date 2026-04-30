# GPP Estimation from Sentinel-2: Inference Pipeline

This folder contains the specialized tools and scripts for the **Inference Phase** of the Sentinel-2 Gross Primary Production (GPP) estimation service. These scripts are designed to apply pre-trained machine learning models to multi-spectral geospatial rasters (GeoTIFFs) to generate high-resolution carbon flux maps.

## Contents

*   **`GeospatialGPPS2Inference.py`**: The primary Python script for performing pixel-wise GPP inference on Sentinel-2 imagery.
*   **Support Artifacts**: (To be placed here from the Training output):
    *   `gpp_model.pkl`: The trained MLP/ANN regressor.
    *   `scaler_x.pkl` / `scaler_y.pkl`: Pre-computed scaling parameters.
    *   `features_list.pkl`: Metadata ensuring the correct order of spectral bands during inference.

## Technical Workflow

The inference engine follows a structured geospatial pipeline:

1.  **Temporal Feature Extraction**: Converts the user-defined acquisition date into a **Day of Year (DOY)** value to account for seasonal vegetation dynamics.
2.  **Spectral Pre-processing**: 
    *   Reads 10 specific Sentinel-2 bands: **B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12**.
    *   Calculates the **NDVI** (Normalized Difference Vegetation Index) from B08 and B04.
3.  **NoData Management**: Detects internal masks and NoData values in the input GeoTIFF to ensure that non-vegetated or missing pixels remain null in the final output.
4.  **Neural Network Prediction**: Applies the normalized feature matrix to the trained MLP model.
5.  **Raster Reconstruction**: Denormalizes the output values (gC/m²/day) and saves them into a new 32-bit float GeoTIFF, preserving the original Coordinate Reference System (CRS) and spatial transform.

## Usage Instructions

### 1. Requirements
Ensure the `gpp_env` Conda environment is active.
```bash
conda activate gpp_env
```

### 2. Configuration
Open `GeospatialGPPS2Inference.py` and verify the following variables:
*   `INPUT_GEOTIFF_PATH`: Path to your S2 L2A image.
*   `INPUT_DATE_STR`: The date of acquisition (format: `YYYY-MM-DD`).
*   `OUTPUT_GEOTIFF_PATH`: Destination for the resulting GPP map.

### 3. Execution
Run the script to generate the map:
```bash
python GeospatialGPPS2Inference.py
```

## Output Specifications
*   **Format**: Single-band GeoTIFF.
*   **Data Type**: Float32.
*   **Units**: gC/m²/day.
*   **NoData Value**: 0 (Default, represents masked or physically null GPP areas).

---
**Funding**: Developed under the **ScaleAgData** project (Horizon Europe Grant Agreement No. 101083401).
