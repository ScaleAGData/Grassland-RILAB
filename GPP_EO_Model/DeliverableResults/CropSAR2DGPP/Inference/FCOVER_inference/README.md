# GPP Spatial Inference Pipeline

## Project Overview
This pipeline contains the inference engine designed to generate spatial maps of **Gross Primary Production (GPP)**. It utilizes pre-trained machine learning models developed within the 
**ScaleAgData - WP5 Grassland Rilab** project. 

The script translates pixel-level biophysical variables and temporal information into georeferenced productivity estimates, allowing for the transition from point-based modeling to 
landscape-scale monitoring.

---

## 1. Inputs
The script requires three primary files and a specific metadata parameter to execute correctly:

*   **Model Artifact (`best_model_GPP_Minimal_RandomForest.pkl`):** 
    *   A serialized Random Forest regressor containing the optimized weights for GPP estimation.
*   **Metadata Configuration (`inference_metadata.json`):** 
    *   A JSON file defining the required feature set—specifically `FCOVER`, `DOY_sin`, and `DOY_cos`—to ensure the input data alignment.
*   **FCOVER GeoTIFF:** 
    *   A single-band raster file representing the biophysical state of the grassland at a specific timestamp.
*   **Acquisition Date:** 
    *   An input string in `YYYYMMDD` format (e.g., `20260404`) used to derive seasonal cyclical features.

---

## 2. Processing Workflow
The script follows a rigorous workflow to ensure spatial and scientific accuracy:

### Phase 1: Initialization
*   **Model Loading:** Loads the Random Forest model and identifies the necessary features from the metadata.
*   **Temporal Engineering:** 
    *   Converts the acquisition date into the **Day of Year (DOY)**.
    *   Computes **Sine** and **Cosine** transformations of the DOY to represent seasonality as a cyclical variable.

### Phase 2: Raster Handling & Masking
*   **Spatial Profile Extraction:** Reads the input GeoTIFF and extracts its metadata (CRS, transform, dimensions).
*   **Optimization:** Creates a boolean mask to isolate valid vegetation pixels, excluding "NoData" areas (water, infrastructure) to minimize computational overhead.

### Phase 3: Prediction & Reconstruction
*   **Matrix Construction:** Assembles a feature matrix where each row represents a valid pixel. The column order is strictly maintained as: `[FCOVER, DOY_sin, DOY_cos]`.
*   **Inference:** Applies the Random Forest model to the matrix to predict GPP values.
*   **Spatial Mapping:** Reconstructs the 1D prediction array back into the 2D dimensions of the original input.

---

## 3. Outputs
The process produces a high-resolution geospatial product:

### GPP Map (GeoTIFF)
*   **Description:** A georeferenced raster file matching the dimensions of the input.
*   **Data Type:** `Float32`.
*   **Unit:** Estimated Gross Primary Production.
*   **CRS:** Inherited from the input source.
*   **NoData Value:** Inherited from the input or set to a standard null value for non-processed pixels.

---

## 4. Execution Example
To run the inference, use the following command structure:

```bash
python spatial_inference.py \
    --model best_model_GPP_Minimal_RandomForest.pkl \
    --metadata inference_metadata.json \
    --input_raster input_fcover.tif \
    --date 20260404
```

> **Note:** Ensure that the input GeoTIFF resolution and coordinate system are consistent with the requirements of your specific RILAB study area.
