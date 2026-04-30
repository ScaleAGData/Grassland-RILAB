Geospatial GPP Inference Pipeline

Project Overview
This script represents the inference (application) phase of the ScaleAgData project. It utilizes a pre-trained Multilayer 
Perceptron (MLP) neural network to transform raw Sentinel-1 satellite imagery into spatial maps of Gross Primary Production (GPP).
The script processes the input image pixel-by-pixel, applying learned weights to translate radar backscatter and temporal data 
into biological carbon flux estimates.

1. Necessary Inputs
To run the script successfully, the following files must be present in the specified directory:
- Multi-band GeoTIFF (.tiff): The source satellite image containing Sentinel-1 data (typically Band 1 for VH and Band 2 for VV).
  For further details on how to obtain the input image for Geospatial GPP Inference from the Sentinel-1 image downloaded from the 
  Copernicus hub, see the Sentinel1_preprocessing_info file (in the folder utilities).
- Pre-trained Model (.keras): The saved Keras model file generated during the training phase (e.g., ECGPP_MLP_best_model_2.keras).
- Serialized Scalers (.pkl):
       - scaler_X.pkl: Used to normalize input features (VH, VV, and DOY) based on training statistics.
       - scaler_y.pkl: Used to "inverse-transform" the model's output from a 0–1 range back into physical GPP units.
- Acquisition Date (ac_date): A string parameter (format: 'YYYY-MM-DD') representing when the image was captured, used to calculate 
  the Day of Year (DOY).

2. Processing Workflow
The script follows a rigorous geospatial data pipeline:

1) Asset Loading: It initializes the TensorFlow/Keras environment and loads the normalization parameters using joblib.
2) Temporal Feature Engineering: The provided ac_date is converted into a numerical "Day of Year" (1–366) to allow the model to 
    account for seasonal vegetation cycles.
3) Geospatial Masking: 
     - The script opens the TIFF using the rasterio library to maintain spatial metadata. 
       It identifies NoData pixels or background areas (e.g., values of 0 or -9999).
     - Optimization: Inference is only performed on valid pixels to maximize computational efficiency.

4) Model Inference:
     - Valid pixel data (VH and VV backscatter) are paired with the DOY.
     - Data is normalized and passed through the MLP model to generate predictions.
5) Data Reconstruction: The predicted 1D array is reshaped back into a 2D spatial grid corresponding to the original image dimensions.

3. Final Outputs
The execution results in a specialized geospatial product:
- File Name: gpp_IFAPA_Test_Corrected.tif (or user-defined output path).
- Technical Specifications:
      - Single-band Raster: Contains the predicted GPP values.
      - Data Type: 32-bit Floating Point (float32) for high numerical precision.
      - Georeferencing: The output inherits the Coordinate Reference System (CRS) and spatial transform from the input, ensuring 
        it aligns perfectly with other maps in GIS software like QGIS or ArcGIS.
      - NoData Integrity: Pixels that were empty in the input remain empty in the output, preventing artifacts.
