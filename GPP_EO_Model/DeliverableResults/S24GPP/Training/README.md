GPP Estimation with S2 data Pipeline (ScaleAgData)
Project Overview
Developed under the ScaleAgData initiative (Work Package 5, SpainSite Grassland), this script provides 
a Python-based pipeline to train a Multilayer Perceptron (MLP) neural network. The model is designed 
to estimate Gross Primary Production (GPP)—specifically EC-GPP—using satellite data and temporal 
features. By integrating remote sensing reflectance with in-situ Eddy Covariance measurements (IFAPA), 
the script creates a robust predictive model for monitoring grassland productivity.

1. Input Requirements
The script expects an Excel file (IFAPA_insitu_Sentinel2_Test1.xlsx) containing synchronized satellite 
and ground-truth data.
  - Spectral Data: Sentinel-2 bands (B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12).
    Note: Atmospheric noise bands B01, B09, and B10 are automatically filtered out.
  - Target Variable: GPP_DT_uStar (In-situ GPP measurements).
  - Temporal Data: A Date column used to derive the Day of Year (DOY) to capture seasonal vegetation cycles.
  - Feature Engineering: The script automatically calculates the NDVI (Normalized Difference Vegetation Index) 
    as an additional predictive feature:$$NDVI = \frac{B08 - B04}{B08 + B04}$$

2. Training Model & Workflow
The pipeline utilizes a supervised learning approach with an Artificial Neural Network (ANN).
  - Architecture: Multi-Layer Perceptron (MLP) with two hidden layers of 12 and 6 neurons respectively.
  - Hyperparameters:Activation: ReLU (Rectified Linear Unit).
  - Solver: lbfgs (optimized for stability on smaller/medium datasets).
  - Preprocessing: Data is normalized using MinMaxScaler to a range of $[0, 1]$.
  - Data Splitting: 
    - 70% Training: Used for model fitting.
    - 15% Validation: Used during the optimization loop to select the best weight initialization.
    - 15% Testing: A final "hold-out" set to evaluate generalization on unseen data.
  - Optimization Loop: The script performs up to 10,000 random initializations to find the weight state 
    that maximizes the Validation $R^2$ (targeting a threshold of $0.90$).

3. Results & Outputs
Upon completion, the script generates a timestamped folder 
(Run_GPP_Cleaned_YYYYMMDD_HHMMSS) containing:
  A. Model Artifacts (.pkl files)
    - gpp_model.pkl: The trained MLP regressor.
    - scaler_x.pkl & scaler_y.pkl: Scalers required to normalize/denormalize data for future inference.
    - features_list.pkl: A list of the exact columns used during training.
  B. Performance Report (Excel)
An Excel file containing two sheets:
PREDICTIONS: A comparison of observed vs. predicted GPP values for the Test Set (denormalized to original units: $gC/m^2/day$).
METRICS: Key statistical indicators including:
    - $R^2$ (Coefficient of Determination): For both Validation and Test sets.
    - RMSE (Root Mean Squared Error): Absolute error magnitude.
    - RRMSE & RAE: Relative error metrics for normalized performance assessment.

4. Execution
To run the script, ensure all dependencies (scikit-learn, pandas, joblib, openpyxl) are installed and the input Excel file 
is located in the same directory as the script.
Bash: python EO_GPP_model_6.py
