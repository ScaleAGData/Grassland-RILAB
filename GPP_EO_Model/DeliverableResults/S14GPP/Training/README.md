README: GPP Prediction using Sentinel-1 and MLP
Project Overview
This project, developed under the ScaleAgData initiative, provides a Python-based pipeline to train a 
Multilayer Perceptron (MLP) neural network. The model is designed to estimate Gross Primary Production 
(GPP)—specifically EC-GPP—using satellite data from Sentinel-1 (SAR backscatter) and temporal features.

1. Necessary Inputs

Data Requirements
The script expects an Excel file (.xlsx) containing the following columns:
- ac_date: The acquisition date of the satellite data (used for time-sorting and generating DOY).
- vh_av: The average VH polarization backscatter from Sentinel-1.
- vv_av: The average VV polarization backscatter from Sentinel-1.
- EC - GPP: The target variable (Ground-truth GPP from Eddy Covariance towers).

Software Dependencies
To run the script, ensure you have the following libraries installed:
- pandas & openpyxl: For data manipulation and Excel reading.
- numpy: For numerical operations.
- scikit-learn: For data scaling and evaluation metrics.
- tensorflow: For building and training the MLP model.
- joblib: For exporting the data scalers.

2. How the Code Works
The script follows a structured machine learning workflow:

A. Data Preprocessing & Feature Engineering
1) Cleaning: Rows with missing values in the required columns are dropped.
2) Sorting: The dataset is sorted chronologically by ac_date.
3) DOY Generation: A "Day Of Year" (DOY) feature is extracted from the date to help the model 
   understand seasonality.
4) Cyclic Splitting: Instead of a random split, the script uses a modulo-based cyclic split (70% 
   Train, 15% Val, 15% Test). This ensures that all three sets contain samples from across the entire 
   time series, preventing seasonal bias.

B. Normalization
- Features (X): Standardized using StandardScaler (Mean=0, Std=1).
- Target (y): Normalized using MinMaxScaler into a range of $[0, 1]$.
Note: Scalers are fitted only on the training set and then applied to validation/test sets to prevent data 
leakage.

C. Model Architecture & Training
- Architecture: A Sequential MLP with a user-defined number of hidden layers (default: 2) 
  containing 64 neurons each with ReLU activation.
- Output Layer: A single neuron with a ReLU activation to ensure GPP predictions are never negative.
- Optimization: Uses the Adam optimizer and Mean Squared Error (MSE) loss.
- Callbacks:
    - EarlyStopping: Stops training if validation loss stops improving (patience of 15 epochs).
    - ModelCheckpoint: Automatically saves the version of the model that performed best on the validation set.

3. Final Outputs
After execution, the script generates several files in the script's directory:

Model & Scalers
- ECGPP_MLP_best_model_5.keras: The trained Keras model (best weights).

- scaler_X.pkl: The saved scaler for input features.

- scaler_y.pkl: The saved scaler for the target variable (required to inverse-transform predictions back 
  to original GPP units).

Data Exports
- validation_set5.csv: The specific rows from the original Excel file used for validation.
- test_set5.csv: The specific rows used for the final independent test.

Performance Logs
- ECGPP_MLP_results5.txt: A detailed log file containing:Execution timestamp and input file path.
- Model configuration (layers, batch size, features used).
- Training history summary (actual epochs run, best validation loss).
- Final Metrics: R^2, RMSE, and MAE calculated on the test set in the original physical scale of GPP.
Usage Note
To adapt the script for your own environment, update the input_excel variable in the User Parameters 
section with the local path to your dataset.
