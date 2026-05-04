GPP CropSAR2D Multi-Model Training Pipeline (ScaleAgData)
Project Overview
This project, developed under the ScaleAgData initiative (Work Package 5, SpainSite Grassland), provides 
a Python-based pipeline to train, compare, and optimize machine learning models for estimating Gross Primary 
Production (GPP).
The script focuses on a multi-model approach, simultaneously training and evaluating Random Forest (RF) 
and XGBoost (XGB) regressors. It utilizes biophysical variables (FCOVER, FAPAR) and vegetation indices (NDVI) 
to estimate GPP, ensuring the most stable and accurate model is selected through an iterative optimization process.

1. Input Requirements
The script processes a CSV file containing synchronized biophysical and in-situ data.
- Source File: Input_IFAPA_GPP_cropsar2D_3vars.csv
- Predictor Variables (Features):
  - FCOVER: Fraction of Vegetation Cover.
  - NDVI Average: Normalized Difference Vegetation Index (temporal average).
  - FAPAR Average: Fraction of Absorbed Photosynthetically Active Radiation.
- Target Variable:
  - GPP: Gross Primary Production (Ground-truth measurements).
Temporal Data: 
  - A date column (handled in European format DD/MM/YYYY) used for seasonal feature engineering.

2. Processing Pipeline
The script executes an automated workflow to clean data, engineer new features, and optimize model performance:  
 - Data Cleaning & Parsing: Loads the source file and standardizes column names by removing whitespace. 
   It automatically detects and parses the date column to ensure valid temporal records.  
 - Feature Engineering:
     * Interaction Feature: Creates NDVI_FAPAR_Interact by multiplying the two variables to capture Light Use 
       Efficiency logic.  
     * Seasonal Encoding: Transforms the Day of Year (DOY) into cyclical Sine and Cosine components 
       (DOY_sin, DOY_cos).  
     * Stratified Data Splitting: Bins GPP values into five levels to ensure a representative distribution 
       across Train (70%), Validation (15%), and Test (15%) sets.  
     * Iterative Optimization: Performs 1,000 training cycles for both Random Forest (RF) and XGBoost (XGB). 
       It independently tracks the best-performing version of each algorithm based on Validation R^2.  
     * Performance Evaluation: Calculates comprehensive error statistics, including $R^2$, RMSE, 
       Relative Root Mean Squared Error (RRMSE), and Relative Absolute Error (RAE).  
3. Results & OutputsThe script creates a timestamped output folder (e.g., GPP_3vars_Inference_20260309_120000) 
containing:
A. Model Artifacts (.pkl)
  - best_model_RandomForest.pkl: The optimized Random Forest regressor.
  - best_model_XGBoost.pkl: The optimized XGBoost regressor.
B. Statistical Reports
  - Performance Report (Best_Model_Performance.xlsx): 
    * Sheet 1 (Predictions): Raw comparison between measured test values and model predictions.
    * Sheet 2 (Metrics): Summary of $R^2$ (Train & Test), RMSE, RRMSE, and RAE for the top-performing model.
  - Iteration History (iteration_history.csv): A log of $R^2$ and RMSE for every iteration and every model for 
    stability analysis.
C. Visualizations
- Comparison Plots (performance_comparison.png): 
  * Scatter Plots: Observed vs. Predicted GPP for both RF and XGB.
  * Feature Importance: Horizontal bar charts showing which variables (NDVI, FCOVER, or FAPAR) contributed 
    most to the estimations.

4. Execution
To run the script, ensure scikit-learn, xgboost, pandas, matplotlib, and seaborn are installed
Bash
python GPP_Training_Pipeline.py