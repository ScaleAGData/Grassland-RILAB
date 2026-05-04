# GPP CropSAR2D Multi-Model Training Pipeline
### Project Overview: ScaleAgData (WP5, SpainSite Grassland)

This pipeline provides a Python-based framework to train, compare, and optimize machine learning models for estimating **Gross Primary Production (GPP)**. Developed under the **ScaleAgData** initiative, the script focuses on a multi-model approach using **Random Forest (RF)** and **XGBoost (XGB)** regressors. 

By leveraging biophysical variables and vegetation indices, the pipeline ensures the selection of the most stable and accurate model through an iterative optimization process.

---

## 1. Input Requirements
The script processes a CSV file containing synchronized biophysical and in-situ data.

*   **Source File:** `Input_IFAPA_GPP_cropsar2D_3vars.csv`
*   **Predictor Variables (Features):**
    *   `FCOVER`: Fraction of Vegetation Cover.
    *   `NDVI Average`: Normalized Difference Vegetation Index (temporal average).
    *   `FAPAR Average`: Fraction of Absorbed Photosynthetically Active Radiation.
*   **Target Variable:**
    *   `GPP`: Gross Primary Production (Ground-truth measurements).
*   **Temporal Data:**
    *   Date column (Format: `DD/MM/YYYY`) used for seasonal feature engineering.

---

## 2. Processing Pipeline
The automated workflow includes data cleaning, feature engineering, and performance optimization:

### Data Cleaning & Parsing
*   Standardizes column names (removes whitespace).
*   Detects and parses date columns for valid temporal record keeping.

### Feature Engineering
*   **Interaction Feature:** Creates `NDVI_FAPAR_Interact` (NDVI × FAPAR) to capture Light Use Efficiency (LUE) logic.
*   **Seasonal Encoding:** Transforms the Day of Year (DOY) into cyclical components: $DOY_{sin}$ and $DOY_{cos}$.

### Training & Optimization
*   **Stratified Data Splitting:** Bins GPP values into five levels to ensure representative distribution across Train (70%), Validation (15%), and Test (15%) sets.
*   **Iterative Optimization:** Performs **1,000 training cycles** for both RF and XGB.
*   **Selection Criteria:** Independently tracks the best version of each algorithm based on **Validation $R^2$**.

### Performance Evaluation
Calculates comprehensive error statistics:
*   Coefficient of Determination ($R^2$)
*   Root Mean Squared Error (RMSE)
*   Relative Root Mean Squared Error (RRMSE)
*   Relative Absolute Error (RAE)

---

## 3. Results & Outputs
Outputs are saved in a timestamped folder (e.g., `GPP_3vars_Inference_YYYYMMDD_HHMMSS`):

### A. Model Artifacts (`.pkl`)
*   `best_model_RandomForest.pkl`: The optimized RF regressor.
*   `best_model_XGBoost.pkl`: The optimized XGBoost regressor.

### B. Statistical Reports
*   **Performance Report (`Best_Model_Performance.xlsx`):**
    *   *Sheet 1 (Predictions):* Raw comparison between measured test values and model predictions.
    *   *Sheet 2 (Metrics):* Summary of $R^2$ (Train & Test), RMSE, RRMSE, and RAE.
*   **Iteration History (`iteration_history.csv`):** A log of $R^2$ and RMSE for every iteration to enable stability analysis.

### C. Visualizations (`performance_comparison.png`)
*   **Scatter Plots:** Observed vs. Predicted GPP for both RF and XGB.
*   **Feature Importance:** Horizontal bar charts identifying the contribution of `NDVI`, `FCOVER`, and `FAPAR`.

---

## 4. Execution & Requirements
To run the pipeline, ensure the following Python libraries are installed:

```bash
pip install scikit-learn xgboost pandas matplotlib seaborn openpyxl
```

**Usage:**
```bash
python gpp_cropsar2d_pipeline.py
```
