## 🌱 TransConvRegressor: LAI Estimation with SAR Backscatter Signals

This repository implements **TransConvRegressor**, a 1D TransUNet-based regressor for **Leaf Area Index (LAI)** estimation using temporal Sentinel-1 backscatter signals and ancillary features. It includes training, evaluation, and prediction pipelines.

- Model Architecture
The core model is a 1D Transformer-based architecture, combining convolutional layers with transformer-based attention mechanisms to capture spatial and temporal dependencies in the Sentinel-1 backscatter signals.
<img width="3919" height="1470" alt="image" src="https://github.com/user-attachments/assets/f9de8c09-1b3e-4b2a-a651-179af88a4919" />



## 📂 Repository Structure
- **network.py**: Contains the `model` function and helper functions for building the 1D TransConvRegressor model.
- **train.py**: Loads data, applies preprocessing, creates train/val/test splits, trains the model, saves the model, scaler and evaluation metrics.
- **prediction.py**: Loads the trained model, applies preprocessing, makes predictions on new data and saves outputs.
- **requirements.txt**: Lists all Python packages required to run the scripts.

## Data Description
- S1 SAR (RTC): We downloaded the S1 RTC from the Microsoft Planetary Computer as netcdfs. RTC is radiometrically terrain corrected GRD data, to account for radiometrics effects resulting from the topography.
- Sentinel-2 derived LAI: We computed LAI from the S2 L2A files over selected years and tiles using the SNAP Biophysical Processor.
- Soil Moisture: We considered Surface Soil Moisture based on a combination of SAR and optical imagery [1]. We simulated the required soil moisture data over the AOI.
- Topographical Classes: Type of meadows are selected from LAFIS data. We have only considered colline, submontane, montane and subalpine (e.g. S_coll, N_coll, S_submont, N_submont, S_mont, N_mont, S_subalp, N_subalp) split into south and north facing meadows for this study.
- AOI: LAFIS Grasslands over South Tyrol (Italy)
[1] F. Greifeneder, C. Notarnicola, and W. Wagner, “A machine learning-based approach for surface soil moisture estimations with google earth engine,” Remote Sensing, vol. 13, no. 11, p. 2099, 2021.

## Installation
- gh repo clone Eurac-Research-Institute-for-EO/TransConvRegressor
- cd TransConvRegressor
- pip install -r requirements.txt

## Training, Validation and Test
python train.py
- This will initiate the training process, including data preprocessing, model training, validation and test.
- We used a CSV dataset containing Sentinel‑1 RTC features, soil moisture information, altitudinal classes and target S2 LAI for training and validation, which can be summarized as follows:
<img width="521" height="104" alt="image" src="https://github.com/user-attachments/assets/655a9ef6-d7bf-4818-89a3-97af202483a4" />


# Analysis of the trained model on the test set
- Comparison of scatter plot between predicted LAI and S2-LAI on the test set
<img width="5280" height="3032" alt="image" src="https://github.com/user-attachments/assets/8da11491-d632-46d1-8262-9b13b40e0042" />
- Comparison of box plot between predicted LAI and S2-LAI on the test sets during the growing season on the test set.
<img width="4023" height="2216" alt="image" src="https://github.com/user-attachments/assets/4d63ceeb-7b0d-41ba-812b-e94664a5fd05" />


## Prediction
python prediction.py
- This script loads the trained model, performs inference on the test data, and saves the predicted LAI values along with additional information to CSV files.
- We used a CSV dataset containing Sentinel‑1 RTC features, soil moisture information and altitudinal classes for prediction on unseen sites, which can be summarized as follows:
<img width="494" height="92" alt="image" src="https://github.com/user-attachments/assets/ab71c4e1-49c2-47f4-a4f5-8a49c65cef42" />

- (a) Predicted LAI vs S2-LAI in one of the unseen field sites in Fondo F1 (Trento), (b) Trend of gap-filled LAI with Sentinel-2 LAI in Fondo - F1(Trento)
  
<img width="4986" height="2407" alt="image" src="https://github.com/user-attachments/assets/2c4577b9-23a2-424e-aa0c-ebaa20bff296" />


## Link to the Paper
[Link](https://ieeexplore.ieee.org/document/11303207)
- Cite this paper:
A. Singh et al., "A Transformer-based Convolutional Regressor to include SAR Backscatter Signals in Monitoring Alpine Grasslands," in IEEE Geoscience and Remote Sensing Letters, doi: 10.1109/LGRS.2025.3645675.




