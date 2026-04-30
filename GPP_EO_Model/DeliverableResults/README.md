Improved grassland GPP maps based on flux tower sensors

1. Project Overview: GPP Estimation and Grassland Monitoring

1.1 ScaleAgData Initiative
The ScaleAgData project is a European research initiative funded under the Horizon Europe framework 
(Grant Agreement No. 101083401). Running from January 2023 to December 2026, the project aims to improve 
the environmental performance and competitiveness of the European agricultural sector by enhancing the use 
of data-driven solutions.
A core component of the project is the implementation of RILABs (Regional Innovation Laboratories). 
These labs act as bridge-builders between research and practice,providing real-world testing grounds where 
satellite data, in-situ sensors, and modeling techniques are integrated to address specific regional 
agricultural challenges.

1.2 RILAB Grassland
The Grassland RILAB is a collaborative effort involving multidisciplinary teams—including experts from Indra 
Space (previously Deimos Space), EURAC and IFAPA focused on monitoring the productivity and health of 
grassland ecosystems. The primary goal of this RILAB is to provide actionable intelligence for sustainable 
livestock management and carbon sequestration tracking. Within this framework, three key services have been 
developed:
a. Gap-filled grassland LAI maps at parcel level: High-resolution monitoring of Leaf Area Index (LAI) and FCOVER.
b. Estimated grassland yield at parcel level: Quantifying available forage for livestock management.
c. Improved grassland GPP maps based on flux tower sensors: Modeling the total carbon fixed by photosynthesis to 
   assess ecosystem efficiency and health.  

1.3 GPP Estimation Pipeline
This specific repository provides a modular and automated pipeline for the Estimation of Gross Primary 
Production (GPP). It integrates multi-source satellite data to produce high-frequency carbon flux maps.
The software included here manages the entire lifecycle of the GPP service:  
- Data Integration: Handling Sentinel-1 (Radar), Sentinel-2 (Optical), and CropSAR2D (Gap-filled) datasets.  
- Model Training: Using Random Forest, XGBoost, and MLP (Multi-Layer Perceptron) architectures to correlate 
  satellite observations with ground-truth Eddy Covariance measurements.  
- Geospatial Inference: Generating georeferenced GeoTIFF maps ready for GIS analysis.

2. Folder Sctructure

This folder contains the following files:
- CropSAR2D4GPP: This folder contains the training and inference scripts for models that estimate GPP 
  using CropSAR2D data. CropSAR2D data is provided by VITO and can be found at the following platforms: 
    * https://portal.terrascope.be
    * https://marketplace-portal.dataspace.copernicus.eu/catalogue/app-details/80
- S14GPP: This folder contains the training and inference scripts for models that estimate GPP using 
  Sentinel-1 data.  
- S24GPP: This folder contains the training and inference scripts for models that estimate GPP using 
  Sentinel-2 data.  
- environment.yml: A YAML file used to create a Conda environment containing all the libraries required 
  to run the scripts in this folder and its subfolders.

Both Sentinel-1 and Sentinel-2 data had been acquired from the Copernicus catalogue 
(https://browser.dataspace.copernicus.eu/). Their preprocessing is described in the README files 
of their respective models. 
To ensure consistency and compatibility between geospatial libraries (Rasterio) and machine learning 
frameworks (TensorFlow, XGBoost), please follow the instructions below to set up your Python environment.

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or 
  [Anaconda](https://www.anaconda.com/) installed on your system.
- (Optional but Recommended) [Mamba](https://mamba.readthedocs.io/en/latest/installation.html) 
  for faster dependency resolution.

## Environment Installation

The `environment.yml` file contains all necessary dependencies, including:
- **ML Frameworks:** Scikit-learn, XGBoost, TensorFlow.
- **Geospatial Tools:** Rasterio, Scikit-image.
- **Data Handling:** Pandas, Numpy, Openpyxl.

### Using Conda
Open your terminal or Anaconda Prompt, navigate to the project folder, and run:
```bash
conda env create -f environment.yml
