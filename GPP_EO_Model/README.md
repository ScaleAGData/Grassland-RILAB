# ScaleAgData: GPP Estimation Model (machine learning methods) – Grassland RILAB

This repository contains the models and processing pipelines for estimating **Gross Primary Production (GPP)** as part of the **ScaleAgData** project.

## Project and RILAB Context

### ScaleAgData Initiative
**ScaleAgData** is a European research project funded under the **Horizon Europe** framework (Grant Agreement No. **101083401**). The project runs from **January 2023 to December 2026** and focuses on enhancing the environmental performance and competitiveness of European agriculture through data-driven innovation.

### The Grassland RILAB
The **RILAB** (Regional Innovation Laboratory) concept is central to the project, acting as a collaborative space to test and validate research in real-world conditions. The **Grassland RILAB**—composed of partners including **Indra** and **VITO**—is dedicated to monitoring grassland ecosystems. Its primary objective is to support sustainable management and carbon tracking through three core services:
1.  **Biophysical Variable Retrieval** (e.g., FCOVER, LAI).
2.  **Biomass Estimation**.
3.  **GPP Estimation** (the focus of this specific repository).

---

## Repository Content

This folder acts as the central hub for the Earth Observation (EO) models developed to estimate carbon flux (GPP). It organizes different machine learning approaches based on the specific satellite data source used.

### Main Components:

*   **`DeliverableResults/`**: The primary directory containing subfolders for each sensor-specific model:
    *   **S14GPP**: Models and inference scripts using **Sentinel-1** (radar) data.
    *   **S24GPP**: Models and inference scripts using **Sentinel-2** (multispectral) data.
    *   **CropSAR2D4GPP**: Models utilizing the **CropSAR2D** gap-filled product (provided by VITO via [Terrascope](https://portal.terrascope.be)).
*   **`utils/`**: A collection of auxiliary Python scripts for data preprocessing, including:
    *   `EOdata_csvunifier.py`: Consolidates various data sources into a single CSV format.
    *   `Sentinel1_preprocessing4ANN.py` and `Sentinel2_preprocessing4ANN.py`: Scripts designed to prepare raw satellite features for Neural Network training.
*   **`environment.yml`**: The configuration file required to create the **Conda environment** (`gpp_env`). This file ensures all necessary libraries (TensorFlow, XGBoost, Rasterio, Scikit-learn) are installed with the correct versions to run the scripts in this repository.

---

## Quick Start

1.  **Environment Setup**:
    Create the environment using the provided YAML file:
    ```bash
    conda env create -f environment.yml
    conda activate gpp_env
    ```

2.  **Execution**:
    Each sensor subfolder (S1, S2, CropSAR) contains its own specific `README` with instructions on how to run training and inference for that particular model.

---
**Author:** Paolo Cosmo Silvestro
**Updated:** April 2026
```
