# Improved Grassland GPP Maps Based on Flux Tower Sensors

## 1. Project Overview: GPP Estimation and Grassland Monitoring

### 1.1 ScaleAgData Initiative
The **ScaleAgData** project is a European research initiative funded under the **Horizon Europe** framework (Grant Agreement No. 101083401). Spanning from January 2023 to December 2026, the project enhances the environmental performance and competitiveness of the European agricultural sector through data-driven solutions.

A core component of the project is the implementation of **RILABs (Regional Innovation Laboratories)**. These labs bridge the gap between research and practice, providing real-world testing grounds where satellite data, in-situ sensors, and modeling techniques address regional agricultural challenges.

### 1.2 RILAB Grassland
The Grassland RILAB is a collaborative effort involving multidisciplinary teams—including experts from **Indra Space** (formerly Deimos Space), **EURAC**, and **IFAPA**. The goal is to monitor the productivity and health of grassland ecosystems to provide actionable intelligence for sustainable livestock management and carbon sequestration.

**Key Services:**
*   **Gap-filled LAI maps:** High-resolution monitoring of Leaf Area Index (LAI) and FCOVER at the parcel level.
*   **Estimated Grassland Yield:** Quantifying available forage for livestock management.
*   **Improved GPP Maps:** Modeling Gross Primary Production (GPP) based on flux tower sensors to assess ecosystem efficiency.

### 1.3 GPP Estimation Pipeline
This repository provides a modular, automated pipeline for the estimation of GPP, integrating multi-source satellite data to produce high-frequency carbon flux maps.

*   **Data Integration:** Sentinel-1 (Radar), Sentinel-2 (Optical), and CropSAR2D (Gap-filled) datasets.
*   **Model Training:** Employs Random Forest, XGBoost, and MLP (Multi-Layer Perceptron) architectures.
*   **Geospatial Inference:** Generation of georeferenced **GeoTIFF** maps for GIS analysis.

---

## 2. Folder Structure

| Folder/File | Description |
| :--- | :--- |
| `CropSAR2D4GPP/` | Training and inference scripts using CropSAR2D data (provided by VITO). |
| `S14GPP/` | Training and inference scripts utilizing Sentinel-1 data. |
| `S24GPP/` | Training and inference scripts utilizing Sentinel-2 data. |
| `environment.yml` | Conda environment configuration file with all required libraries. |

> **Data Sources:** 
> *   **CropSAR2D:** Available via [Terrascope](https://portal.terrascope.be) or the [Copernicus Marketplace](https://marketplace-portal.dataspace.copernicus.eu/catalogue/app-details/80).
> *   **Sentinel-1 & 2:** Acquired from the [Copernicus Data Space Ecosystem](https://browser.dataspace.copernicus.eu/).

---

## 3. Setup and Installation

### Prerequisites
*   [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).
*   **Mamba** (Optional but recommended for faster dependency resolution).

### Environment Installation
The `environment.yml` file ensures consistency between geospatial libraries (`Rasterio`) and machine learning frameworks (`TensorFlow`, `XGBoost`).

1.  **Create the environment:**
    ```bash
    conda env create -f environment.yml
    ```
2.  **Activate the environment:**
    
```bash
    conda activate gpp_env
    ```

### Included Dependencies
*   **ML Frameworks:** Scikit-learn, XGBoost, TensorFlow.
*   **Geospatial Tools:** Rasterio, Scikit-image.
*   **Data Handling:** Pandas, Numpy, Openpyxl.
```
