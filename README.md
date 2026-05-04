# ScaleAgData: Grassland RILAB

## 1. Project Overview
The **Grassland RILAB** is a core component of the **ScaleAgData** project, a European research initiative funded under the **Horizon Europe** framework (Grant Agreement No. 101083401). 

The project aims to enhance the environmental performance and competitiveness of European agriculture by integrating multi-source data—including satellite imagery (Sentinel-1/2), in-situ sensors, and advanced modeling—into actionable data-driven solutions. This repository specifically hosts the tools for monitoring grassland productivity and carbon sequestration through 2026.

---

## 2. RILAB Grassland Framework

### 2.1 Regional Innovation Laboratories (RILABs)
RILABs serve as bridge-builders between research and practice. They provide real-world testing grounds where technology is validated against practical agricultural challenges.

### 2.2 Participants & Collaboration
The Grassland RILAB is a multidisciplinary collaboration involving:
*   **EURAC Research**: Remote sensing and environmental monitoring expertise.
*   **IFAPA**: Agrarian research and ground-truth data collection.
*   **Indra Space** (formerly Deimos Space): Geospatial processing and service architecture.

### 2.3 Areas of Interest (AOIs)
The primary study sites are located in Italy and Spain, utilizing high-frequency data from specialized research parcels and commercial grasslands.

---

## 3. Core Products
The RILAB is committed to delivering three primary geospatial products:

| Product | Description | Use Case |
| :--- | :--- | :--- |
| **Gap-filled grassland LAI maps at parcel level** | High-resolution monitoring of Leaf Area Index and FCOVER at the parcel level. | Vegetation density & health assessment. |
| **Estimated grassland yield at parcel level** | Quantifying available forage at the parcel level. | Livestock grazing and harvest planning. |
| **Improved grassland GPP maps based on flux tower sensors** | Gross Primary Production maps based on flux tower sensor calibration. | Carbon sequestration and ecosystem efficiency. |

---

## 4. Repository Structure
The main folder is organized to facilitate the transition from model training to large-scale geospatial inference.

### Directories
*   **`TransConvRegressor-Eurac/`**: The **TransConvRegressor** repository implements a 1D TransUNet-based regressor for Leaf Area Index (LAI) estimation using temporal Sentinel-1 backscatter signals and ancillary features. It includes training, evaluation, and prediction pipelines.
*   **`GPP_EO_Model/`**: It contains all the models developed for the prduct **Improved grassland GPP maps based on flux tower sensors**. All the details are provided in teh correspondent README file.
    
---

**Funding Acknowledgment:**  
*This project has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement No. 101083401.*
