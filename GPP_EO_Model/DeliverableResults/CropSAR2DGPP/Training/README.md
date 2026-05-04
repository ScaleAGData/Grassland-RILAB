GRASSLAND RILAB - SCALEAGDATA PROJECT
REFERENCE FOLDER: GPP ESTIMATION MODELS
=======================================

1. PROJECT CONTEXT
------------------
This folder contains models developed for the "Improved grassland GPP maps 
based on flux tower sensors" task within the Grassland Rilab of the 
ScaleAgData project. These models utilize CropSAR-2D data (Sentinel-1 and 
Sentinel-2 fusion) for Gross Primary Productivity (GPP) estimation.

2. FOLDER STRUCTURE
-------------------
The repository is organized into two primary subfolders:

- 3varmodel:
    Contains all components required to train the 3-variable GPP estimation 
    model using FAPAR, FCOVER, and NDVI. Refer to the subfolder's README 
    for full technical details.

- FCOVERModel:
    Contains all components required to train the GPP estimation model 
    based exclusively on FCOVER data. Refer to the subfolder's README 
    for full technical details.

3. MODEL SELECTION & SENSITIVITY ANALYSIS
-----------------------------------------
A sensitivity analysis performed on the 3-variable model indicated that 
model outputs were strongly influenced by FCOVER. 

Given that the performance between the 3-variable model and the 
FCOVER-only model was very similar, the FCOVER model was selected as 
the reference for inference to ensure economy of use and computational 
efficiency.

CROPSAR-2D DATA DESCRIPTION
===========================

1. OVERVIEW
-----------
CropSAR-2D (also known as CropSAR_px) is a gap-filled, cloud-free satellite 
data service available through the Copernicus Data Space Ecosystem (CDSE). 
It provides a continuous time series of vegetation indices by fusing 
radar and optical satellite imagery.

2. CORE METHODOLOGY: RADAR-OPTICAL FUSION
-----------------------------------------
The dataset addresses the limitation of "optical gaps" caused by cloud 
cover using two primary inputs:
- Sentinel-1 (Synthetic Aperture Radar): Provides data regardless of weather 
  conditions or cloud cover.
- Sentinel-2 (Optical): Provides multispectral imagery but is weather-dependent.

A deep learning architecture (combining ResNet blocks and Transformers) is 
used to infer optical vegetation characteristics from radar signals during 
cloudy periods, ensuring a seamless data stream.

3. TECHNICAL SPECIFICATIONS
---------------------------
- Temporal Resolution: Fixed 5-day intervals.
- Spatial Resolution: 10 meters (aligned with Sentinel-2).
- Input Sources: Collocated Sentinel-1 and Sentinel-2 datacubes.
- Primary Outputs: 
    - NDVI (Normalized Difference Vegetation Index)
    - FAPAR (Fraction of Absorbed Photosynthetically Active Radiation)
    - FCOVER (Fraction of Green Vegetation Cover)
