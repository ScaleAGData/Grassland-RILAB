GPP Estimation from Sentinel-1 (S14GPP)

This repository contains the training pipeline and inference tools for estimating Gross Primary Production (GPP) using Sentinel-1 (SAR) satellite data. This work is part of the ScaleAgData initiative (WP5 - Grassland RILAB).  

Overview

Unlike optical sensors, Sentinel-1's Synthetic Aperture Radar (SAR) can penetrate cloud cover, providing a consistent 
temporal signal of grassland structure. This module uses backscatter coefficients (VH and VV) combined with temporal features to model carbon fluxes.  

Repository Structure

1. Training/: Contains the core logic for model development. Further details have been provided in the README.md contained in Training fodler
2. Inference/: Contains the best model resulted from the train, all the ausiliary info, a S1 preprocessed example and the script to create the GPP map. Further details have been provided in the README.md contained in Inference folder
