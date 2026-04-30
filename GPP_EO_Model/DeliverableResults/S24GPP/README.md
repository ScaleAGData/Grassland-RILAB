GPP Estimation from Sentinel-2 (S24GPP)

This repository contains the training pipeline and inference tools for estimating Gross Primary Production (GPP) using Sentinel-1 (SAR) satellite data. 
This work is part of the ScaleAgData initiative (WP5 - Grassland RILAB).

Overview
The S24GPP module aims to develop a GPP estimation model for grassland by training a model based on in situ data collected with eddy covariance towers and Sentinel-2 data.
Full details on how the model was developed are contained in the Training folder. 
Full details on how to use the model to generate GPP maps from Sentinel-2 data are contained in the Inference folder.


Repository Structure

Training/ : Contains the core logic for model development. Further details have been provided in the README.md contained in Training fodler
Inference/: Contains the best model resulted from the train, all the ausiliary info, a Se preprocessed example and the script to create the GPP map. 
            Further details have been provided in the README.md contained in Inference fodler
