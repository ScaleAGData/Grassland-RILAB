# Sentinel-2 Automated Download & Processing Tool  
_A Python workflow for querying, downloading, and extracting Sentinel-2 spectral information from the Copernicus Data Space Ecosystem._

---

## 📌 Overview

This repository contains a Python script designed to **automatically download and process Sentinel-2 L2A satellite images** over a user-defined **time period** and **Area of Interest (AOI)**.

The tool:
- Queries the **Copernicus Data Space Ecosystem (CDSE)** catalogue.
- Authenticates using your Copernicus credentials.
- Downloads Sentinel-2 L2A products intersecting your AOI.
- Extracts cloud probability information.
- Extracts pixel values for all Sentinel-2 bands (B01–B12).
- Computes average reflectance per band.
- Saves all results to **CSV and pickle files**.
- Cleans up intermediate data automatically.

It is intended for remote sensing workflows where automatic retrieval and preprocessing of Sentinel-2 imagery is needed for analysis, machine learning, or large-scale monitoring.

---

## 🚀 Features

✓ Automatic query of Sentinel-2 L2A products  
✓ AOI-based product filtering  
✓ Secure authentication via CDSE API  
✓ Automatic download and ZIP extraction  
✓ Cloud probability extraction (`MSK_CLDPRB_20m.jp2`)  
✓ Pixel value extraction for all S2 spectral bands  
✓ Average reflectance calculation per band  
✓ CSV and pickle export  
✓ Automatic cleanup of temporary folders  
✓ Supports restart/reprocessing of unfinished lists  

---

## 📥 Required Inputs

You must define the following in the `main()` function (or pass them externally):

### **1. `start_date` / `end_date`**
Time interval for querying Sentinel-2 data.  
Format: `"YYYY-MM-DD"`

### **2. `aoi_download`**
A WKT polygon (in EPSG:4326) defining the **search area** used to query the catalogue.

### **3. `aoi_fp`**
A WKT polygon defining the **extraction area** for pixel and cloud values.

### **4. Copernicus Data Space Ecosystem credentials**
Edit the following line before running:

```python
access_token = get_access_token("your_email@xxx.com", "your_password")
