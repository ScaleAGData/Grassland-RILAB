# Sentinel-1 GRD Processing and Backscatter Extraction  
_Automated workflow for querying, downloading, processing, and extracting VV/VH backscatter from Sentinel-1 using Copernicus Data Space and SNAP._

---

## 📌 Overview

This repository contains a Python script that automates the processing of **Sentinel-1 GRD (IW_GRDH)** data from the **Copernicus Data Space Ecosystem (CDSE)** and extracts **VV and VH backscatter statistics** over a user-defined **Area of Interest (AOI)**.

The script:

1. Queries the **Copernicus Data Space OData API** for Sentinel-1 products intersecting an AOI and within a given date range.
2. Filters products by acquisition mode and product type (e.g. `IW_GRDH`).
3. Authenticates to CDSE and downloads each product as a ZIP file.
4. Extracts the SAFE structure and locates the Sentinel-1 measurement rasters.
5. Modifies a **SNAP GPT XML graph** (e.g. `S1_manual.xml`) to:
   - set the input product,
   - set the correct pixel region (width/height),
   - set the output product name.
6. Executes the SNAP graph via `gpt` to generate a processed product (e.g. calibrated, speckle-filtered, terrain-corrected).
7. Searches the output directory for the resulting `.img` rasters (VV and VH).
8. Extracts all pixel values within a user-defined AOI (WKT polygon) for both VV and VH.
9. Computes the mean VV and VH values for the AOI.
10. Saves the results into:
    - a **CSV file** with basic metadata and VV/VH statistics,
    - **pickle files** with the full pandas DataFrame and the original product list.

The script is designed for applications such as agricultural monitoring, radar-based indices, and integration into the **ScaleAgData** workflow.

---

## 🚀 Features

- Automatic Sentinel-1 product discovery via CDSE OData API  
- Flexible AOI definition using WKT polygons (EPSG:4326)  
- Product type filtering (e.g. `IW_GRDH`)  
- SNAP GPT integration for standardized S1 processing graphs  
- VV and VH backscatter extraction over polygons  
- Mean VV/VH backscatter statistics per acquisition  
- CSV + pickle exports for downstream analysis  
- Automatic cleanup of temporary product folders  

---

## 📥 Inputs

The main inputs are defined in `main()` or in the `if __name__ == "__main__":` block:

- **start_date**: start of the search interval (e.g. `"2018-01-21"`)
- **end_date**: end of the search interval (e.g. `"2018-01-25"`)
- **data_collection**: Copernicus collection name (e.g. `"SENTINEL-1"`)
- **datatype**: S1 product type/mode to select (default: `'IW_GRDH'`)
- **aoi_download**: WKT polygon (EPSG:4326) used to query the catalogue
- **aoi_fp**: WKT polygon (EPSG:4326) used to extract pixel values
- **csv_name**: output CSV filename (e.g. `"STC_2018-01-212018-01-25.csv"`)
- **graph_path**: path to the SNAP GPT XML graph (e.g. `S1_manual.xml`)

You must also provide **Copernicus Data Space credentials** in:

```python
access_token = get_access_token("your_email@xxx.com", "your_password")
