# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 18:30:02 2024

@author: pcss
"""

import requests
import pandas as pd
import zipfile
import os
from shapely import wkt
from shapely.ops import transform
from rasterio.features import geometry_mask
import rasterio
import pyproj
import shutil
import pickle

def fetch_satellite_data(start_date, end_date, data_collection, aoi):
    """Fetch satellite data from Copernicus catalogue."""
    response = requests.get(
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq '{data_collection}' and OData.CSC.Intersects(area=geography'SRID=4326;{aoi}) and ContentDate/Start gt {start_date}T00:00:00.000Z and ContentDate/Start lt {end_date}T00:00:00.000Z"
    )
    
    try:
        json_data = response.json()
        output_list = pd.DataFrame.from_dict(json_data["value"]).head(50).values.tolist()
    except KeyError:
        print("Error: 'value' not found in JSON response.")
        output_list = []
    
    return output_list

def filter_output_list_by_datatype(output_list, datatype):
    """Filter the output list by datatype."""
    return [item for item in output_list if 'MSI' + datatype in item[2]]

def generate_url(input_string):
    """Generate a URL based on the provided input string."""
    return f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({input_string})/$value"

def get_access_token(username, password):
    """Get access token for authorization."""
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    try:
        r = requests.post(
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
            data=data,
        )
        r.raise_for_status()
    except Exception:
        raise Exception(
            f"Access token creation failed. Response from the server was: {r.json()}"
        )
    return r.json()["access_token"]

def download_file(access_token, url, output_name):
    """Download file using access token and URL."""
    headers = {"Authorization": f"Bearer {access_token}"}
    session = requests.Session()
    session.headers.update(headers)
    response = session.get(url, headers=headers, stream=True)

    print(f"Attempting to write to file: {output_name}")

    try:
        with open(output_name, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
    except OSError as e:
        print(f"Error writing file: {e}")

def extract_and_remove_zip(zip_path, extract_to):
    """Extract and remove zip file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            extracted_items = zip_ref.namelist()
            root_folder = os.path.commonpath(extracted_items)
            extracted_folder = os.path.join(extract_to, root_folder)
        
        os.remove(zip_path)
        return extracted_folder
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def check_and_update_path(inputpath):
    """Check and update the path to include necessary subdirectories."""
    outpath = os.path.join(inputpath, "GRANULE")
    if not os.path.exists(outpath):
        return outpath, "outpath does not exist"
    
    subdirs = [d for d in os.listdir(outpath) if os.path.isdir(os.path.join(outpath, d))]
    if not subdirs:
        return outpath, "outpath is empty"
    
    outpath = os.path.join(outpath, subdirs[0], "IMG_DATA")
    if not os.path.exists(outpath):
        return outpath, "outpath does not exist"
    
    required_dirs = ["R10m", "R20m", "R60m"]
    existing_dirs = [d for d in required_dirs if os.path.exists(os.path.join(outpath, d))]
    
    if not existing_dirs:
        return outpath, "outpath is empty"
    
    if len(existing_dirs) == 1:
        return outpath, f"the folder {existing_dirs[0]} is present"
    elif len(existing_dirs) == 2:
        return outpath, f"the folders {existing_dirs[0]} and {existing_dirs[1]} are present"
    else:
        return outpath, f"the folders {existing_dirs[0]}, {existing_dirs[1]} and {existing_dirs[2]} are present"

def get_pixel_values(raster_path, aoi_wkt):
    """Get pixel values from raster data within the specified AOI."""
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        aoi = wkt.loads(aoi_wkt)
        project = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:4326'), raster_crs, always_xy=True).transform
        aoi_transformed = transform(project, aoi)
        geoms = [aoi_transformed]
        mask = geometry_mask(geoms, transform=src.transform, invert=True, out_shape=(src.height, src.width))
        data = src.read(1)
        pixel_values = data[mask]
    
    return pixel_values

def get_pixel_values_with_expansion(raster_path, aoi_wkt, px_dim=10):
    """Get pixel values from raster data within the specified AOI, expanded by px_dim meters."""
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        aoi = wkt.loads(aoi_wkt)
        project = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:4326'), raster_crs, always_xy=True).transform
        aoi_transformed = transform(project, aoi)
        expanded_aoi = aoi_transformed.buffer(px_dim)
        geoms = [expanded_aoi]
        mask = geometry_mask(geoms, transform=src.transform, invert=True, out_shape=(src.height, src.width))
        data = src.read(1)
        pixel_values = data[mask]
    
    return pixel_values

def create_result_dict(path, url, item, aoi_wkt, cldprb_value):
    """Create a dictionary with result attributes."""
    result = {
        'ID': item[2],
        'link': url,
        'aoi': aoi_wkt,
        'ac_date': item[5].split('T')[0],
        'ac_time': item[5].split('T')[1],
        'cloud_prob': cldprb_value
    }
    return result

def add_band_to_result(result, path, aoi_wkt, band='B02', res='R10m'):
    """Add band information to the result dictionary."""
    outpath = os.path.join(path, res)
    band_file = None

    for file in os.listdir(outpath):
        if band in file:
            band_file = os.path.join(outpath, file)
            break

    if band_file:
        points = get_pixel_values(band_file, aoi_wkt)
        if len(points) < 4:
            px_dim = int(res[1:-1])
            points = get_pixel_values_with_expansion(band_file, aoi_wkt, px_dim)

        result[band] = points
        result[band + '_av'] = int(sum(points) / len(points))
    else:
        result[band] = None
        result[band + '_av'] = None

    return result

def create_new_path(img_data_path):
    """Create a new path based on the IMG_DATA path."""
    if not img_data_path.endswith('IMG_DATA'):
        print("The path does not end with 'IMG_DATA'.")
        return None

    return img_data_path.replace('IMG_DATA', 'QI_DATA/MSK_CLDPRB_20m.jp2')

def s2_band_resolution_list():
    """Return a list of band and resolution pairs for Sentinel-2."""
    return [
        ("B01", "R60m"),
        ("B02", "R10m"),
        ("B03", "R10m"),
        ("B04", "R10m"),
        ("B05", "R20m"),
        ("B06", "R20m"),
        ("B07", "R20m"),
        ("B08", "R10m"),
        ("B8A", "R20m"),
        ("B09", "R60m"),
        ("B11", "R20m"),
        ("B12", "R20m"),
    ]

def dict_to_dataframe(result, existing_df=None):
    """Convert a dictionary to a DataFrame and merge with an existing DataFrame if provided."""
    data = {}
    for key, value in result.items():
        if isinstance(value, (list, tuple)):
            data[key] = value
        else:
            data[key] = [value] * max(len(v) if isinstance(v, (list, tuple)) else 1 for v in result.values())

    temp_df = pd.DataFrame(data)
    temp_df.dropna(axis=1, how='all', inplace=True)

    if existing_df is not None:
        existing_df.dropna(axis=1, how='all', inplace=True)
        updated_df = pd.concat([existing_df, temp_df], ignore_index=True)
    else:
        updated_df = temp_df

    return updated_df

def export_to_csv(df, output_list, filename='output.csv', thd_cloud=50):
    """Export DataFrame to CSV, filtering by cloud probability threshold."""
    filename_pd_pkl = filename[:-4] + 'result_pd' + '.pkl'
    filename_ol_pkl = filename[:-4] + 'output_list_in' + '.pkl'
    df.to_pickle(filename_pd_pkl)
    with open(filename_ol_pkl, 'wb') as file:
        pickle.dump(output_list, file)
    
    columns_to_keep = [
        'ID', 'link', 'aoi', 'ac_date', 'ac_time', 'cloud_prob',
        'B01_av', 'B02_av', 'B03_av', 'B04_av', 'B05_av', 'B06_av',
        'B07_av', 'B08_av', 'B8A_av', 'B09_av', 'B11_av', 'B12_av'
    ]
    
    df_filtered = df[df['cloud_prob'] < thd_cloud]
    df_to_export = df_filtered[columns_to_keep].copy()
    df_to_export.to_csv(filename, index=False)
    print(f"CSV file saved as {filename}")
    
def delete_folder(folder_path):
    """Delete a folder and its contents."""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"The folder {folder_path} and all its contents have been deleted.")
    else:
        print(f"The folder {folder_path} does not exist.")
    
def S2processing_frompolygon(start_date, end_date, aoi_download, aoi_fp, results, data_collection= "SENTINEL-2", datatype = 'L2A', output_list=None, result_pd=None):
    """Process Sentinel-2 data from a polygon."""
    if output_list is None:
        output_list = fetch_satellite_data(start_date, end_date, data_collection, aoi_download)
        output_list = filter_output_list_by_datatype(output_list, datatype)
        
    output_list_in = output_list
    access_token = get_access_token("paolocosmo.silvestro@gmail.com", "Alessandro2023?")
    processed_items = []

    if result_pd is None: 
        columns = [
            'ID', 'link', 'aoi', 'ac_date', 'ac_time', 'cloud_prob', 'B01',
            'B01_av','B02', 'B02_av','B03', 'B03_av','B04', 'B04_av','B05', 'B05_av','B06', 'B06_av',
            'B07', 'B07_av','B08', 'B08_av','B8A', 'B8A_av','B09', 'B09_av','B11', 'B11_av','B12', 'B12_av'
        ]
        result_pd = pd.DataFrame(columns=columns)
        
    s2_resolution_pairs = s2_band_resolution_list()
    current_processed = []
    n_images = len(output_list)
    for idx, item in enumerate(output_list, start=1):
        print(f"Downloading image {idx} of {n_images}, ({item[2]})")
        output_name = item[2] + ".zip"
        url = generate_url(item[1])
        download_file(access_token, url, output_name)
        print(f" The image {idx} of {n_images} {item[2]} is downloaded")
        
        zip_path = output_name
        extract_to = os.getcwd()
        extract_folder = extract_and_remove_zip(zip_path, extract_to)
        
        print(f" Data extraction from {item[2]} is starting")
        if extract_folder is None:
            print(f"Skipping image {idx} due to extraction error")
            continue
        
        inputpath = extract_folder
        outpath_t, outcheck_t = check_and_update_path(inputpath)
        cloud_path = create_new_path(outpath_t)
        cldprb_points = get_pixel_values(cloud_path, aoi_fp)
        cldprb_value = sum(cldprb_points) / len(cldprb_points)
        result = create_result_dict(outpath_t, url, item, aoi_fp, cldprb_value)
        
        for band, res in s2_resolution_pairs:
            result = add_band_to_result(result, outpath_t, aoi_fp, band=band, res=res)
        
        results.append(result)
        temp_df = pd.DataFrame.from_dict([result])
        result_pd = pd.concat([result_pd, temp_df], ignore_index=True)
        
        current_processed.append(item)
        
        print(f" The processing of {item[2]} is finished")
        processed_items.extend(current_processed)
        delete_folder(inputpath)
        output_list = [item for item in output_list if item not in processed_items]
    

    return output_list_in, output_list, processed_items, result_pd, results
        
def main():
    """Main function to execute Sentinel-2 data processing."""
    start_date = "2020-09-30"
    end_date = "2020-10-31"
    data_collection = "SENTINEL-2"
    aoi_download = "POLYGON((-4.78305154699348 38.349247670976325, -4.78305154699348 38.37625826438388, -4.752389021711286 38.37625826438388, -4.752389021711286 38.349247670976325, -4.78305154699348 38.349247670976325))'"    
    datatype = 'L2A'
    aoi_fp = "POLYGON((-4.76764154430692 38.3618962903239, -4.76764154430692 38.3621075912977, -4.7673322500241 38.3621075912977, -4.7673322500241 38.3618962903239, -4.76764154430692 38.3618962903239))"
    csv_name = "MOR_" + start_date + end_date +'.csv'

    results = list()
    output_list_in, output_list, processed_items, result_pd, results = S2processing_frompolygon(
        start_date, end_date, aoi_download, aoi_fp, results, data_collection=data_collection, datatype=datatype, output_list=None, result_pd=None
    )
    
    output_list_in2 = None
    while len(output_list) != 0:
        output_list_in2, output_list, processed_items, result_pd, results = S2processing_frompolygon(
            start_date, end_date, aoi_download, aoi_fp, results, data_collection=data_collection, datatype=datatype, output_list=output_list, result_pd=result_pd
        )
    export_to_csv(result_pd, output_list_in ,filename=csv_name, thd_cloud=50)
    return output_list_in, output_list_in2, output_list, processed_items, result_pd, results

if __name__ == "__main__":
    output_list_in, output_list_in2, output_list, processed_items, result_pd, results = main()
    # print(output_list_in, output_list_in2, output_list, processed_items, result_pd, results)
    print("The end of the script")
# output_list, processed_items, result_pd = S2processing_frompolygon(start_date, end_date, aoi_download, aoi_fp, data_collection= "SENTINEL-2", datatype = 'L2A' )
