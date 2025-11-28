Created on Wed Nov 20 2024
Last update on Fri Nov 28 2025
@project: ScaleAgData
@author: paolo cosmo silvestro
"""

import requests
import pandas as pd
import os
import zipfile
import rasterio
from shapely import wkt
from rasterio.features import geometry_mask
import numpy as np
import subprocess
import shutil
import pickle
from urllib.parse import quote


def verify_and_find_files(raster_path):
    # List the files in the raster_path directory
    files = os.listdir(raster_path)
    
    # Verify that there are exactly 2 files in the directory
    if len(files) != 2:
        raise ValueError("The directory must contain exactly 2 files.")
    
    # Initialize the paths of the files to return
    vvraster_path = None
    vhraster_path = None
    
    # Search for files containing 'grd-vh' and 'grd-vv' in their names
    for file in files:
        if 'grd-vv' in file.lower():
            vvraster_path = os.path.join(raster_path, file)
        elif 'grd-vh' in file.lower():
            vhraster_path = os.path.join(raster_path, file)
    
    # Verify that both files were found
    if not vvraster_path or not vhraster_path:
        raise ValueError("Files containing 'grd-vv' and 'grd-vh' were not found.")
    
    return vvraster_path, vhraster_path

def verify_raster_dimensions(vvraster_path, vhraster_path):
    with rasterio.open(vvraster_path) as vv_raster:
        vv_height = vv_raster.height
        vv_width = vv_raster.width
    
    with rasterio.open(vhraster_path) as vh_raster:
        vh_height = vh_raster.height
        vh_width = vh_raster.width
    
    if vv_height == vh_height and vv_width == vh_width:
        return vv_height, vv_width
    else:
        raise ValueError("The rasters do not have the same dimensions.")

def modify_xml(graph_path, xml_name, input_name, vv_height, vv_width, output_name):
    """
    Modify an XML file by replacing specific lines based on input parameters.

    Args:
        graph_path (str): Path to the input XML file.
        xml_name (str): Path to save the modified XML file.
        input_name (str): New value for the input file line.
        vv_height (int): New height value for the pixelRegion line.
        vv_width (int): New width value for the pixelRegion line.
        output_name (str): New value for the output file line.
    """
    with open(graph_path, 'r') as file:
        lines = file.readlines()

    # Modify specific lines
    lines[7] = f"      <file>{input_name}</file>\n"
    lines[10] = f"      <pixelRegion>0,0,{vv_width},{vv_height}</pixelRegion>\n"
    lines[126] = f"      <file>{output_name}</file>\n"

    # Save the modified XML
    with open(xml_name, 'w') as file:
        file.writelines(lines)
        
def execute_snap_graph(xml_name):
    """
    Execute a SNAP graph using the gpt command-line tool.

    Args:
        xml_name (str): Path to the SNAP XML graph file.
    """
    try:
        # Run the SNAP gpt tool with the XML graph file
        result = subprocess.run(['gpt', xml_name], check=True, capture_output=True, text=True)
        print("SNAP graph executed successfully.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error executing SNAP graph:")
        print(e.stderr)

def S1_GRD22horto(S1_path, graph_path):
    """
    Process Sentinel-1 data by modifying an XML graph and executing it with SNAP.

    Args:
        S1_path (str): Path to the Sentinel-1 data.
        graph_path (str): Path to the input XML graph file.
    """
    
    try:
        # Verify and find the raster files
        print("Verifying and finding raster files...")
        raster_path = os.path.join(S1_path, 'measurement')
        vvraster_path, vhraster_path = verify_and_find_files(raster_path)
        print(f"Found VV raster: {vvraster_path}")
        print(f"Found VH raster: {vhraster_path}")
        
        # Verify raster dimensions
        print("Verifying raster dimensions...")
        vv_height, vv_width = verify_raster_dimensions(vvraster_path, vhraster_path)
        print(f"Raster dimensions - Height: {vv_height}, Width: {vv_width}")
        
        # Modify the XML graph
        print("Modifying the XML graph...")
        input_name = os.path.join(S1_path, 'manifest.safe')
        output_name = S1_path[:-5] + '_Orb_Cal_Spk_TC'
        xml_name = S1_path[:-5] + '_test.xml'
        modify_xml(graph_path, xml_name, input_name, vv_height, vv_width, output_name)
        print(f"Modified XML saved as: {xml_name}")
        
        # Execute the SNAP graph
        print("Executing the SNAP graph...")
        execute_snap_graph(xml_name)
        output_name = output_name + '.data'
    except Exception as e:
        print(f"Error during processing: {e}")
        output_name = []
    
    return output_name

def vv_vvh(output_name):
    """
    Return the paths of all .img files in the given directory.
    
    Args:
        output_name (str): Path to the directory to search for .img files.
    
    Returns:
        list: List of paths to .img files.
    """
    img_files = []
    
    # Traverse the directory and collect all .img file paths
    for root, _, files in os.walk(output_name):
        for file in files:
            if file.endswith('.img'):
                img_files.append(os.path.join(root, file))
    
    return img_files

def pixels_in_aoi(raster_path, aoi_wkt):
    """
    Given a raster path and an AOI in WKT format, return a vector of pixels included in the AOI.

    Args:
        raster_path (str): Path to the raster file.
        aoi_wkt (str): AOI in WKT format.

    Returns:
        np.ndarray: A vector of pixel values included in the AOI.
    """
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Read the entire raster data
        raster_data = src.read(1)
        
        # Convert the AOI WKT to a shapely geometry
        aoi_geom = wkt.loads(aoi_wkt)
        
        # Create a mask for the AOI
        mask = geometry_mask([aoi_geom], transform=src.transform, invert=True, out_shape=src.shape)
        
        # Extract pixels within the AOI
        pixels = raster_data[mask]
    
    return pixels

def vv_vh_values(output_name, aoi_wkt):
    """
    Given an output directory and an AOI in WKT format, return the pixel values for VH and VV raster files within the AOI.

    Args:
        output_name (str): Path to the output directory containing the raster files.
        aoi_wkt (str): AOI in WKT format.

    Returns:
        tuple: Two numpy arrays containing the pixel values for VH and VV rasters within the AOI.
    """
    # Get the paths of all .img files in the output directory
    img_files = vv_vvh(output_name)
    
    # Verify that there are at least two files
    if len(img_files) < 2:
        pixels_vh = 0
        pixels_vv = 0
    else: 
        # Assume the first .img file is VH and the second is VV
        vh_path = img_files[0]
        vv_path = img_files[1]
        # Extract pixels within the AOI for both VH and VV rasters
        pixels_vh = pixels_in_aoi(vh_path, aoi_wkt)
        pixels_vv = pixels_in_aoi(vv_path, aoi_wkt)
    
    return pixels_vh, pixels_vv
def fetch_satellite_data(start_date, end_date, data_collection, aoi):
    aoi_encoded = quote(f"SRID=4326;{aoi}")
    
    url = (
        "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        f"?$filter=Collection/Name eq '{data_collection}' and "
        f"OData.CSC.Intersects(area=geography'{aoi_encoded}') and "
        f"ContentDate/Start gt {start_date}T00:00:00.000Z and "
        f"ContentDate/Start lt {end_date}T00:00:00.000Z"
    )

    response = requests.get(url)
    
    try:
        json_data = response.json()
        output_list = pd.DataFrame.from_dict(json_data["value"]).head(50).values.tolist()
    except KeyError:
        print("Errore: 'value' non trovato nella risposta JSON.")
        print("Risposta completa:", response.text)
        output_list = []
    
    return output_list
# def fetch_satellite_data(start_date, end_date, data_collection, aoi):
#     response = requests.get(
#         f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq '{data_collection}' and OData.CSC.Intersects(area=geography'SRID=4326;{aoi}) and ContentDate/Start gt {start_date}T00:00:00.000Z and ContentDate/Start lt {end_date}T00:00:00.000Z"
#     )
    
#     try:
#         json_data = response.json()
#         output_list = pd.DataFrame.from_dict(json_data["value"]).head(50).values.tolist()
#     except KeyError:
#         print("Errore: 'value' non trovato nella risposta JSON.")
#         output_list = []
    
#     return output_list

def filter_output_list_by_datatype(output_list, datatype='IW_GRDH'):
    """Filter the output list by datatype."""
    return [item for item in output_list if datatype in item[2]]

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

def download_file(access_token, item , output_name):
    """Download file using access token and URL."""
    url = generate_url(item[1])
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
    outpath = os.path.join(inputpath, "measurement")
    if not os.path.exists(outpath):
        return outpath, "outpath does not exist"
    
    subdirs = [d for d in os.listdir(outpath) if os.path.isdir(os.path.join(outpath, d))]
    if not subdirs:
        return outpath, "outpath is empty"
    
def create_result_dict(path, url, item, aoi_wkt):
    """Create a dictionary with result attributes."""
    result = {
        'ID': item[2],
        'link': url,
        'aoi': aoi_wkt,
        'ac_date': item[5].split('T')[0],
        'ac_time': item[5].split('T')[1],
        
    }
    return result

def modify_path(original_path):
    # Remove the 'measurement' directory from the path
    base_path = os.path.dirname(original_path)
    
    # Add '\preview\map-overlay.kml' to the path
    new_path = os.path.join(base_path, 'preview', 'map-overlay.kml')
    
    # Check if the new path is an existing file
    if os.path.isfile(new_path):
        print(f"The file {new_path} exists.")
    else:
        print(f"The file {new_path} does not exist.")
    
    return new_path

def delete_folder(folder_path):
    """Delete a folder and its contents."""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"The folder {folder_path} and all its contents have been deleted.")
    else:
        print(f"The folder {folder_path} does not exist.")

def export_to_csv(df, output_list, filename='output.csv', thd_cloud=0):
    """Export DataFrame to CSV, filtering by cloud probability threshold."""
    filename_pd_pkl = filename[:-4] + 'result_pd' + '.pkl'
    filename_ol_pkl = filename[:-4] + 'output_list_in' + '.pkl'
    df.to_pickle(filename_pd_pkl)
    with open(filename_ol_pkl, 'wb') as file:
        pickle.dump(output_list, file)
    
    columns_to_keep = [
        'ID', 'link', 'aoi', 'ac_date', 'ac_time', 'cloud_prob',
        'vh_av', 'vv_av'
    ]
    
    
    df_to_export = df[columns_to_keep].copy()
    df_to_export.to_csv(filename, index=False)
    print(f"CSV file saved as {filename}")
        

def S1processing_frompolygon(start_date, end_date, aoi_download, aoi_fp, results,csv_name, graph_path = 'S1_manual.xml', data_collection="SENTINEL-1", datatype='IW_GRDH', output_list=None, result_pd=None):
    """Process Sentinel-1 data from a polygon."""
    if output_list is None:
        output_list = fetch_satellite_data(start_date, end_date, data_collection, aoi_download)
        output_list = filter_output_list_by_datatype(output_list, datatype)
        
    output_list_in = output_list
    access_token = get_access_token("paolocosmo.silvestro@gmail.com", "Alessandro2025!")
    processed_items = []

    if result_pd is None: 
        columns = [
            'ID', 'link', 'aoi', 'ac_date', 'ac_time', 'cloud_prob', 'vv', 'vv_av', 'vh', 'vh_av'
        ]
        result_pd = pd.DataFrame(columns=columns)
        
    current_processed = []
    n_images = len(output_list)
    for idx, item in enumerate(output_list, start=1):
        print(f"Downloading image {idx} of {n_images}, ({item[2]})")
        output_name = item[2] + ".zip"
        url = generate_url(item[1])
        download_file(access_token, item, output_name)
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
        S1_path = os.path.dirname(outpath_t)
        tmp_output_name = S1_GRD22horto(S1_path, graph_path)
        if tmp_output_name == []:
            pixels_vh = 0
            pixels_vv = 0
        else:      
            pixels_vh, pixels_vv = vv_vh_values(tmp_output_name, aoi_fp)
                
        result = create_result_dict(outpath_t, url, item, aoi_fp)
        result['vh'] = pixels_vh
        result['vh_av'] = np.mean(pixels_vh)
        
        result['vv'] = pixels_vv
        result['vv_av'] = np.mean(pixels_vv)
        
        results.append(result)
        temp_df = pd.DataFrame.from_dict([result])
        result_pd = pd.concat([result_pd, temp_df], ignore_index=True)
        
        current_processed.append(item)
        export_to_csv(result_pd, output_list_in ,filename=csv_name, thd_cloud=0)
        print(f" The processing of {item[2]} is finished")
        processed_items.extend(current_processed)
        delete_folder(inputpath)
        output_list = [item for item in output_list if item not in processed_items]

    return output_list_in, output_list, processed_items, result_pd, results

def main(start_date, end_date, data_collection, aoi_download, datatype, aoi_fp, csv_name):
    """Main function to execute Sentinel-2 data processing."""
    print(csv_name)

    results = list()
    output_list_in, output_list, processed_items, result_pd, results = S1processing_frompolygon(
        start_date, end_date, aoi_download, aoi_fp, results, csv_name, data_collection=data_collection, datatype=datatype, output_list=None, result_pd=None
    )
    
    output_list_in2 = None
    while len(output_list) != 0:
        output_list_in2, output_list, processed_items, result_pd, results = S1processing_frompolygon(
            start_date, end_date, aoi_download, aoi_fp, results,csv_name, data_collection=data_collection, datatype=datatype, output_list=output_list, result_pd=result_pd
        )
    export_to_csv(result_pd, output_list_in ,filename=csv_name, thd_cloud=0)
    return output_list_in, output_list_in2, output_list, processed_items, result_pd, results

if __name__ == "__main__":
    start_date = "2018-01-21"
    end_date = "2018-01-25"
    data_collection = "SENTINEL-1"
    aoi_download = "POLYGON((-4.307525 38.22363, -4.307525 38.19661, -4.272885 38.19661, -4.272885 38.22363, -4.307525 38.22363))"
    datatype = 'IW_GRDH'
    aoi_fp = "POLYGON((-4.29036 38.21015, -4.29005 38.21015, -4.29005 38.21009, -4.29036 38.21009 ,  -4.29036 38.21015))"
    csv_name = "STC_" + start_date + end_date +'.csv'
    graph_path = 'S1_manual.xml'
    print(csv_name)
    output_list_in, output_list_in2, output_list, processed_items, result_pd, results = main(start_date, end_date, data_collection, aoi_download, datatype, aoi_fp, csv_name)
    print("The end of the script")
#     download_file(access_token, item , output_name)
