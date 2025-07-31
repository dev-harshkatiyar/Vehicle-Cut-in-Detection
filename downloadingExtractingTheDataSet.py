import requests
import zipfile
import os

def download_idd_temporal(url, save_path):
    # Download the file from the URL
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    # Save the downloaded file to the specified path
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            file.write(data)
        print(f"Downloaded {save_path}")

def extract_zip(file_path, extract_to):
    # Extract the zip file
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted {file_path} to {extract_to}")

# Define the URL for the IDD Temporal dataset
idd_temporal_url = "http://idd.insaan.iiit.ac.in/IDD_Temporal.zip"

# Define the path to save the downloaded file and the extraction path
save_path = "IDD_Temporal.zip"
extract_to = "IDD_Temporal"

# Create the extraction directory if it doesn't exist
if not os.path.exists(extract_to):
    os.makedirs(extract_to)

# Download and extract the dataset
download_idd_temporal(idd_temporal_url, save_path)
extract_zip(save_path, extract_to)