import os
import requests
import shutil

# URL to a single Spleen CT file (approx 17MB) provided by Project MONAI test data
URL = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/spleen_19.nii.gz"
SAVE_PATH = "data/dummy_ct.nii.gz" # We overwrite the dummy so we don't have to change paths later

def download_file():
    print(f"Downloading Real CT Scan from: {URL}")
    print("Please wait... (~17 MB)")
    
    os.makedirs("data", exist_ok=True)
    
    response = requests.get(URL, stream=True)
    if response.status_code == 200:
        with open(SAVE_PATH, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        print("\n✅ Success! Real patient data saved.")
        print(f"   Location: {SAVE_PATH}")
    else:
        print("❌ Failed to download. Check internet connection.")

if __name__ == "__main__":
    download_file()