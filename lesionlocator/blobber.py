import nibabel as nib
import numpy as np
import scipy.ndimage
import os

def create_phantom():
    print("Generating 'Medical Phantom' CT Scan...")
    
    # 1. Setup Volume (64x64x64)
    # Background = -1000 (Air in CT Hounsfield Units)
    vol = np.ones((64, 64, 64), dtype=np.float32) * -1000
    
    # 2. Create the "Body" (A large cylinder)
    # Tissue = 0 to 50 HU
    Y, X = np.ogrid[:64, :64]
    dist_from_center = np.sqrt((X - 32)**2 + (Y - 32)**2)
    body_mask = dist_from_center <= 28
    # Propagate along Z axis
    for z in range(64):
        vol[z, body_mask] = 40.0 + np.random.normal(0, 5, size=body_mask.sum()) # Tissue + slight noise

    # 3. Create the "Tumor" (A bright, soft blob)
    # Tumors with contrast = 100 to 150 HU
    # We make it a Gaussian blob so it looks "organic" to the AI
    tumor_center = (32, 32, 32)
    
    # Create a dot
    vol[32, 32, 32] = 500.0 
    
    # Blur the dot to make a ball (Gaussian Filter)
    # This creates smooth gradients which U-Nets love
    tumor_structure = np.zeros_like(vol)
    tumor_structure[28:36, 28:36, 28:36] = 200.0 # Box
    tumor_structure = scipy.ndimage.gaussian_filter(tumor_structure, sigma=2)
    
    # Add tumor to body
    vol = vol + tumor_structure
    
    # 4. Normalize (Critical for Deep Learning)
    # The model expects values roughly between -1 and 1, or 0 and 1
    vol = (vol - np.mean(vol)) / (np.std(vol) + 1e-8)

    # Save
    os.makedirs("data", exist_ok=True)
    save_path = "data/dummy_ct.nii.gz" # Overwrite the old one
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(vol, affine)
    nib.save(nifti_img, save_path)
    
    print(f"✅ Created 'Bio-Mimic' Phantom at {save_path}")

if __name__ == "__main__":
    create_phantom()