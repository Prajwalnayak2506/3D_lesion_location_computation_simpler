import nibabel as nib
import numpy as np
import os

def create_dummy_ct():
    print("Generating lightweight dummy CT scan...")
    
    # 1. Create a small 3D volume (64x64x64) 
    # Real CTs are 512x512x512 (too big for your RAM right now)
    # We use random noise to simulate 'tissue'
    fake_vol = np.random.rand(64, 64, 64).astype(np.float32)
    
    # 2. Add a 'fake tumor' (a bright spot in the middle)
    # Lesions in CT are often brighter/different intensity
    fake_vol[30:40, 30:40, 30:40] += 2.0 
    
    # 3. Save as NIfTI (.nii.gz) - The standard medical format
    # We use an identity affine (maps pixels to physical space 1:1)
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(fake_vol, affine)
    
    os.makedirs("data", exist_ok=True)
    save_path = "data/dummy_ct.nii.gz"
    nib.save(nifti_img, save_path)
    
    print(f"✅ Success! Created {save_path}")
    print(f"   Size: {os.path.getsize(save_path) / 1024:.2f} KB (Tiny!)")

if __name__ == "__main__":
    create_dummy_ct()