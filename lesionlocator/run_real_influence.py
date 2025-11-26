import torch
import nibabel as nib
import numpy as np
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from lesionlocator.inference.lesionlocator_segment import LesionLocatorSegmenter

# --- CONFIGURATION ---
PATCH_SIZE = 64 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODEL_FOLDER = os.path.join(BASE_DIR, "weights", "LesionLocatorCheckpoint", "LesionLocatorSeg", "point_optimized")
# Auto-fix path if nested
if not os.path.exists(MODEL_FOLDER):
    MODEL_FOLDER = os.path.join(os.path.dirname(BASE_DIR), "weights", "LesionLocatorCheckpoint", "LesionLocatorSeg", "point_optimized")

INPUT_PATH = os.path.join(BASE_DIR, "data", "CT_Abdo.nii.gz") 

# Experiment Points (Z, Y, X)
CLICKS_TO_TEST = [
    (32, 32, 32),  # Center
    (32, 20, 20),  # Top-Left
    (32, 45, 45),  # Bottom-Right
    (20, 20, 32),  # Front-Center
]

def create_ball_prompt(shape, center, radius=3):
    """
    Implements the 'Ball Region' prompt from the LesionLocator paper.
    A single pixel is too small; we need a sphere of 1s.
    """
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist_from_center = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    
    mask = dist_from_center <= radius
    return mask.astype(np.float32)

def run_presentation_demo():
    print("\n--- LESION LOCATOR: BALL PROMPT EXPERIMENT ---")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. Initialize Engine
    print("[1/3] Loading Engine...")
    segmenter = LesionLocatorSegmenter(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False, 
        perform_everything_on_device=False, device=torch.device('cpu'), verbose=False
    )
    try:
        segmenter.initialize_from_trained_model_folder(MODEL_FOLDER, use_folds=[0])
    except Exception as e:
        print(f"⚠️ Weights Error: {e}")
        return

    # 2. Load Data
    if not os.path.exists(INPUT_PATH):
        print("❌ Data missing. Run make_phantom_data.py first.")
        return
    
    nifti_img = nib.load(INPUT_PATH)
    full_vol = nifti_img.get_fdata()
    
    # Center Crop to 64x64x64
    x_c, y_c, z_c = np.array(full_vol.shape) // 2
    half = PATCH_SIZE // 2
    # Safety bounds
    patch_np = np.zeros((PATCH_SIZE, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    
    # Simple crop logic (assuming input > 64)
    sl_x = slice(x_c-half, x_c+half)
    sl_y = slice(y_c-half, y_c+half)
    sl_z = slice(z_c-half, z_c+half)
    patch_np = full_vol[sl_x, sl_y, sl_z]
    
    # Normalize
    patch_np = (patch_np - np.mean(patch_np)) / (np.std(patch_np) + 1e-8)
    image_tensor = torch.from_numpy(patch_np).float().unsqueeze(0).unsqueeze(0)

    # 3. Run Multi-Click Loop
    print(f"[2/3] Testing {len(CLICKS_TO_TEST)} prompts with Radius=3...")
    
    segmenter.network.eval()
    
    for i, click in enumerate(CLICKS_TO_TEST):
        print(f"   -> Click {i+1}: {click}")
        
        # A. Create BALL Prompt (Not single pixel)
        prompt_data = create_ball_prompt(patch_np.shape, click, radius=4)
        prompt_tensor = torch.from_numpy(prompt_data).float().unsqueeze(0).unsqueeze(0)
        
        # B. Combine
        input_tensor = torch.cat([image_tensor, prompt_tensor], dim=1)
        
        # C. Inference
        with torch.no_grad():
            output = segmenter.network(input_tensor)
            if isinstance(output, (tuple, list)): output = output[0]
            
        # D. Visualization
        tumor_prob = torch.sigmoid(output[0, 1]).cpu().numpy()
        
        # Enhance contrast for presentation (auto-scaling)
        viz_heatmap = (tumor_prob - tumor_prob.min()) / (tumor_prob.max() - tumor_prob.min() + 1e-8)
        
        plt.figure(figsize=(10, 5))
        
        # Plot Input + Click
        plt.subplot(1, 2, 1)
        plt.title(f"Prompt {i+1} (Ball Radius=4)")
        plt.imshow(np.rot90(patch_np[32, :, :]), cmap="gray")
        # Show the actual prompt region, not just a cross
        prompt_slice = np.rot90(prompt_data[32, :, :])
        plt.imshow(prompt_slice, cmap="Reds", alpha=0.4) 
        plt.axis("off")
        
        # Plot Output
        plt.subplot(1, 2, 2)
        plt.title("Model Segmentation")
        plt.imshow(np.rot90(patch_np[32, :, :]), cmap="gray", alpha=0.5)
        plt.imshow(np.rot90(viz_heatmap[32, :, :]), cmap="jet", alpha=0.6)
        plt.axis("off")
        
        save_path = os.path.join(PLOTS_DIR, f"demo_click_{i+1}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"      Saved: {save_path}")

    print("\n✅ DONE. Check the 'plots' folder for the comparison.")

if __name__ == "__main__":
    run_presentation_demo()