import torch
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.ndimage
from lesionlocator.inference.lesionlocator_segment import LesionLocatorSegmenter

# --- CONFIGURATION ---
PATCH_SIZE = 64 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODEL_FOLDER = os.path.join(BASE_DIR, "weights", "LesionLocatorCheckpoint", "LesionLocatorSeg", "point_optimized")
if not os.path.exists(MODEL_FOLDER):
    MODEL_FOLDER = os.path.join(os.path.dirname(BASE_DIR), "weights", "LesionLocatorCheckpoint", "LesionLocatorSeg", "point_optimized")

# --- DATA GENERATION FUNCTIONS (For Longitudinal) ---

def generate_longitudinal_phantom(tumor_growth_factor=1.2, translation=(3, 3, 3)):
    """Generates two related time points (Baseline and Follow-Up) for tracking."""
    
    # 1. BASELINE (Time T)
    vol_base = np.ones((64, 64, 64), dtype=np.float32) * -1000 
    
    # Organ structure (Irregular Blob)
    organ_mask = np.zeros((64, 64, 64))
    organ_mask[10:54, 10:54, 10:54] = 1
    organ_mask = scipy.ndimage.gaussian_filter(organ_mask, sigma=4) > 0.3
    vol_base[organ_mask] = 50.0 + np.random.normal(0, 10, size=organ_mask.sum())
    
    # Tumor Mask (The ground truth lesion)
    tumor_mask_base = np.zeros((64, 64, 64))
    tumor_mask_base[30:35, 30:35, 30:35] = 1 # 5x5x5 lesion
    tumor_mask_base = scipy.ndimage.gaussian_filter(tumor_mask_base, sigma=1)
    
    # Inject Tumor Signal (High contrast)
    vol_base[tumor_mask_base > 0.1] = 2000.0 
    
    # Normalize and copy for Follow-Up
    vol_base = (vol_base - np.mean(vol_base)) / (np.std(vol_base) + 1e-8)
    
    # 2. FOLLOW-UP (Time T+1)
    
    # Start with baseline structure
    vol_followup = vol_base.copy() 
    
    # Simulate Growth/Change (Grow the tumor mask, shift the whole image slightly)
    
    # A. Lesion Change (Growth)
    tumor_mask_followup = np.zeros((64, 64, 64))
    tumor_mask_followup[30:int(35*tumor_growth_factor), 30:int(35*tumor_growth_factor), 30:int(35*tumor_growth_factor)] = 1
    tumor_mask_followup = scipy.ndimage.gaussian_filter(tumor_mask_followup, sigma=1)
    
    # Inject NEW tumor signal (1.2x size)
    # Reset center of follow-up tumor to avoid artifacts from base tumor
    vol_followup[tumor_mask_base > 0.1] = (vol_followup[tumor_mask_base > 0.1] * 0.1) 
    vol_followup[tumor_mask_followup > 0.1] = 2000.0 
    
    # B. Patient Misalignment (Slight spatial shift)
    vol_followup = scipy.ndimage.shift(vol_followup, translation, mode='nearest')
    
    # Re-normalize Follow-Up
    vol_followup = (vol_followup - np.mean(vol_followup)) / (np.std(vol_followup) + 1e-8)

    return vol_base, vol_followup, tumor_mask_base 


# --- MAIN EXECUTION LOGIC ---

def run_tracking_demo(model):
    print("\n--- RUNNING 4D TRACKING SIMULATION ---")
    
    # Generate Longitudinal Data
    vol_base, vol_followup, gt_mask_base = generate_longitudinal_phantom()
    
    # --- BASELINE (I_t) INFERENCE ---
    
    # 1. Create Baseline Prompt (Click on the baseline lesion)
    center = 32
    image_tensor_base = torch.from_numpy(vol_base).float().unsqueeze(0).unsqueeze(0)
    prompt_tensor = torch.zeros_like(image_tensor_base)
    prompt_tensor[0, 0, center, center, center] = 1.0 # Click center of the initial tumor
    input_tensor_base = torch.cat([image_tensor_base, prompt_tensor], dim=1)

    # 2. Segment Baseline (This gives the segmented mask M_t)
    model.network.eval()
    with torch.no_grad():
        # NOTE: output_base is a tuple/list, we take the main tensor [0]
        output_base = model.network(input_tensor_base)
        if isinstance(output_base, (tuple, list)):
            output_base = output_base[0]
    
    # M_t: The model's prediction for the baseline tumor
    mask_base = (torch.sigmoid(output_base[0, 1]).cpu().numpy() > 0.5) 
    print("✅ Baseline Segmentation (M_t) complete.")

    # --- FOLLOW-UP (I_t+1) TRACKING ---
    
    # 3. Create Follow-Up Prompt (M_t used as prompt for I_t+1)
    
    # Create Follow-Up Image Tensor [B, C=1, Z, Y, X]
    image_tensor_followup = torch.from_numpy(vol_followup).float().unsqueeze(0).unsqueeze(0)

    # 🛑 THE FIX: Convert 3D NumPy mask to 5D PyTorch tensor [B, C, Z, Y, X]
    # 1. Convert to float tensor
    mask_base_tensor = torch.from_numpy(mask_base.astype(np.float32)).float()
    
    # 2. Reshape to 5D: [1, 1, Z, Y, X]
    mask_base_tensor = mask_base_tensor.reshape(1, 1, *mask_base_tensor.shape) 

    # Combine M_t as prompt with I_t+1 (Now both are 5D)
    input_tensor_followup = torch.cat([image_tensor_followup, mask_base_tensor], dim=1)


    # 4. Segment Follow-Up
    with torch.no_grad():
        output_followup = model.network(input_tensor_followup)
        if isinstance(output_followup, (tuple, list)):
            output_followup = output_followup[0]
        
    mask_followup_prob = torch.sigmoid(output_followup[0, 1]).cpu().numpy()
    print("✅ Follow-Up Tracking complete.")

    # --- VISUALIZATION (Plotting) ---

    mid_slice = 32
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    center = 32 # Center of patch
    
    # A. Baseline Segmentation (Time T)
    axes[0, 0].set_title("Time T: Baseline Scan", fontsize=14)
    axes[0, 0].imshow(np.rot90(vol_base[mid_slice, :, :]), cmap="gray")
    axes[0, 0].scatter([center], [64-center], c='red', marker='x', s=150, linewidth=3)
    axes[0, 0].axis("off")

    axes[0, 1].set_title("Prediction: Baseline (M_t)", fontsize=14)
    axes[0, 1].imshow(np.rot90(vol_base[mid_slice, :, :]), cmap="gray", alpha=0.5)
    axes[0, 1].imshow(np.rot90(mask_base[mid_slice, :, :]), cmap="jet", alpha=0.6)
    axes[0, 1].axis("off")


    # B. Follow-Up Tracking (Time T+1)
    axes[1, 0].set_title("Time T+1: Follow-Up Scan (Shifted/Grown)", fontsize=14)
    axes[1, 0].imshow(np.rot90(vol_followup[mid_slice, :, :]), cmap="gray")
    axes[1, 0].axis("off")

    axes[1, 1].set_title("Tracking Result (M_t+1)", fontsize=14)
    axes[1, 1].imshow(np.rot90(vol_followup[mid_slice, :, :]), cmap="gray", alpha=0.5)
    # Show the probability output for the final result
    axes[1, 1].imshow(np.rot90(mask_followup_prob[mid_slice, :, :]), cmap="jet", alpha=0.6) 
    axes[1, 1].axis("off")


    plt.suptitle("4D TRACKING DEMONSTRATION: LESION GROWTH & MISALIGNMENT", fontsize=16, fontweight='bold')
    
    # Save the 4-panel image
    save_path = os.path.join(PLOTS_DIR, "4D_Tracking_Demo.png")
    plt.savefig(save_path)
    plt.close()
    print(f"\n✅ Final 4D Tracking Slide Generated: {save_path}")

    # For quick view, display the result
    plt.imshow(plt.imread(save_path))
    plt.show()

# --- END run_tracking_demo ---

# --- MAIN EXECUTION LOGIC ---

def run_main_demo():
    print("\n--- INITIALIZING 4D DEPLOYMENT ---")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. Load Engine ONCE
    segmenter = LesionLocatorSegmenter(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False, 
        perform_everything_on_device=False, device=torch.device('cpu'), verbose=False
    )
    try:
        # Load the Segmentation weights (which contains the required U-Net for processing)
        segmenter.initialize_from_trained_model_folder(MODEL_FOLDER, use_folds=[0])
        print("✅ Engine Loaded.")
    except Exception as e:
        print(f"⚠️ Weights Error: {e}")
        return

    # 2. Run the tracking simulation
    run_tracking_demo(segmenter)


if __name__ == "__main__":
    run_main_demo()