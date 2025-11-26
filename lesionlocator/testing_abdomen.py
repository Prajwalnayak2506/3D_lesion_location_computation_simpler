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

INPUT_PATH = os.path.join(BASE_DIR, "data", "CT_Abdo.nii.gz") 

# Experiment Points (Z, Y, X) relative to 64x64x64 patch
CLICKS_TO_TEST = [
    # Original Diagnostic Points (For reference)
    (32, 32, 32), # Center (Soft Tissue/Gut)
    (32, 50, 32), # Bottom Center (Spine/Bone area)
    (32, 20, 25), # Top Left (Likely Organ area)
    
    # 100 Randomly Generated Points for Robustness Testing
    (14, 21, 20), (45, 12, 10), (32, 10, 48), (28, 41, 14), (54, 30, 49),
    (46, 52, 30), (49, 39, 44), (22, 17, 31), (39, 48, 20), (41, 25, 41),
    (24, 27, 49), (42, 45, 27), (35, 23, 24), (29, 38, 17), (43, 49, 13),
    (17, 18, 28), (44, 21, 25), (10, 34, 46), (36, 15, 36), (40, 22, 52),
    (31, 51, 18), (12, 11, 40), (25, 33, 47), (19, 53, 37), (50, 26, 16),
    (53, 40, 50), (18, 28, 54), (11, 46, 11), (37, 13, 39), (47, 43, 29),
    (52, 37, 42), (20, 16, 51), (38, 54, 43), (33, 47, 15), (13, 29, 22),
    (26, 35, 12), (34, 44, 26), (48, 14, 53), (21, 50, 23), (15, 20, 45),
    (51, 36, 19), (27, 42, 35), (23, 19, 46), (54, 49, 17), (46, 31, 13),
    (14, 41, 54), (32, 18, 12), (45, 42, 38), (28, 24, 21), (50, 47, 40),
    (39, 39, 34), (41, 27, 16), (24, 51, 33), (42, 15, 47), (35, 20, 29),
    (29, 34, 11), (43, 22, 52), (17, 44, 39), (44, 29, 15), (10, 48, 41),
    (36, 37, 36), (40, 33, 23), (31, 17, 50), (12, 49, 28), (25, 21, 19),
    (19, 36, 53), (53, 43, 49), (18, 26, 14), (11, 30, 25), (37, 52, 16),
    (47, 24, 45), (52, 38, 29), (20, 13, 26), (38, 35, 40), (33, 46, 18),
    (13, 25, 54), (26, 16, 42), (34, 41, 23), (48, 29, 38), (21, 14, 49),
    (15, 33, 10), (51, 31, 43), (27, 40, 28), (23, 11, 17), (49, 53, 31),
    (22, 23, 41), (46, 28, 17), (30, 11, 40), (45, 37, 24), (16, 46, 11),
    (28, 20, 52), (54, 48, 35), (46, 34, 19), (49, 29, 44), (22, 13, 30)
]

# --- VISUALIZATION FUNCTION (OVERHAULED FOR 3D) ---

def create_ball_prompt(shape, center, radius=4):
    """Creates the 3D numpy array for the prompt sphere."""
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist_from_center = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    mask = dist_from_center <= radius
    return mask.astype(np.float32)

def save_figure(img_vol_np, pred_vol_np, prompt_vol_np, click_pos, idx):
    """Generates a 3x2 grid showing Axial, Coronal, and Sagittal views."""
    
    # 1. Setup Grid and Views
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))
    cz, cy, cx = click_pos
    
    # Define slices and their rotational adjustments (Z=Axial, Y=Coronal, X=Sagittal)
    views = [
        (cz, 0, "Axial (Z)", [cz, cy, cx]),     # Slice along Z (depth)
        (cy, 1, "Coronal (Y)", [cz, cy, cx]),   # Slice along Y (height)
        (cx, 2, "Sagittal (X)", [cz, cy, cx])  # Slice along X (width)
    ]
    
    # 2. Plotting Loop
    for i, (center_idx, row_index, title, _) in enumerate(views):
        
        # Get slice data using numpy.take
        input_slice = np.take(img_vol_np, center_idx, axis=i)
        prompt_slice = np.take(prompt_vol_np, center_idx, axis=i)
        output_slice = np.take(pred_vol_np, center_idx, axis=i)
        
        # Rotate all views to be aesthetically consistent (Axial usually needs no rot)
        if i != 0:
            input_slice = np.rot90(input_slice)
            prompt_slice = np.rot90(prompt_slice)
            output_slice = np.rot90(output_slice)

        # Plot 1: Input + Prompt (Column 1)
        axes[row_index, 0].imshow(input_slice, cmap="gray")
        # Green Prompt, Transparent
        axes[row_index, 0].imshow(prompt_slice, cmap="Greens", alpha=0.3) 
        axes[row_index, 0].set_title(f"Input: {title}", fontsize=12)
        axes[row_index, 0].axis("off")

        # Plot 2: Output Heatmap (Column 2)
        axes[row_index, 1].imshow(input_slice, cmap="gray", alpha=0.5)
        # Final Segmentation Output
        axes[row_index, 1].imshow(output_slice, cmap="jet", alpha=0.6)
        axes[row_index, 1].set_title(f"Prediction: {title}", fontsize=12)
        axes[row_index, 1].axis("off")
        
    # Final save
    fig.suptitle(f"EXPERIMENT {idx}: INTERACTIVE 3D RESPONSE", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"result_experiment_{idx}_3D.png"))
    plt.close()
    print(f"      Saved: result_experiment_{idx}_3D.png")


# --- MAIN EXECUTION LOGIC ---

def run_presentation_demo():
    print("\n--- FINAL 3D INTERACTIVE DEMO (ALL VIEWS) ---")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. Load Engine ONCE
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

    # 2. Load Data ONCE
    print("[2/3] Loading & Cropping CT Scan...")
    
    nifti_img = nib.load(INPUT_PATH)
    full_vol = nifti_img.get_fdata()
    
    # Center Crop & Normalize
    x_c, y_c, z_c = full_vol.shape[0]//2, full_vol.shape[1]//2, full_vol.shape[2]//2
    half = PATCH_SIZE // 2
    patch_np = full_vol[x_c-half:x_c+half, y_c-half:y_c+half, z_c-half:z_c+half]
    patch_np = (patch_np - np.mean(patch_np)) / (np.std(patch_np) + 1e-8)
    image_tensor = torch.from_numpy(patch_np).float().unsqueeze(0).unsqueeze(0)

    # 3. Run Multi-Click Loop
    print(f"[3/3] Testing {len(CLICKS_TO_TEST)} 3D prompts...")
    segmenter.network.eval()

    for i, click in enumerate(CLICKS_TO_TEST):
        
        # A. Create BALL Prompt
        prompt_data_np = create_ball_prompt(patch_np.shape, click, radius=4)
        prompt_tensor = torch.from_numpy(prompt_data_np).float().unsqueeze(0).unsqueeze(0)
        
        # B. Inference
        input_tensor = torch.cat([image_tensor, prompt_tensor], dim=1)
        with torch.no_grad():
            output = segmenter.network(input_tensor)
            if isinstance(output, (tuple, list)): output = output[0]

        # C. Process Result
        tumor_prob_np = torch.sigmoid(output[0, 1]).cpu().numpy()
        
        # D. Visualization & Save
        save_figure(patch_np, tumor_prob_np, prompt_data_np, click, i+1)

    print(f"\n✅ DONE. Check the 'plots' folder for {len(CLICKS_TO_TEST)} comparison images.")

if __name__ == "__main__":
    run_presentation_demo()