import torch
import torch.nn as nn
import nibabel as nib
import numpy as np

# --- 1. The "Heavy" Standard Block (What LesionLocator/VISTA use) ---
class StandardBlock(nn.Module):
    def __init__(self, channels=768): # High channel count
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm3d(channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.norm(self.conv1(x)))

# --- 2. The "Optimized" EffiDec3D Block (Your Proposal) ---
class EffiDecBlock(nn.Module):
    def __init__(self, in_channels, reduced_channels=48): # Fixed small size (48)
        super().__init__()
        # Reduction layer: Squeeze 768 -> 48 immediately
        self.reduce = nn.Conv3d(in_channels, reduced_channels, kernel_size=1)
        # Efficient convolution on small channels
        self.conv = nn.Conv3d(reduced_channels, reduced_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm3d(reduced_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.reduce(x)
        return self.relu(self.norm(self.conv(x)))

def run_comparison():
    print("\n--- ARCHITECTURE COMPARISON FOR PRESENTATION ---")
    
    # 1. Load your dummy data
    img = nib.load("data/dummy_ct.nii.gz")
    # Convert to Tensor [Batch, Channel, D, H, W]
    input_tensor = torch.from_numpy(img.get_fdata()).float().unsqueeze(0).unsqueeze(0)
    
    # Simulate a high-level feature map (e.g., 768 channels, 16x16x16 resolution)
    # This simulates the bottleneck of a large model
    feature_map = torch.randn(1, 768, 16, 16, 16)
    print(f"Input Feature Map: {feature_map.shape} (Simulated Bottleneck)")

    # 2. Instantiate Models
    standard_model = StandardBlock(channels=768)
    optimized_model = EffiDecBlock(in_channels=768, reduced_channels=48)

    # 3. Calculate Parameters
    std_params = sum(p.numel() for p in standard_model.parameters())
    opt_params = sum(p.numel() for p in optimized_model.parameters())

    print(f"\n[Standard Block] Parameters: {std_params:,}")
    print(f"[EffiDec Block ] Parameters: {opt_params:,}")
    print(f"✅ IMPACT: {(1 - opt_params/std_params)*100:.1f}% reduction in parameters")

    # 4. Memory/Speed Test (Forward Pass)
    print("\nRunning Forward Pass...")
    
    # Standard
    out_std = standard_model(feature_map)
    print(f"Standard Output Shape: {out_std.shape}")
    
    # Optimized
    out_opt = optimized_model(feature_map)
    print(f"Optimized Output Shape: {out_opt.shape}")
    print("------------------------------------------------")
    print("Take a screenshot of this output for your slides!")

if __name__ == "__main__":
    run_comparison()