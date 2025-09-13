#!/usr/bin/env python3
import os
import sys
import torch
from PIL import Image
from pathlib import Path

# Add HEIC support if needed
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass  # HEIC support not installed

# Add the stable-fast-3d directory to path
sys.path.insert(0, '.')

from sf3d.system import SF3D

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_cli.py <image_path>")
        print("Example: python run_cli.py test.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found")
        sys.exit(1)
    
    output_name = Path(image_path).stem + ".usdz"
    
    print("🚀 Loading Stable Fast 3D model...")
    model = SF3D.from_pretrained(
        "stabilityai/stable-fast-3d",
        config_name="config.yaml",
        weight_name="model.safetensors"
    )
    model.eval()
    
    # Use MPS if available (Apple Silicon)
    if torch.backends.mps.is_available():
        model = model.to("mps")
        print("✅ Using Apple Silicon GPU")
    else:
        print("✅ Using CPU")
    
    print(f"📷 Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    
    print("⚙️ Generating 3D model (this takes ~10-30 seconds)...")
    with torch.no_grad():
        # Call run_image with just the image - check what parameters it actually accepts
        mesh = model.run_image(image)
    
    # Export the mesh
    mesh.export(output_name)
    print(f"✅ Success! Saved 3D model to: {output_name}")
    print(f"📦 You can view this file in Blender, Unity, or any 3D viewer")

if __name__ == "__main__":
    main()
