import os
import tempfile
import io
from contextlib import nullcontext
from typing import Optional

import rembg
import torch
from PIL import Image
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

from sf3d.system import SF3D
from sf3d.utils import get_device, remove_background, resize_foreground

# Initialize Flask application
app = Flask(__name__)

# Configuration settings for the API
class Config:
    def __init__(self):
        self.device = get_device()
        self.pretrained_model = "stabilityai/stable-fast-3d"
        self.foreground_ratio = 0.85
        self.texture_resolution = 1024
        self.remesh_option = "none"
        self.target_vertex_count = -1
        self.allowed_extensions = {'png', 'jpg', 'jpeg'}

# Global configuration instance
config = Config()

# Global model and session variables (initialized once)
model = None
rembg_session = None

def initialize_model():
    """Initialize the SF3D model and rembg session"""
    global model, rembg_session

    # Determine the appropriate device to use
    device = config.device
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        device = "cpu"

    print(f"Initializing model on device: {device}")

    # Load the pretrained SF3D model
    model = SF3D.from_pretrained(
        config.pretrained_model,
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    model.to(device)
    model.eval()

    # Initialize background removal session
    rembg_session = rembg.new_session()
    print("Model initialization complete")

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.allowed_extensions

def process_image_to_mesh(image_file) -> str:
    """
    Process an uploaded image file and generate a 3D mesh

    Args:
        image_file: Uploaded image file object

    Returns:
        str: Path to the generated GLB file
    """
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded image to temporary location
        input_image_path = os.path.join(temp_dir, "input_image.png")
        image_file.save(input_image_path)

        # Load and preprocess the image
        image = Image.open(input_image_path).convert("RGBA")

        # Remove background and resize foreground
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, config.foreground_ratio)

        # Generate 3D mesh using the SF3D model
        device = config.device
        if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
            device = "cpu"

        with torch.no_grad():
            # Use autocast for better performance on CUDA devices
            with torch.autocast(
                device_type=device, dtype=torch.bfloat16
            ) if "cuda" in device else nullcontext():
                mesh, glob_dict = model.run_image(
                    [image],  # Pass image as list for batch processing
                    bake_resolution=config.texture_resolution,
                    remesh=config.remesh_option,
                    vertex_count=config.target_vertex_count,
                )

        # Export mesh to GLB format
        output_mesh_path = os.path.join(temp_dir, "output_mesh.glb")
        mesh.export(output_mesh_path, include_normals=True)

        # Read the GLB file and return as bytes
        with open(output_mesh_path, 'rb') as f:
            glb_data = f.read()

        # Create a temporary file to store the GLB data for sending
        temp_glb = tempfile.NamedTemporaryFile(delete=False, suffix='.glb')
        temp_glb.write(glb_data)
        temp_glb.close()

        return temp_glb.name

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API is running"""
    return jsonify({"status": "healthy", "message": "SF3D API is running"})

@app.route('/generate', methods=['POST'])
def generate_3d_model():
    """
    Main endpoint to generate 3D model from uploaded image

    Expects:
        - POST request with image file in 'image' field

    Returns:
        - GLB file as download or error message
    """
    try:
        # Check if image file is present in request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']

        # Check if a file was actually selected
        if image_file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Validate file type
        if not allowed_file(image_file.filename):
            return jsonify({
                "error": f"Invalid file type. Allowed types: {', '.join(config.allowed_extensions)}"
            }), 400

        # Process the image and generate 3D mesh
        try:
            glb_file_path = process_image_to_mesh(image_file)

            # Return the GLB file as a download
            return send_file(
                glb_file_path,
                as_attachment=True,
                download_name=f"{secure_filename(image_file.filename)}_3d_model.glb",
                mimetype='model/gltf-binary'
            )

        except Exception as processing_error:
            print(f"Error during 3D generation: {str(processing_error)}")
            return jsonify({
                "error": "Failed to generate 3D model",
                "details": str(processing_error)
            }), 500

    except Exception as e:
        print(f"Unexpected error in generate_3d_model: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

@app.route('/config', methods=['GET'])
def get_config():
    """Get current API configuration"""
    return jsonify({
        "device": config.device,
        "texture_resolution": config.texture_resolution,
        "foreground_ratio": config.foreground_ratio,
        "remesh_option": config.remesh_option,
        "target_vertex_count": config.target_vertex_count,
        "allowed_extensions": list(config.allowed_extensions)
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle not found error"""
    return jsonify({"error": "Endpoint not found"}), 404

if __name__ == '__main__':
    print("Starting SF3D API Server...")

    # Initialize the model and dependencies
    initialize_model()

    # Start the Flask development server
    print("Server starting on http://0.0.0.0:8002")
    app.run(host='0.0.0.0', port=8002, debug=False)