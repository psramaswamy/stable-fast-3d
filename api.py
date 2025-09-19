from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import io
import os
import uuid
from pathlib import Path
from datetime import datetime
import shutil
from typing import Optional
from contextlib import asynccontextmanager
import subprocess
import json

# Model global variable
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model
    print("Loading Stable Fast 3D model...")
    from sf3d.system import SF3D
    model = SF3D.from_pretrained(
        "stabilityai/stable-fast-3d",
        config_name="config.yaml",
        weight_name="model.safetensors"
    )
    model.eval()
    if torch.backends.mps.is_available():
        model = model.to("mps")
        print("Using Apple Silicon GPU")
    else:
        print("Using CPU")
    print("Model loaded successfully!")
    
    yield
    
    # Cleanup on shutdown
    print("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Stable Fast 3D API",
    description="Convert 2D images to 3D GLB models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
OUTPUT_DIR = Path("api_outputs")
UPLOAD_DIR = Path("api_uploads")
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# Store job status
jobs = {}

@app.get("/")
async def root():
    return {
        "message": "Stable Fast 3D API",
        "endpoints": {
            "POST /generate": "Generate 3D model from image",
            "GET /status/{job_id}": "Check job status",
            "GET /download/{job_id}": "Download generated model (GLB format)",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": "mps" if torch.backends.mps.is_available() else "cpu"
    }

@app.post("/generate")
async def generate_3d(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    texture_resolution: Optional[int] = 1024,
    remesh_option: Optional[str] = "none",
    target_vertex_count: Optional[int] = 10000
):
    # Validate file type
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        raise HTTPException(status_code=400, detail="Only JPG, PNG, and WebP images are supported")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    upload_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with open(upload_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Initialize job status
    jobs[job_id] = {
        "status": "processing",
        "created_at": datetime.now().isoformat(),
        "input_file": file.filename,
        "output_file": None,
        "error": None
    }
    
    # Process in background using subprocess (most reliable)
    background_tasks.add_task(
        process_image_subprocess, 
        job_id, 
        upload_path, 
        texture_resolution, 
        remesh_option,
        target_vertex_count
    )
    
    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Your 3D model is being generated. Check status with /status/{job_id}"
    }

def process_image_subprocess(job_id: str, image_path: Path, texture_resolution: int, remesh_option: str, target_vertex_count: int):
    """Process using the run.py script which we know works"""
    try:
        output_filename = f"{job_id}.glb"
        
        # Build command - run.py creates subdirectories with index numbers
        cmd = [
            "python", "run.py",
            str(image_path),
            "--output-dir", str(OUTPUT_DIR),
            "--texture-resolution", str(texture_resolution),
            "--remesh_option", remesh_option
        ]
        
        if remesh_option != "none":
            cmd.extend(["--target_vertex_count", str(target_vertex_count)])
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Count existing subdirectories to predict where output will be
        existing_dirs = [d for d in OUTPUT_DIR.iterdir() if d.is_dir() and d.name.isdigit()]
        next_index = len(existing_dirs)
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Process failed: {result.stderr}")
        
        print(f"Command output: {result.stdout}")
        
        # run.py saves files as {output_dir}/{index}/mesh.usdz (or mesh.glb as fallback)
        # where index starts from 0 and increments
        expected_output_dir = OUTPUT_DIR / str(next_index)
        expected_mesh_usdz = expected_output_dir / "mesh.usdz"
        expected_mesh_glb = expected_output_dir / "mesh.glb"
        final_output = OUTPUT_DIR / output_filename
        
        print(f"Looking for mesh files at: {expected_mesh_usdz} or {expected_mesh_glb}")
        
        # Check for GLB first, then USDZ
        if expected_mesh_glb.exists():
            expected_mesh_file = expected_mesh_glb
            print(f"Found GLB file: {expected_mesh_file}")
        elif expected_mesh_usdz.exists():
            expected_mesh_file = expected_mesh_usdz
            # Update output filename and path to match USDZ format
            output_filename = f"{job_id}.usdz"
            final_output = OUTPUT_DIR / output_filename
            print(f"Found USDZ file (GLB failed): {expected_mesh_file}")
        else:
            expected_mesh_file = None
        
        if expected_mesh_file and expected_mesh_file.exists():
            # Move the mesh file to the main directory with job_id name
            expected_mesh_file.rename(final_output)
            # Clean up the subdirectory if it's now empty (except input.png)
            remaining_files = list(expected_output_dir.glob("*"))
            if len(remaining_files) <= 1:  # Only input.png might remain
                shutil.rmtree(expected_output_dir)
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["output_file"] = output_filename
            print(f"Successfully moved mesh to: {final_output}")
        else:
            # Fallback: look for any newly created mesh files (GLB first, then USDZ)
            print("Expected files not found, searching for any mesh files...")
            mesh_files = list(OUTPUT_DIR.glob("*/mesh.glb")) + list(OUTPUT_DIR.glob("*/mesh.usdz"))
            if mesh_files:
                # Get the most recently created mesh file
                newest_mesh = max(mesh_files, key=lambda p: p.stat().st_mtime)
                print(f"Found mesh file: {newest_mesh}")
                
                # Update output filename and path based on the found file extension
                if newest_mesh.suffix == ".glb":
                    output_filename = f"{job_id}.glb"
                    final_output = OUTPUT_DIR / output_filename
                else:  # .usdz
                    output_filename = f"{job_id}.usdz"
                    final_output = OUTPUT_DIR / output_filename
                
                # Move it to the main directory
                newest_mesh.rename(final_output)
                # Clean up empty subdirectory if possible
                mesh_dir = newest_mesh.parent
                remaining_files = list(mesh_dir.glob("*"))
                if len(remaining_files) <= 1:  # Only input.png might remain
                    shutil.rmtree(mesh_dir)
                jobs[job_id]["status"] = "completed"
                jobs[job_id]["output_file"] = output_filename
                print(f"Successfully moved mesh to: {final_output}")
            else:
                # List all files in output directory for debugging
                all_files = list(OUTPUT_DIR.rglob("*"))
                print(f"All files in output directory: {all_files}")
                raise Exception("Output file not found after processing")
        
        # Clean up upload file
        os.remove(image_path)
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        print(f"Error processing job {job_id}: {e}")
        # Keep the uploaded file for debugging if needed
        if image_path.exists():
            print(f"Upload file preserved for debugging: {image_path}")

# Alternative: Direct model call (if you want to figure out the right parameters)
def process_image_direct(job_id: str, image_path: Path):
    """Direct model call - you'll need to figure out the exact parameters"""
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Generate 3D model - using just the image parameter
        with torch.no_grad():
            # The run_image method likely just takes the image
            mesh = model.run_image(image)
        
        # Save output
        output_filename = f"{job_id}.glb"
        output_path = OUTPUT_DIR / output_filename
        
        # Export as GLB directly
        mesh.export(str(output_path), file_type="glb", include_normals=True)
        
        # Update job status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["output_file"] = output_filename
        
        # Clean up upload
        os.remove(image_path)
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        print(f"Error processing job {job_id}: {e}")

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]

@app.get("/download/{job_id}")
async def download_model(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_status = jobs[job_id]["status"]
    if job_status != "completed":
        if job_status == "failed":
            error_msg = jobs[job_id].get("error", "Unknown error")
            raise HTTPException(status_code=400, detail=f"Job failed: {error_msg}")
        else:
            raise HTTPException(status_code=400, detail=f"Job status: {job_status}")
    
    output_file = jobs[job_id].get("output_file")
    if not output_file:
        raise HTTPException(status_code=500, detail="No output file recorded for completed job")
        
    file_path = OUTPUT_DIR / output_file
    if not file_path.exists():
        # List files in output directory for debugging
        existing_files = list(OUTPUT_DIR.glob("*"))
        print(f"Files in output directory: {existing_files}")
        raise HTTPException(status_code=404, detail=f"Output file not found: {output_file}")
    
    # Determine file extension from the output file
    file_extension = Path(output_file).suffix
    filename = f"model_{job_id}{file_extension}"
    
    return FileResponse(
        path=file_path,
        media_type="model/gltf-binary",
        filename=filename,
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Type": "model/gltf-binary"
        }
    )

@app.delete("/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete output file if exists
    if jobs[job_id]["output_file"]:
        file_path = OUTPUT_DIR / jobs[job_id]["output_file"]
        if file_path.exists():
            os.remove(file_path)
    
    # Remove from jobs
    del jobs[job_id]
    
    return {"message": "Job cleaned up successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
