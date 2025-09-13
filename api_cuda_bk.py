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
import tempfile
from typing import Optional
from contextlib import asynccontextmanager, nullcontext
import numpy as np
import rembg

# Import SF3D system
from sf3d.system import SF3D
import sf3d.utils as sf3d_utils

# Model and session global variables
model = None
rembg_session = None

# Configuration constants
COND_WIDTH = 512
COND_HEIGHT = 512
COND_DISTANCE = 1.6
COND_FOVY_DEG = 40
BACKGROUND_COLOR = [0.5, 0.5, 0.5]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and rembg session on startup
    global model, rembg_session
    print("Loading Stable Fast 3D model...")
    
    # Load the main model
    model = SF3D.from_pretrained(
        "stabilityai/stable-fast-3d",
        config_name="config.yaml",
        weight_name="model.safetensors"
    )
    model.eval()
    
    # Use CUDA if available
    device = sf3d_utils.get_device()
    model = model.to(device)
    print(f"Model loaded on device: {device}")
    
    # Load background removal session
    print("Loading background removal model...")
    rembg_session = rembg.new_session()
    print("Background removal model loaded!")
    
    # Pre-compute cached values (from gradio_app.py)
    global c2w_cond, intrinsic, intrinsic_normed_cond
    c2w_cond = sf3d_utils.default_cond_c2w(COND_DISTANCE)
    intrinsic, intrinsic_normed_cond = sf3d_utils.create_intrinsic_from_fov_deg(
        COND_FOVY_DEG, COND_HEIGHT, COND_WIDTH
    )
    
    print("✅ All models loaded successfully!")
    
    yield
    
    # Cleanup on shutdown
    print("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Stable Fast 3D CUDA API",
    description="High-performance CUDA-optimized API to convert 2D images to 3D GLB models",
    version="2.0.0",
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
        "message": "Stable Fast 3D CUDA API",
        "version": "2.0.0",
        "features": ["Direct model integration", "CUDA acceleration", "Background removal"],
        "endpoints": {
            "POST /generate": "Generate 3D model from image",
            "GET /status/{job_id}": "Check job status",
            "GET /download/{job_id}": "Download generated model",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    device = sf3d_utils.get_device()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "memory_allocated": torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    }

@app.post("/generate")
async def generate_3d(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    texture_resolution: Optional[int] = 1024,
    remesh_option: Optional[str] = "none",
    target_vertex_count: Optional[int] = 10000,
    foreground_ratio: Optional[float] = 0.85,
    remove_background: Optional[bool] = True
):
    # Validate file type
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        raise HTTPException(status_code=400, detail="Only JPG, PNG, and WebP images are supported")
    
    # Read file content before starting background task (to avoid closed file error)
    file_content = await file.read()
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    jobs[job_id] = {
        "status": "processing",
        "created_at": datetime.now().isoformat(),
        "input_file": file.filename,
        "output_file": None,
        "error": None,
        "progress": "Starting processing..."
    }
    
    # Process in background using direct model calls
    background_tasks.add_task(
        process_image_direct, 
        job_id, 
        file_content,  # Pass bytes instead of UploadFile
        texture_resolution, 
        remesh_option,
        target_vertex_count,
        foreground_ratio,
        remove_background
    )
    
    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Your 3D model is being generated using direct CUDA acceleration. Check status with /status/{job_id}"
    }

async def process_image_direct(
    job_id: str, 
    file_content: bytes, 
    texture_resolution: int, 
    remesh_option: str, 
    target_vertex_count: int,
    foreground_ratio: float,
    remove_background: bool
):
    """Process image using direct model calls - much faster than subprocess"""
    try:
        # Process uploaded file content
        jobs[job_id]["progress"] = "Processing uploaded image..."
        input_image = Image.open(io.BytesIO(file_content)).convert("RGBA")
        
        # Remove background if requested
        if remove_background:
            jobs[job_id]["progress"] = "Removing background..."
            # Check if image already has transparency
            alpha_channel = np.array(input_image.getchannel("A"))
            if alpha_channel.min() == 255:  # No transparency, need to remove background
                input_image = rembg.remove(input_image, session=rembg_session)
        
        # Resize foreground
        jobs[job_id]["progress"] = "Resizing image..."
        processed_image = sf3d_utils.resize_foreground(
            input_image, foreground_ratio, out_size=(COND_WIDTH, COND_HEIGHT)
        )
        
        # Create batch for model inference
        jobs[job_id]["progress"] = "Preparing model input..."
        model_batch = create_batch(processed_image)
        
        # Get device
        device = sf3d_utils.get_device()
        
        # Move batch to device
        model_batch = {k: v.to(device) for k, v in model_batch.items()}
        
        # Generate 3D model
        jobs[job_id]["progress"] = "Generating 3D model..."
        
        # Reset CUDA memory stats for monitoring
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            with torch.autocast(
                device_type=device, dtype=torch.bfloat16
            ) if "cuda" in device else nullcontext():
                # Use generate_mesh method directly
                trimesh_meshes, global_dict = model.generate_mesh(
                    model_batch, 
                    texture_resolution, 
                    remesh_option, 
                    target_vertex_count
                )
                trimesh_mesh = trimesh_meshes[0]  # Get first mesh
        
        # Log memory usage
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"Peak CUDA Memory for job {job_id}: {peak_memory:.2f} MB")
        
        # Save the generated mesh
        jobs[job_id]["progress"] = "Saving 3D model..."
        output_filename = f"{job_id}.glb"
        output_path = OUTPUT_DIR / output_filename
        
        # Export as GLB with normals
        trimesh_mesh.export(str(output_path), file_type="glb", include_normals=True)
        
        # Update job status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["output_file"] = output_filename
        jobs[job_id]["progress"] = "Complete!"
        
        print(f"✅ Job {job_id} completed successfully")
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["progress"] = f"Failed: {str(e)}"
        print(f"❌ Error processing job {job_id}: {e}")
        import traceback
        traceback.print_exc()

def create_batch(input_image: Image) -> dict[str, torch.Tensor]:
    """Create batch for model inference (adapted from gradio_app.py)"""
    img_cond = (
        torch.from_numpy(
            np.asarray(input_image.resize((COND_WIDTH, COND_HEIGHT))).astype(np.float32)
            / 255.0
        )
        .float()
        .clip(0, 1)
    )
    mask_cond = img_cond[:, :, -1:]
    rgb_cond = torch.lerp(
        torch.tensor(BACKGROUND_COLOR)[None, None, :], img_cond[:, :, :3], mask_cond
    )

    batch_elem = {
        "rgb_cond": rgb_cond,
        "mask_cond": mask_cond,
        "c2w_cond": c2w_cond.unsqueeze(0),
        "intrinsic_cond": intrinsic.unsqueeze(0),
        "intrinsic_normed_cond": intrinsic_normed_cond.unsqueeze(0),
    }
    # Add batch dim
    batched = {k: v.unsqueeze(0) for k, v in batch_elem.items()}
    return batched

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Add memory info if CUDA is available
    status = jobs[job_id].copy()
    if torch.cuda.is_available():
        status["cuda_memory_allocated"] = torch.cuda.memory_allocated() / 1024 / 1024
        status["cuda_memory_cached"] = torch.cuda.memory_reserved() / 1024 / 1024
    
    return status

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
            progress = jobs[job_id].get("progress", "Processing...")
            raise HTTPException(status_code=400, detail=f"Job status: {job_status}. Progress: {progress}")
    
    output_file = jobs[job_id].get("output_file")
    if not output_file:
        raise HTTPException(status_code=500, detail="No output file recorded for completed job")
        
    file_path = OUTPUT_DIR / output_file
    if not file_path.exists():
        # List files in output directory for debugging
        existing_files = list(OUTPUT_DIR.glob("*"))
        print(f"Files in output directory: {existing_files}")
        raise HTTPException(status_code=404, detail=f"Output file not found: {output_file}")
    
    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=f"model_{job_id}.glb",
        headers={
            "Content-Disposition": f"attachment; filename=model_{job_id}.glb",
            "Content-Type": "application/octet-stream"
        }
    )

@app.delete("/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete output file if exists
    if jobs[job_id].get("output_file"):
        file_path = OUTPUT_DIR / jobs[job_id]["output_file"]
        if file_path.exists():
            os.remove(file_path)
    
    # Remove from jobs
    del jobs[job_id]
    
    return {"message": "Job cleaned up successfully"}

@app.get("/stats")
async def get_stats():
    """Get system and processing statistics"""
    stats = {
        "total_jobs": len(jobs),
        "active_jobs": len([j for j in jobs.values() if j["status"] == "processing"]),
        "completed_jobs": len([j for j in jobs.values() if j["status"] == "completed"]),
        "failed_jobs": len([j for j in jobs.values() if j["status"] == "failed"]),
    }
    
    if torch.cuda.is_available():
        stats.update({
            "cuda_device": torch.cuda.get_device_name(),
            "cuda_memory_allocated": torch.cuda.memory_allocated() / 1024 / 1024,
            "cuda_memory_cached": torch.cuda.memory_reserved() / 1024 / 1024,
            "cuda_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        })
    
    return stats

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)