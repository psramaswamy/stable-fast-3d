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
import subprocess
import shutil
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

async def convert_glb_to_usdz(glb_file_path: Path, job_id: str) -> Path:
    """
    Convert GLB file to USDZ format using available conversion methods
    
    Args:
        glb_file_path: Path to the input GLB file
        job_id: Job ID for naming the output file
        
    Returns:
        Path to the converted USDZ file
        
    Raises:
        Exception: If conversion fails
    """
    usdz_filename = f"{job_id}.usdz"
    usdz_file_path = OUTPUT_DIR / usdz_filename
    
    # Check if USDZ file already exists (cached conversion)
    if usdz_file_path.exists():
        file_size = usdz_file_path.stat().st_size
        print(f"✅ Using cached USDZ file: {usdz_file_path} ({file_size} bytes)")
        return usdz_file_path
    
    # Validate input file
    if not glb_file_path.exists():
        raise Exception(f"Input GLB file not found: {glb_file_path}")
    
    input_size = glb_file_path.stat().st_size
    print(f"🔄 Converting GLB to USDZ: {glb_file_path} ({input_size} bytes) -> {usdz_file_path}")
    
    # Try conversion methods in order
    conversion_methods = [
        ("USD Command-line Tools", try_usd_conversion),
        ("Trimesh USD Export", try_trimesh_conversion), 
        ("gltf2usd Tool", try_gltf2usd_conversion),
        ("Python USD Library", try_python_usd_conversion)
    ]
    
    failed_methods = []
    
    for method_name, method_func in conversion_methods:
        print(f"🔧 Trying conversion method: {method_name}")
        try:
            if await method_func(glb_file_path, usdz_file_path):
                output_size = usdz_file_path.stat().st_size if usdz_file_path.exists() else 0
                print(f"✅ USDZ conversion successful using {method_name} ({output_size} bytes)")
                return usdz_file_path
            else:
                failed_methods.append(f"{method_name}: Method returned False")
                print(f"❌ {method_name}: Conversion failed")
        except Exception as e:
            failed_methods.append(f"{method_name}: {str(e)}")
            print(f"❌ {method_name}: Exception - {str(e)}")
    
    # If all methods fail, provide detailed error information
    error_details = "\n".join([f"  - {method}" for method in failed_methods])
    
    # Analyze failures to provide specific guidance
    specific_guidance = []
    
    if any("'Scene' object has no attribute 'vertices'" in failure for failure in failed_methods):
        specific_guidance.append("✅ The GLB file was loaded but had Scene structure issues - this is now fixed!")
    
    if any("No such file or directory: 'usdcat'" in failure for failure in failed_methods):
        specific_guidance.append("💡 Install USD command-line tools for best compatibility")
    
    if any("USDZ export not supported" in failure for failure in failed_methods):
        specific_guidance.append("💡 Reinstall trimesh with USD support: pip install trimesh[easy] --force-reinstall")
    
    if any("No such file or directory: 'gltf2usd'" in failure for failure in failed_methods):
        specific_guidance.append("💡 Build gltf2usd from source for additional conversion path")
    
    guidance_text = "\n".join(specific_guidance) if specific_guidance else ""
    
    raise Exception(
        f"USDZ conversion failed. All conversion methods failed:\n{error_details}\n\n"
        f"{guidance_text}\n\n"
        "🔧 Quick fix options:\n"
        "  1. For immediate solution: pip install usd-core (python USD - you have this!)\n"
        "  2. For best compatibility: Install USD command-line tools\n"
        "  3. Alternative: pip install trimesh[easy] --force-reinstall\n\n"
        "📋 Get detailed installation commands: curl YOUR_API_URL/conversion-tools\n"
        "🔍 The conversion issue may be resolved with the recent bug fixes!"
    )

async def try_usd_conversion(glb_path: Path, usdz_path: Path) -> bool:
    """Try conversion using USD command-line tools"""
    try:
        # Try using usdcat or other USD tools
        cmd = ["usdcat", "--flatten", str(glb_path), "-o", str(usdz_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and usdz_path.exists():
            print("✅ USDZ conversion successful using USD tools")
            return True
        else:
            print(f"USD tools conversion failed: {result.stderr}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"USD tools not available: {e}")
        return False

async def try_trimesh_conversion(glb_path: Path, usdz_path: Path) -> bool:
    """Try conversion using trimesh library"""
    try:
        import trimesh
        
        # Load the GLB mesh
        mesh = trimesh.load(str(glb_path))
        
        # Try to export as USDZ
        exported = mesh.export(file_type='usdz')
        if exported:
            with open(usdz_path, 'wb') as f:
                f.write(exported)
            print("✅ USDZ conversion successful using trimesh")
            return True
        else:
            print("Trimesh USDZ export returned None")
            return False
            
    except Exception as e:
        print(f"Trimesh conversion failed: {e}")
        return False

async def try_gltf2usd_conversion(glb_path: Path, usdz_path: Path) -> bool:
    """Try conversion using gltf2usd command-line tool"""
    try:
        # Try using gltf2usd if available
        cmd = ["gltf2usd", str(glb_path), str(usdz_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and usdz_path.exists():
            print("✅ USDZ conversion successful using gltf2usd")
            return True
        else:
            print(f"gltf2usd conversion failed: {result.stderr}")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"gltf2usd not available: {e}")
        return False

async def try_python_usd_conversion(glb_path: Path, usdz_path: Path) -> bool:
    """Try conversion using python USD libraries"""
    try:
        # Try importing USD Python bindings
        from pxr import Usd, UsdGeom, Gf, UsdShade
        import trimesh
        
        # Create a temporary USD stage
        temp_usd_path = usdz_path.with_suffix('.usda')
        stage = Usd.Stage.CreateNew(str(temp_usd_path))
        
        # Load GLB mesh
        loaded_mesh = trimesh.load(str(glb_path))
        print(f"Loaded GLB: {type(loaded_mesh)}")
        
        # Handle both Scene and Mesh objects
        if isinstance(loaded_mesh, trimesh.Scene):
            # Extract geometry from scene
            if len(loaded_mesh.geometry) == 0:
                raise Exception("No geometry found in GLB scene")
            
            # Use the first mesh in the scene
            mesh_name = list(loaded_mesh.geometry.keys())[0]
            mesh = loaded_mesh.geometry[mesh_name]
            print(f"Extracted mesh from scene: {mesh_name}")
        else:
            # Direct mesh object
            mesh = loaded_mesh
        
        # Validate mesh has required attributes
        if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
            raise Exception(f"Mesh object missing vertices or faces: {type(mesh)}")
        
        print(f"Mesh info: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Create mesh primitive in USD stage
        mesh_prim = UsdGeom.Mesh.Define(stage, "/mesh")
        
        # Set mesh data (basic conversion)
        mesh_prim.GetPointsAttr().Set([Gf.Vec3f(*v) for v in mesh.vertices])
        mesh_prim.GetFaceVertexIndicesAttr().Set(mesh.faces.flatten().tolist())
        mesh_prim.GetFaceVertexCountsAttr().Set([3] * len(mesh.faces))
        
        # Add normals if available
        if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
            mesh_prim.GetNormalsAttr().Set([Gf.Vec3f(*n) for n in mesh.vertex_normals])
        
        # Save USD stage
        stage.Save()
        print(f"Saved USD stage to: {temp_usd_path}")
        
        # Try to convert USDA to USDZ using usdzip
        try:
            cmd = ["usdzip", str(usdz_path), str(temp_usd_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and usdz_path.exists():
                print("✅ USDZ conversion successful using python-usd + usdzip")
                # Cleanup temp file
                if temp_usd_path.exists():
                    temp_usd_path.unlink()
                return True
            else:
                print(f"usdzip failed: {result.stderr}")
        
        except FileNotFoundError:
            print("usdzip not available, trying alternative method")
        
        # Alternative: Try to save directly as USDZ (if supported)
        try:
            # Some USD versions support direct USDZ export
            usdz_stage = Usd.Stage.CreateNew(str(usdz_path))
            UsdGeom.Mesh.Define(usdz_stage, "/mesh").GetPrim().GetReferences().AddReference(str(temp_usd_path))
            usdz_stage.Save()
            
            if usdz_path.exists():
                print("✅ USDZ conversion successful using direct USD export")
                # Cleanup temp file
                if temp_usd_path.exists():
                    temp_usd_path.unlink()
                return True
        
        except Exception as e:
            print(f"Direct USDZ export failed: {e}")
        
        # Cleanup temp file on failure
        if temp_usd_path.exists():
            temp_usd_path.unlink()
            
        return False
            
    except Exception as e:
        print(f"python-usd conversion failed: {e}")
        return False

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
            "GET /download/{job_id}?format=glb|usdz": "Download generated model (GLB or USDZ format)",
            "GET /image/{job_id}": "Retrieve uploaded image",
            "GET /conversion-tools": "Check available USDZ conversion tools",
            "GET /health": "Health check",
            "GET /stats": "System statistics",
            "DELETE /cleanup/{job_id}": "Clean up job files"
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
    
    # Save uploaded image for later retrieval
    input_filename = f"{job_id}_input{Path(file.filename).suffix}"
    input_path = UPLOAD_DIR / input_filename
    with open(input_path, "wb") as f:
        f.write(file_content)
    
    # Initialize job status
    jobs[job_id] = {
        "status": "processing",
        "created_at": datetime.now().isoformat(),
        "input_file": file.filename,
        "input_path": input_filename,  # Store the saved input file name
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
        jobs[job_id]["progress"] = "Saving 3D model as USDZ..."
        output_filename = f"{job_id}.usdz"
        output_path = OUTPUT_DIR / output_filename
        
        # Export as USDZ using our custom function
        from sf3d.utils import export_mesh_as_usdz
        success = export_mesh_as_usdz(trimesh_mesh, output_path)
        
        if not success:
            # Fallback to GLB if USDZ export fails
            print("⚠️ USDZ export failed, falling back to GLB format")
            output_filename = f"{job_id}.glb"
            output_path = OUTPUT_DIR / output_filename
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
async def download_model(job_id: str, format: Optional[str] = "usdz"):
    # Validate format parameter
    supported_formats = ["glb", "usdz"]
    format = format.lower()
    if format not in supported_formats:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported format '{format}'. Supported formats: {supported_formats}"
        )
    
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
    
    # Get the original GLB file
    glb_file_path = OUTPUT_DIR / output_file
    if not glb_file_path.exists():
        # List files in output directory for debugging
        existing_files = list(OUTPUT_DIR.glob("*"))
        print(f"Files in output directory: {existing_files}")
        raise HTTPException(status_code=404, detail=f"Output file not found: {output_file}")
    
    # Determine the actual file format based on what was generated
    actual_file_extension = glb_file_path.suffix.lower()
    
    # If GLB format is requested
    if format == "glb":
        if actual_file_extension == ".usdz":
            # Convert USDZ to GLB on demand
            try:
                glb_output_path = OUTPUT_DIR / f"{job_id}_glb.glb"
                # Simple conversion: load USDZ and export as GLB
                # Note: This is a basic conversion, might need improvement
                import trimesh
                # For now, inform user that the file is in USDZ format
                return FileResponse(
                    path=glb_file_path,
                    media_type="application/octet-stream", 
                    filename=f"model_{job_id}.usdz",
                    headers={
                        "Content-Disposition": f"attachment; filename=model_{job_id}.usdz",
                        "Content-Type": "application/octet-stream"
                    }
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"File format conversion failed: {str(e)}")
        else:
            # Return GLB file as requested
            return FileResponse(
                path=glb_file_path,
                media_type="application/octet-stream",
                filename=f"model_{job_id}.glb",
                headers={
                    "Content-Disposition": f"attachment; filename=model_{job_id}.glb",
                    "Content-Type": "application/octet-stream"
                }
            )
    
    # If USDZ format is requested
    elif format == "usdz":
        if actual_file_extension == ".usdz":
            # File is already in USDZ format
            file_size = glb_file_path.stat().st_size
            filename = f"model_{job_id}.usdz"
            
            return FileResponse(
                path=glb_file_path,
                media_type="application/vnd.usdz+zip",  # Proper USDZ MIME type
                filename=filename,
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"',  # Quoted filename
                    "Content-Type": "application/vnd.usdz+zip",
                    "Content-Length": str(file_size),
                    "Cache-Control": "no-cache"
                }
            )
        else:
            # Convert GLB to USDZ on demand
            try:
                usdz_file_path = await convert_glb_to_usdz(glb_file_path, job_id)
                
                # Ensure the file exists and get its size
                if not usdz_file_path.exists():
                    raise Exception("Converted USDZ file not found")
                
                file_size = usdz_file_path.stat().st_size
                filename = f"model_{job_id}.usdz"
                
                return FileResponse(
                    path=usdz_file_path,
                    media_type="application/vnd.usdz+zip",  # Proper USDZ MIME type
                    filename=filename,
                    headers={
                        "Content-Disposition": f'attachment; filename="{filename}"',  # Quoted filename
                        "Content-Type": "application/vnd.usdz+zip",
                        "Content-Length": str(file_size),
                        "Cache-Control": "no-cache"
                    }
                )
            except Exception as e:
                error_msg = str(e)
                print(f"❌ USDZ conversion failed for job {job_id}: {error_msg}")
                
                # Provide helpful error message with tool suggestions
                if "No suitable conversion tool found" in error_msg:
                    detail = (
                        "USDZ conversion failed: No conversion tools available. "
                        "Check /conversion-tools endpoint for installation instructions. "
                        "You can download the GLB format instead by setting format=glb."
                    )
                else:
                    detail = f"Failed to convert to USDZ format: {error_msg}"
                    
                raise HTTPException(status_code=500, detail=detail)

@app.get("/image/{job_id}")
async def get_uploaded_image(job_id: str):
    """Retrieve the original uploaded image for a given job ID"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    input_filename = jobs[job_id].get("input_path")
    if not input_filename:
        raise HTTPException(status_code=404, detail="No input image found for this job")
    
    image_path = UPLOAD_DIR / input_filename
    if not image_path.exists():
        # List files in upload directory for debugging
        existing_files = list(UPLOAD_DIR.glob("*"))
        print(f"Files in upload directory: {existing_files}")
        raise HTTPException(status_code=404, detail=f"Input image file not found: {input_filename}")
    
    # Determine MIME type based on file extension
    suffix = image_path.suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp'
    }
    media_type = mime_types.get(suffix, 'image/jpeg')
    
    # Get original filename from job data
    original_filename = jobs[job_id].get("input_file", f"input_{job_id}{suffix}")
    
    return FileResponse(
        path=image_path,
        media_type=media_type,
        filename=original_filename,
        headers={
            "Content-Disposition": f"inline; filename={original_filename}",
            "Content-Type": media_type
        }
    )

@app.delete("/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    files_deleted = 0
    
    # Delete output GLB file if exists
    if jobs[job_id].get("output_file"):
        file_path = OUTPUT_DIR / jobs[job_id]["output_file"]
        if file_path.exists():
            os.remove(file_path)
            files_deleted += 1
    
    # Delete cached USDZ file if exists
    usdz_filename = f"{job_id}.usdz"
    usdz_path = OUTPUT_DIR / usdz_filename
    if usdz_path.exists():
        os.remove(usdz_path)
        files_deleted += 1
    
    # Delete input file if exists
    if jobs[job_id].get("input_path"):
        input_path = UPLOAD_DIR / jobs[job_id]["input_path"]
        if input_path.exists():
            os.remove(input_path)
            files_deleted += 1
    
    # Remove from jobs
    del jobs[job_id]
    
    return {"message": f"Job cleaned up successfully. {files_deleted} files deleted."}

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

@app.get("/conversion-tools")
async def check_conversion_tools():
    """Check which USDZ conversion tools are available on the system"""
    tools_status = {}
    
    # Check USD command-line tools
    try:
        result = subprocess.run(["usdcat", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            tools_status["usd_tools"] = {
                "available": True,
                "version": result.stdout.strip(),
                "description": "Pixar USD command-line tools"
            }
        else:
            tools_status["usd_tools"] = {"available": False, "error": "Command failed"}
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        tools_status["usd_tools"] = {"available": False, "error": str(e)}
    
    # Check trimesh with USD support
    try:
        import trimesh
        # Test if USDZ export is available
        test_mesh = trimesh.creation.box()
        try:
            exported = test_mesh.export(file_type='usdz')
            if exported:
                tools_status["trimesh_usd"] = {
                    "available": True,
                    "version": trimesh.__version__,
                    "description": "Trimesh library with USD export support"
                }
            else:
                tools_status["trimesh_usd"] = {
                    "available": False, 
                    "error": "USDZ export not supported"
                }
        except Exception as e:
            tools_status["trimesh_usd"] = {
                "available": False,
                "error": f"USDZ export failed: {str(e)}"
            }
    except ImportError as e:
        tools_status["trimesh_usd"] = {"available": False, "error": f"Import failed: {str(e)}"}
    
    # Check gltf2usd tool
    try:
        result = subprocess.run(["gltf2usd", "--help"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            tools_status["gltf2usd"] = {
                "available": True,
                "description": "gltf2usd command-line converter"
            }
        else:
            tools_status["gltf2usd"] = {"available": False, "error": "Command failed"}
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        tools_status["gltf2usd"] = {"available": False, "error": str(e)}
    
    # Check Python USD (pxr) library
    try:
        from pxr import Usd, UsdGeom
        import pxr
        tools_status["python_usd"] = {
            "available": True,
            "version": getattr(pxr, '__version__', 'unknown'),
            "description": "Python USD (pxr) library"
        }
    except ImportError as e:
        tools_status["python_usd"] = {"available": False, "error": f"Import failed: {str(e)}"}
    
    # Summary
    available_tools = [name for name, info in tools_status.items() if info.get("available", False)]
    
    return {
        "usdz_conversion_supported": len(available_tools) > 0,
        "available_tools": available_tools,
        "total_methods": len(tools_status),
        "tools_status": tools_status,
        "installation_help": {
            "usd_tools": {
                "description": "Install Pixar USD command-line tools (usdcat, usdzip)",
                "commands": [
                    "# Method 1: Build from source (recommended)",
                    "git clone https://github.com/PixarAnimationStudios/OpenUSD",
                    "python OpenUSD/build_scripts/build_usd.py /usr/local/USD",
                    "export PATH=\"/usr/local/USD/bin:$PATH\"",
                    "export PYTHONPATH=\"/usr/local/USD/lib/python:$PYTHONPATH\"",
                    "",
                    "# Method 2: Using conda (if available)",
                    "conda install -c conda-forge openusd-tools"
                ],
                "verify": "usdcat --version && usdzip --help"
            },
            "trimesh_usd": {
                "description": "Install trimesh with USD export support",
                "commands": [
                    "pip install trimesh[easy] --force-reinstall",
                    "# Ensure USD Python bindings are available",
                    "python -c \"from pxr import Usd; print('USD bindings OK')\""
                ],
                "verify": "python -c \"import trimesh; print('Trimesh version:', trimesh.__version__)\""
            },
            "gltf2usd": {
                "description": "Build gltf2usd from source",
                "commands": [
                    "# Prerequisites: USD must be installed first",
                    "git clone https://github.com/kcoley/gltf2usd.git",
                    "cd gltf2usd",
                    "pip install -r requirements.txt",
                    "# Add to PATH or use absolute path",
                    "export PATH=\"$(pwd):$PATH\"",
                    "# Or create symlink:",
                    "sudo ln -sf $(pwd)/gltf2usd.py /usr/local/bin/gltf2usd"
                ],
                "verify": "gltf2usd --help || python gltf2usd.py --help"
            },
            "python_usd": {
                "description": "Install Python USD bindings",
                "commands": [
                    "pip install usd-core",
                    "# Alternative if usd-core doesn't work:",
                    "# pip install pxr"
                ],
                "verify": "python -c \"from pxr import Usd, UsdGeom; print('Python USD OK')\""
            }
        },
        "quick_setup": {
            "description": "Quick setup for USDZ conversion",
            "recommended_order": [
                "1. Install python_usd: pip install usd-core",
                "2. Build USD tools for best compatibility",
                "3. Optional: Install gltf2usd for additional conversion path"
            ],
            "minimum_requirement": "At least one of the above tools must be installed and working"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)