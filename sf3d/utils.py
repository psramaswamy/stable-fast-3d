import os
from typing import Any, Union
from pathlib import Path

import numpy as np
import rembg
import torch
import torchvision.transforms.functional as torchvision_F
from PIL import Image
import trimesh

import sf3d.models.utils as sf3d_utils


def get_device():
    if os.environ.get("SF3D_USE_CPU", "0") == "1":
        return "cpu"

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device


def create_intrinsic_from_fov_deg(fov_deg: float, cond_height: int, cond_width: int):
    intrinsic = sf3d_utils.get_intrinsic_from_fov(
        np.deg2rad(fov_deg),
        H=cond_height,
        W=cond_width,
    )
    intrinsic_normed_cond = intrinsic.clone()
    intrinsic_normed_cond[..., 0, 2] /= cond_width
    intrinsic_normed_cond[..., 1, 2] /= cond_height
    intrinsic_normed_cond[..., 0, 0] /= cond_width
    intrinsic_normed_cond[..., 1, 1] /= cond_height

    return intrinsic, intrinsic_normed_cond


def default_cond_c2w(distance: float):
    c2w_cond = torch.as_tensor(
        [
            [0, 0, 1, distance],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]
    ).float()
    return c2w_cond


def remove_background(
    image: Image,
    rembg_session: Any = None,
    force: bool = False,
    **rembg_kwargs,
) -> Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image


def get_1d_bounds(arr):
    nz = np.flatnonzero(arr)
    return nz[0], nz[-1]


def get_bbox_from_mask(mask, thr=0.5):
    masks_for_box = (mask > thr).astype(np.float32)
    assert masks_for_box.sum() > 0, "Empty mask!"
    x0, x1 = get_1d_bounds(masks_for_box.sum(axis=-2))
    y0, y1 = get_1d_bounds(masks_for_box.sum(axis=-1))
    return x0, y0, x1, y1


def resize_foreground(
    image: Union[Image.Image, np.ndarray],
    ratio: float,
    out_size=None,
) -> Image:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image, mode="RGBA")
    assert image.mode == "RGBA"
    # Get bounding box
    mask_np = np.array(image)[:, :, -1]
    x1, y1, x2, y2 = get_bbox_from_mask(mask_np, thr=0.5)
    h, w = y2 - y1, x2 - x1
    yc, xc = (y1 + y2) / 2, (x1 + x2) / 2
    scale = max(h, w) / ratio

    new_image = torchvision_F.crop(
        image,
        top=int(yc - scale / 2),
        left=int(xc - scale / 2),
        height=int(scale),
        width=int(scale),
    )
    if out_size is not None:
        new_image = new_image.resize(out_size)

    return new_image


def export_mesh_as_usdz(mesh: trimesh.Trimesh, output_path: Union[str, Path]) -> bool:
    """
    Export a trimesh Mesh object as USDZ format
    
    Args:
        mesh: trimesh.Trimesh object to export
        output_path: Path where to save the USDZ file
        
    Returns:
        bool: True if export succeeded, False otherwise
    """
    try:
        from pxr import Usd, UsdGeom, Gf, UsdShade, Sdf
        import tempfile
        import subprocess
        
        # Debug: Check mesh type and attributes
        print(f"🔍 Debug: mesh type = {type(mesh)}")
        print(f"🔍 Debug: mesh attributes = {dir(mesh)}")
        
        # Handle Scene objects (which contain meshes)
        if hasattr(mesh, 'geometry') and hasattr(mesh, 'dump'):
            # This is likely a trimesh Scene object
            print("🔍 Debug: Detected Scene object, extracting geometry...")
            geometries = list(mesh.geometry.values())
            if not geometries:
                print("❌ No geometry found in Scene object")
                return False
            mesh = geometries[0]  # Use the first geometry
            print(f"🔍 Debug: Extracted mesh type = {type(mesh)}")
        
        # Verify it's a proper trimesh object
        if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
            print(f"❌ Invalid mesh object: missing vertices or faces. Type: {type(mesh)}")
            return False
        
        output_path = Path(output_path)
        
        # Create a temporary USD file first
        with tempfile.NamedTemporaryFile(suffix='.usda', delete=False) as temp_file:
            temp_usd_path = Path(temp_file.name)
        
        try:
            # Create USD stage
            stage = Usd.Stage.CreateNew(str(temp_usd_path))
            
            # Create root transform
            root_prim = UsdGeom.Xform.Define(stage, "/Root")
            stage.SetDefaultPrim(root_prim.GetPrim())
            
            # Create mesh geometry
            mesh_prim = UsdGeom.Mesh.Define(stage, "/Root/Mesh")
            
            # Set mesh data
            if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                # Convert vertices to USD format
                points = [Gf.Vec3f(*v) for v in mesh.vertices]
                mesh_prim.GetPointsAttr().Set(points)
                
                # Convert faces to USD format
                face_vertex_indices = mesh.faces.flatten().tolist()
                mesh_prim.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
                mesh_prim.GetFaceVertexCountsAttr().Set([3] * len(mesh.faces))
                
                # Add normals if available
                if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                    normals = [Gf.Vec3f(*n) for n in mesh.vertex_normals]
                    mesh_prim.GetNormalsAttr().Set(normals)
                
                # Handle UV coordinates and materials if available
                if hasattr(mesh, 'visual') and mesh.visual is not None:
                    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                        # Convert UV coordinates
                        uv_coords = mesh.visual.uv
                        # USD expects flipped V coordinate
                        uv_coords_flipped = uv_coords.copy()
                        uv_coords_flipped[:, 1] = 1.0 - uv_coords_flipped[:, 1]
                        
                        # Create primvar for UV coordinates using proper USD API
                        try:
                            print(f"🔍 Debug: mesh_prim type = {type(mesh_prim)}")
                            from pxr import UsdGeom
                            # Use PrimvarsAPI to create primvars
                            primvars_api = UsdGeom.PrimvarsAPI(mesh_prim)
                            uv_primvar = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray)
                            uv_primvar.Set([Gf.Vec2f(*uv) for uv in uv_coords_flipped])
                            uv_primvar.SetInterpolation("vertex")
                            print("✅ UV coordinates added successfully")
                        except Exception as e:
                            print(f"❌ Failed to add UV coordinates: {e}")
                            print(f"🔍 Debug: mesh_prim = {mesh_prim}")
                            # Continue without UV coordinates
                    
                    # Handle materials
                    if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
                        try:
                            material = mesh.visual.material
                            print(f"🔍 Debug: material type = {type(material)}")
                            print(f"🔍 Debug: material attributes = {[attr for attr in dir(material) if not attr.startswith('_')]}")
                            
                            # Create material
                            material_path = "/Root/Materials/Material"
                            usd_material = UsdShade.Material.Define(stage, material_path)
                            
                            # Create shader
                            shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
                            shader.CreateIdAttr("UsdPreviewSurface")
                            
                            # Set basic material properties with null checks
                            if hasattr(material, 'baseColorFactor') and material.baseColorFactor is not None:
                                print(f"🔍 Debug: baseColorFactor = {material.baseColorFactor}")
                                if len(material.baseColorFactor) >= 3:
                                    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                                        Gf.Vec3f(*material.baseColorFactor[:3])
                                    )
                            
                            if hasattr(material, 'roughnessFactor') and material.roughnessFactor is not None:
                                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
                                    float(material.roughnessFactor)
                                )
                            
                            if hasattr(material, 'metallicFactor') and material.metallicFactor is not None:
                                shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
                                    float(material.metallicFactor)
                                )
                            
                            # Connect shader to material
                            usd_material.CreateSurfaceOutput().ConnectToSource(
                                shader.ConnectableAPI(), "surface"
                            )
                            
                            # Bind material to mesh
                            UsdShade.MaterialBindingAPI(mesh_prim).Bind(usd_material)
                            
                            print("✅ Material properties added successfully")
                        except Exception as e:
                            print(f"❌ Failed to add material: {e}")
                            # Continue without material
            
            # Save the USD stage
            stage.Save()
            
            # Convert to USDZ using usdzip if available
            try:
                # Try multiple possible usdzip locations
                usdzip_paths = [
                    "/usr/local/usd/scripts/usdzip.sh",  # Most common location
                    "/usr/local/USD/bin/usdzip",         # Alternative build location
                    "usdzip"                             # System PATH
                ]
                
                usdzip_cmd = None
                for path in usdzip_paths:
                    if path == "usdzip" or os.path.exists(path):
                        usdzip_cmd = path
                        break
                
                if usdzip_cmd:
                    print(f"🔍 Found usdzip at: {usdzip_cmd}")
                    
                    # Set up USD environment for subprocess
                    env = os.environ.copy()
                    env.update({
                        'PATH': '/usr/local/usd/scripts:' + env.get('PATH', ''),
                        'PYTHONPATH': '/usr/local/usd/python:' + env.get('PYTHONPATH', ''),
                        'LD_LIBRARY_PATH': '/usr/local/usd/lib:' + env.get('LD_LIBRARY_PATH', ''),
                    })
                    
                    cmd = [usdzip_cmd, str(output_path), str(temp_usd_path)]
                    print(f"🔧 Running usdzip: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)
                    
                    if result.returncode == 0 and output_path.exists():
                        print(f"✅ Successfully exported USDZ: {output_path}")
                        return True
                    else:
                        print(f"❌ usdzip failed: {result.stderr}")
                        print(f"❌ usdzip stdout: {result.stdout}")
                else:
                    print("⚠️ usdzip not found in expected locations")
                    
            except Exception as e:
                print(f"⚠️ usdzip execution failed: {e}")
                
                # Alternative: Save as USD first, then copy with USDZ extension
                try:
                    # Create a temporary .usd file
                    temp_usd = output_path.with_suffix('.usd')
                    usd_stage = Usd.Stage.CreateNew(str(temp_usd))
                    usd_stage.GetRootLayer().TransferContent(stage.GetRootLayer())
                    usd_stage.Save()
                    
                    if temp_usd.exists():
                        # Copy to .usdz extension (simple single-file USDZ)
                        temp_usd.rename(output_path)
                        print(f"✅ Successfully exported USDZ (USD copy): {output_path}")
                        return True
                        
                except Exception as e:
                    print(f"❌ USD copy export failed: {e}")
            
            # Fallback: Copy as .usd file with .usdz extension
            try:
                import shutil
                usd_output = output_path.with_suffix('.usd')
                shutil.copy2(temp_usd_path, usd_output)
                if usd_output.exists():
                    # Rename to .usdz
                    usd_output.rename(output_path)
                    print(f"✅ Exported as USD with USDZ extension: {output_path}")
                    return True
            except Exception as e:
                print(f"❌ USD fallback export failed: {e}")
            
            return False
            
        finally:
            # Clean up temporary file
            if temp_usd_path.exists():
                temp_usd_path.unlink()
        
    except ImportError as e:
        print(f"❌ USD libraries not available: {e}")
        print("💡 Install with: pip install usd-core")
        return False
    except Exception as e:
        print(f"❌ USDZ export failed: {e}")
        return False
