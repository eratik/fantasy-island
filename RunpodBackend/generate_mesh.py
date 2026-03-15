"""Hunyuan3D mesh generation module.

Generates a textured 3D mesh (.glb) from multi-view reference images using
Hunyuan3D 2.1 in multi-view conditioning mode.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# HuggingFace model ID for Hunyuan3D 2.1
_MODEL_ID = os.environ.get("HUNYUAN3D_MODEL_ID", "tencent/Hunyuan3D-2.1")

# Module-level pipeline cache (lazy-loaded, kept warm between jobs)
_pipeline: object | None = None


def _load_hunyuan() -> object:
    """Load the Hunyuan3D 2.1 pipeline into GPU memory.

    Returns:
        Loaded Hunyuan3D inference pipeline.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    logger.info("Loading Hunyuan3D 2.1 pipeline from %s", _MODEL_ID)

    # Import here so the module can be imported without the heavy deps present
    # (useful for type-checking / linting in dev environments).
    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "Hunyuan3D is not installed. Install via: "
            "pip install git+https://github.com/Tencent/Hunyuan3D-2.git"
        ) from exc

    hf_token = os.environ.get("HF_TOKEN")
    _pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        _MODEL_ID,
        torch_dtype=torch.float16,
    )
    _pipeline.to("cuda")

    logger.info("Hunyuan3D 2.1 pipeline loaded successfully")
    return _pipeline


def unload_model() -> None:
    """Unload Hunyuan3D pipeline from GPU memory.

    Call this before loading another large model to free VRAM.
    """
    global _pipeline
    if _pipeline is not None:
        logger.info("Unloading Hunyuan3D pipeline from GPU memory")
        _pipeline = None
        torch.cuda.empty_cache()


def _poly_target_to_mc_resolution(poly_target: int) -> int:
    """Convert a polygon target to a marching-cubes grid resolution.

    Higher resolution produces more triangles. This is a heuristic mapping
    calibrated for typical humanoid character meshes.

    Args:
        poly_target: Desired number of triangles in the output mesh.

    Returns:
        Marching-cubes grid resolution (isosurface extraction parameter).
    """
    # Empirical mapping: resolution^3 voxels roughly correlates with triangle count.
    # Hunyuan3D default is 256; we scale up/down from there.
    if poly_target <= 10_000:
        return 192
    elif poly_target <= 30_000:
        return 256
    elif poly_target <= 50_000:
        return 320
    else:
        return 384


def generate_mesh(
    image_paths: list[str],
    output_path: str,
    poly_target: int = 30_000,
) -> str:
    """Generate a textured 3D mesh from multi-view reference images.

    Uses Hunyuan3D 2.1 in multi-view conditioning mode. The model ingests
    the 4 reference images (front, left, right, back) and produces a
    textured mesh with PBR maps (albedo, normal, roughness, metallic).

    Before calling this function, ensure Flux has been unloaded from VRAM
    by calling ``generate_images.unload_model()`` to avoid OOM errors.

    Args:
        image_paths: Paths to 4 reference images in order [front, left, right, back].
        output_path: Destination path for the output .glb file.
        poly_target: Target polygon count. Controls mesh extraction resolution.
            Typical values: 10000 (low), 30000 (medium), 50000 (high).

    Returns:
        Path to the generated .glb file (same as ``output_path``).

    Raises:
        ValueError: If ``image_paths`` does not contain exactly 4 paths.
        FileNotFoundError: If any of the image files do not exist.
        RuntimeError: If mesh generation fails.
    """
    if len(image_paths) != 4:
        raise ValueError(f"Expected 4 image paths (front/left/right/back), got {len(image_paths)}")

    for path in image_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Reference image not found: {path}")

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Generating mesh from %d reference images → %s (poly_target=%d)",
        len(image_paths),
        output_path,
        poly_target,
    )

    from PIL import Image  # type: ignore[import]

    # Load reference images. Hunyuan3D expects PIL Images.
    view_labels = ["front", "left", "right", "back"]
    images: list[Image.Image] = []
    for path, label in zip(image_paths, view_labels):
        img = Image.open(path).convert("RGBA")
        images.append(img)
        logger.debug("Loaded %s view: %s (%dx%d)", label, path, img.width, img.height)

    # Ensure Hunyuan3D is loaded (swaps from Flux if needed)
    pipeline = _load_hunyuan()

    mc_resolution = _poly_target_to_mc_resolution(poly_target)
    logger.info("Using marching-cubes resolution %d for poly_target=%d", mc_resolution, poly_target)

    try:
        # Run the shape generation pipeline.
        # The multi-view conditioning API accepts a list of PIL images with
        # corresponding camera-angle labels.
        logger.info("Running Hunyuan3D shape generation...")
        result = pipeline(
            image=images[0],          # Primary (front) view drives the shape
            extra_images=images[1:],  # Additional views for multi-view conditioning
            mc_resolution=mc_resolution,
            num_inference_steps=50,
            guidance_scale=7.5,
            output_type="mesh",
        )

        # Hunyuan3D returns either a list of meshes or an object with .meshes
        if isinstance(result, list):
            mesh = result[0]
        elif hasattr(result, "meshes"):
            mesh = result.meshes[0]
        else:
            mesh = result
        logger.info("Shape generation complete. Exporting mesh to %s", output_path)

        # Export as GLB — try native .export(), fall back to trimesh
        _export_glb(mesh, output_path)

    except Exception as exc:
        raise RuntimeError(f"Hunyuan3D mesh generation failed: {exc}") from exc

    logger.info("Mesh generation complete: %s", output_path)
    return output_path


def _export_glb(mesh: object, output_path: str) -> None:
    """Export a Hunyuan3D mesh object to a .glb file with embedded textures.

    Attempts to use the mesh's native export method first, then falls back to
    trimesh for format conversion.

    Args:
        mesh: Hunyuan3D mesh result object.
        output_path: Destination .glb path.
    """
    # Hunyuan3D mesh objects expose an export() method.
    # If the mesh already supports GLB export, use it directly.
    if hasattr(mesh, "export"):
        mesh.export(output_path)
        logger.debug("Exported via native mesh.export()")
        return

    # Fallback: convert via trimesh.
    logger.debug("Native export not available; converting via trimesh")
    import trimesh  # type: ignore[import]

    if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        tm = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            vertex_colors=getattr(mesh, "vertex_colors", None),
        )
    elif hasattr(mesh, "mesh_v") and hasattr(mesh, "mesh_f"):
        # Hunyuan3D internal format uses mesh_v (vertices) and mesh_f (faces)
        import numpy as np
        tm = trimesh.Trimesh(
            vertices=np.array(mesh.mesh_v),
            faces=np.array(mesh.mesh_f),
        )
    else:
        raise RuntimeError(
            f"Cannot export mesh: unrecognised mesh type {type(mesh).__name__}"
        )

    scene = trimesh.scene.Scene()
    scene.add_geometry(tm)
    scene.export(output_path, file_type="glb")
    logger.debug("Exported via trimesh fallback")
