"""Main Runpod serverless handler — orchestrates the character generation pipeline.

Pipeline:
    Text Description
      → [1] Prompt Expansion + Multi-View Image Generation (Flux GGUF)
      → [2] 3D Mesh Generation (Hunyuan3D 2.1)
      → [3] Auto-Rigging & Optimization (Blender headless)
      → .glb output (base64-encoded or presigned URL via Runpod blob storage)
"""

from __future__ import annotations

import base64
import logging
import os
import shutil
import time
from pathlib import Path
from typing import TypedDict

import runpod

from auto_rig import auto_rig
from generate_images import expand_prompt, generate_views
from generate_images import unload_model as unload_image_model
from generate_mesh import generate_mesh
from generate_mesh import unload_model as unload_mesh_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

JOBS_TEMP_DIR = Path("/tmp/jobs")


class JobInput(TypedDict):
    """Schema for the Runpod job input payload."""

    description: str
    style: str
    poly_target: int


class PipelineError(Exception):
    """Base exception for pipeline stage failures."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(f"[{stage}] {message}")


class PromptExpansionError(PipelineError):
    """Raised when prompt expansion fails."""


class ImageGenerationError(PipelineError):
    """Raised when multi-view image generation fails."""


class MeshGenerationError(PipelineError):
    """Raised when 3D mesh generation fails."""


class RiggingError(PipelineError):
    """Raised when Blender auto-rigging fails."""


def _parse_input(raw_input: dict) -> JobInput:
    """Parse and validate the job input payload.

    Args:
        raw_input: Raw dict from ``job["input"]``.

    Returns:
        Validated JobInput with defaults applied.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    description = raw_input.get("description", "").strip()
    if not description:
        raise ValueError("'description' is required and must be a non-empty string.")

    style = raw_input.get("style", "fantasy").strip()
    poly_target = int(raw_input.get("poly_target", 30000))

    if poly_target < 1000 or poly_target > 200000:
        raise ValueError(f"'poly_target' must be between 1000 and 200000, got {poly_target}.")

    return JobInput(description=description, style=style, poly_target=poly_target)


def _make_job_dir(job_id: str) -> Path:
    """Create a temporary working directory for a job.

    Args:
        job_id: Unique Runpod job identifier.

    Returns:
        Path to the created job directory.
    """
    job_dir = JOBS_TEMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def _cleanup_job_dir(job_dir: Path) -> None:
    """Remove the job temp directory and all its contents.

    Args:
        job_dir: Path to the job directory to remove.
    """
    try:
        shutil.rmtree(job_dir)
        logger.info("Cleaned up job directory: %s", job_dir)
    except Exception as exc:
        logger.warning("Failed to clean up job directory %s: %s", job_dir, exc)


def _encode_glb(glb_path: Path) -> str:
    """Read a .glb file and return it as a base64-encoded string.

    Args:
        glb_path: Path to the .glb file.

    Returns:
        Base64-encoded string of the .glb file contents.
    """
    return base64.b64encode(glb_path.read_bytes()).decode("utf-8")


def _generate_thumbnail(image_paths: list[str], output_path: Path) -> str:
    """Create a thumbnail PNG from the front-view reference image.

    Args:
        image_paths: List of 4 view image paths [front, left, right, back].
        output_path: Path to write the thumbnail PNG.

    Returns:
        Base64-encoded string of the thumbnail PNG.
    """
    from PIL import Image

    front_image_path = image_paths[0]
    with Image.open(front_image_path) as img:
        img.thumbnail((512, 512))
        img.save(str(output_path), "PNG")

    return base64.b64encode(output_path.read_bytes()).decode("utf-8")


def run_pipeline(job: dict, job_input: JobInput) -> dict:
    """Execute the full character generation pipeline.

    Stages:
        1. Prompt expansion and multi-view image generation (Flux).
        2. 3D mesh generation (Hunyuan3D).
        3. Auto-rigging and optimization (Blender headless).

    Args:
        job: Full Runpod job dict (used for progress updates).
        job_input: Validated job input parameters.

    Returns:
        Dict with keys: ``glb`` (base64), ``thumbnail`` (base64),
        ``generation_time_seconds`` (float).

    Raises:
        PromptExpansionError: If prompt expansion fails.
        ImageGenerationError: If Flux image generation fails.
        MeshGenerationError: If Hunyuan3D mesh generation fails.
        RiggingError: If Blender auto-rigging fails.
    """
    job_id = job.get("id", "local")
    job_dir = _make_job_dir(job_id)
    start_time = time.monotonic()

    try:
        # --- Stage 1: Prompt expansion ---
        logger.info("[%s] Stage 1: Expanding prompt", job_id)
        runpod.serverless.progress_update(job, {"stage": "expanding_prompt", "progress": 0})
        try:
            prompts = expand_prompt(
                description=job_input["description"],
                style=job_input["style"],
            )
        except Exception as exc:
            raise PromptExpansionError("expand_prompt", str(exc)) from exc

        # --- Stage 2: Multi-view image generation ---
        logger.info("[%s] Stage 2: Generating multi-view images", job_id)
        runpod.serverless.progress_update(job, {"stage": "generating_images", "progress": 10})
        try:
            image_paths = generate_views(
                prompts=prompts,
                output_dir=str(job_dir),
            )
        except Exception as exc:
            raise ImageGenerationError("generate_views", str(exc)) from exc

        # Unload Flux before loading Hunyuan3D to stay within VRAM budget.
        logger.info("[%s] Unloading image model from GPU", job_id)
        unload_image_model()

        # --- Stage 3: 3D mesh generation ---
        logger.info("[%s] Stage 3: Generating 3D mesh", job_id)
        runpod.serverless.progress_update(job, {"stage": "generating_mesh", "progress": 30})
        raw_mesh_path = str(job_dir / "raw_mesh.glb")
        try:
            generate_mesh(
                image_paths=image_paths,
                output_path=raw_mesh_path,
                poly_target=job_input["poly_target"],
            )
        except Exception as exc:
            raise MeshGenerationError("generate_mesh", str(exc)) from exc

        # Unload Hunyuan3D before Blender stage to free VRAM.
        logger.info("[%s] Unloading mesh model from GPU", job_id)
        unload_mesh_model()

        # --- Stage 4: Auto-rigging ---
        logger.info("[%s] Stage 4: Auto-rigging mesh", job_id)
        runpod.serverless.progress_update(job, {"stage": "rigging", "progress": 75})
        rigged_glb_path = job_dir / "rigged_mesh.glb"
        try:
            auto_rig(
                input_mesh_path=raw_mesh_path,
                output_glb_path=str(rigged_glb_path),
                poly_target=job_input["poly_target"],
            )
        except Exception as exc:
            raise RiggingError("auto_rig", str(exc)) from exc

        # --- Finalise output ---
        logger.info("[%s] Stage 5: Encoding output", job_id)
        runpod.serverless.progress_update(job, {"stage": "encoding_output", "progress": 95})

        glb_b64 = _encode_glb(rigged_glb_path)
        thumbnail_path = job_dir / "thumbnail.png"
        thumbnail_b64 = _generate_thumbnail(image_paths, thumbnail_path)

        elapsed = time.monotonic() - start_time
        logger.info("[%s] Pipeline completed in %.1fs", job_id, elapsed)

        return {
            "glb": glb_b64,
            "thumbnail": thumbnail_b64,
            "generation_time_seconds": round(elapsed, 1),
        }

    finally:
        _cleanup_job_dir(job_dir)


def handler(job: dict) -> dict:
    """Runpod serverless handler entry point.

    Receives a job dict from Runpod, validates input, runs the character
    generation pipeline, and returns the result.

    Args:
        job: Runpod job dict. Expected ``job["input"]`` keys:
            - ``description`` (str): Natural language character description.
            - ``style`` (str, optional): Art style hint. Default ``"fantasy"``.
            - ``poly_target`` (int, optional): Target polygon count. Default 30000.

    Returns:
        Dict with:
            - ``glb``: Base64-encoded .glb file (Runpod may convert to presigned URL).
            - ``thumbnail``: Base64-encoded PNG preview.
            - ``generation_time_seconds``: Total pipeline duration.

    Raises:
        PipelineError: On any stage failure — Runpod captures this as a job error.
    """
    logger.info("Job received: id=%s", job.get("id"))

    try:
        job_input = _parse_input(job.get("input", {}))
    except ValueError as exc:
        raise ValueError(f"Invalid job input: {exc}") from exc

    logger.info(
        "Starting pipeline: description='%s', style='%s', poly_target=%d",
        job_input["description"][:80],
        job_input["style"],
        job_input["poly_target"],
    )

    return run_pipeline(job, job_input)


if __name__ == "__main__":
    # Local test mode — Runpod SDK reads RUNPOD_WEBHOOK_GET_JOB from env,
    # but also supports direct invocation with a test input.
    test_input = {
        "input": {
            "description": "A tall elven warrior with silver hair and leather armor",
            "style": "fantasy",
            "poly_target": 30000,
        }
    }
    local_api_params = {
        "handler": handler,
    }
    if os.getenv("RUNPOD_LOCAL_TEST"):
        local_api_params["test_input"] = test_input  # type: ignore[assignment]

    runpod.serverless.start(local_api_params)
