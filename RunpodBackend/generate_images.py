"""Flux multi-view image generation module.

Generates 4 consistent character reference images (front, left, right, back)
from a text description using a GGUF-quantized Flux model.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# HuggingFace model ID for Flux. Override via environment variable to switch
# between full-precision and GGUF variants.
_MODEL_ID: str = os.environ.get(
    "FLUX_MODEL_ID",
    "black-forest-labs/FLUX.1-schnell",  # Default to Schnell (faster, lower VRAM)
)

# Image resolution. 1024x1024 is the Flux sweet spot for quality vs. speed.
_IMAGE_SIZE: int = int(os.environ.get("FLUX_IMAGE_SIZE", "1024"))

# Number of inference steps. Schnell works well at 4; dev needs ~20-50.
_NUM_STEPS: int = int(os.environ.get("FLUX_NUM_STEPS", "4"))

# Style suffix templates keyed by style name.
_STYLE_SUFFIXES: dict[str, str] = {
    "fantasy": (
        "fantasy art style, dramatic lighting, highly detailed, "
        "digital painting, artstation quality"
    ),
    "sci-fi": (
        "sci-fi concept art, hard surface details, futuristic, "
        "digital painting, artstation quality"
    ),
    "realistic": (
        "photorealistic, cinematic lighting, ultra-detailed, "
        "8K, physically based rendering"
    ),
    "anime": (
        "anime art style, cel shading, vibrant colors, "
        "clean linework, studio quality"
    ),
    "cartoon": (
        "cartoon art style, bold outlines, bright colors, "
        "clean and friendly aesthetic"
    ),
}

_DEFAULT_STYLE_SUFFIX = "high quality, detailed, digital art, artstation quality"

# Technical suffix applied to all views for clean backgrounds and full-body shots.
_TECHNICAL_SUFFIX = (
    "pure white background, full body character, character design sheet, "
    "isolated on white, no shadow, orthographic front view"
)

# View angle prefixes.
_VIEW_PREFIXES: list[tuple[str, str]] = [
    ("front", "front view of"),
    ("left", "left side view of"),
    ("right", "right side view of"),
    ("back", "back view of"),
]

# ---------------------------------------------------------------------------
# Module-level pipeline cache
# ---------------------------------------------------------------------------

_pipeline: object | None = None


def _get_pipeline() -> object:
    """Lazy-load the Flux pipeline and cache it across jobs.

    The pipeline is loaded once and kept in GPU memory to avoid cold-start
    overhead on subsequent requests.

    Returns:
        Loaded FluxPipeline instance.

    Raises:
        ImportError: If ``diffusers`` is not installed.
        RuntimeError: If CUDA is not available.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Flux requires a GPU with CUDA support."
        )

    logger.info("Loading Flux pipeline from %s", _MODEL_ID)

    try:
        from diffusers import FluxPipeline  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "diffusers is not installed. Run: pip install diffusers"
        ) from exc

    hf_token = os.environ.get("HF_TOKEN")

    _pipeline = FluxPipeline.from_pretrained(
        _MODEL_ID,
        torch_dtype=torch.float16,
        token=hf_token,
    ).to("cuda")

    logger.info("Flux pipeline loaded successfully (model=%s)", _MODEL_ID)
    return _pipeline


def unload_model() -> None:
    """Unload the Flux pipeline from GPU memory.

    Call this before loading Hunyuan3D to free VRAM on the L40S (48GB).
    The next call to ``_get_pipeline()`` will reload from the HuggingFace
    cache (fast if weights are baked into the Docker image).
    """
    global _pipeline
    if _pipeline is not None:
        logger.info("Unloading Flux pipeline from GPU memory")
        _pipeline = None
        torch.cuda.empty_cache()
        logger.info("GPU memory freed")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def expand_prompt(description: str, style: str) -> list[str]:
    """Expand a short character description into 4 detailed view-specific prompts.

    Uses a template-based approach — no separate LLM dependency. Each prompt
    combines the user's description, a view-angle prefix, a style suffix, and
    a technical suffix (white background, full body, character design sheet).

    The same base description is used across all views so Flux (with a shared
    seed) produces visually consistent results.

    Args:
        description: User's natural language character description.
        style: Art style hint (e.g., ``"fantasy"``, ``"sci-fi"``).

    Returns:
        List of 4 prompt strings in order: [front, left, right, back].
    """
    style_suffix = _STYLE_SUFFIXES.get(style.lower(), _DEFAULT_STYLE_SUFFIX)
    prompts: list[str] = []

    for _, view_prefix in _VIEW_PREFIXES:
        prompt = (
            f"{view_prefix} {description}, "
            f"{style_suffix}, "
            f"{_TECHNICAL_SUFFIX}"
        )
        prompts.append(prompt)
        logger.debug("Prompt: %s", prompt[:120])

    return prompts


def generate_views(
    prompts: list[str],
    output_dir: str,
    seed: int | None = None,
) -> list[str]:
    """Generate 4 reference images from the expanded prompts using Flux.

    All 4 images are generated with the same random seed to maximise visual
    consistency across view angles. Images are saved as PNGs to ``output_dir``.

    Args:
        prompts: 4 view-specific prompts from :func:`expand_prompt`.
            Order must be [front, left, right, back].
        output_dir: Directory to save generated PNG images.
        seed: Optional seed for reproducibility. If ``None``, a random seed
            is chosen and logged so results can be reproduced.

    Returns:
        List of 4 absolute file paths to the generated PNG images,
        in order: [front, left, right, back].

    Raises:
        ValueError: If ``prompts`` does not contain exactly 4 items.
        RuntimeError: If CUDA is unavailable or image generation fails.
    """
    if len(prompts) != 4:
        raise ValueError(f"Expected 4 prompts (front/left/right/back), got {len(prompts)}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    logger.info("Using generation seed: %d", seed)

    pipeline = _get_pipeline()
    image_paths: list[str] = []

    for i, ((view_name, _), prompt) in enumerate(zip(_VIEW_PREFIXES, prompts)):
        logger.info("Generating %s view (%d/4)...", view_name, i + 1)

        generator = torch.Generator(device="cuda").manual_seed(seed)

        try:
            result = pipeline(  # type: ignore[operator]
                prompt=prompt,
                height=_IMAGE_SIZE,
                width=_IMAGE_SIZE,
                num_inference_steps=_NUM_STEPS,
                generator=generator,
                guidance_scale=0.0,  # Flux-schnell uses guidance_scale=0
            )
        except Exception as exc:
            raise RuntimeError(
                f"Flux image generation failed for {view_name} view: {exc}"
            ) from exc

        image = result.images[0]
        out_file = output_path / f"view_{view_name}.png"
        image.save(str(out_file))
        image_paths.append(str(out_file))
        logger.info("Saved %s view → %s", view_name, out_file)

    # Free intermediate GPU tensors before the next stage.
    torch.cuda.empty_cache()

    logger.info("All 4 reference images generated in %s", output_dir)
    return image_paths
