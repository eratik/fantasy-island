# Runpod Backend Architecture

## Overview

The backend is a Runpod serverless endpoint that accepts a text description of a character and returns a fully rigged `.glb` file. The pipeline has four stages executed sequentially:

```
Text Description
  → [1] Prompt Expansion (LLM)
  → [2] Multi-View Image Generation (Flux GGUF)
  → [3] 3D Mesh Generation (Hunyuan3D 2.1)
  → [4] Auto-Rigging & Optimization (Blender headless)
  → .glb output
```

## Module Responsibilities

### `handler.py` — Runpod Serverless Handler (Orchestrator)

**Purpose:** Entry point for Runpod serverless. Receives requests, orchestrates the pipeline, returns results.

**Interface:**
```python
import runpod

def handler(job: dict) -> dict:
    """Runpod serverless handler entry point."""
    ...

runpod.serverless.start({"handler": handler})
```

**Input schema** (from `job["input"]`):
```python
class JobInput(TypedDict):
    description: str        # Natural language character description
    style: str              # Art style hint: "fantasy", "sci-fi", "realistic", etc.
    poly_target: int        # Target polygon count (default: 30000)
```

**Output schema** (returned from handler):
```python
# On success — return the .glb bytes directly.
# Runpod automatically uploads large outputs (>20MB) to blob storage
# and returns a presigned URL to the caller.
# For smaller outputs, the bytes are returned inline (base64).
#
# Return format:
{
    "glb": <base64-encoded .glb bytes OR presigned URL>,
    "thumbnail": <base64-encoded PNG>,
    "generation_time_seconds": float,
}
```

**Orchestration flow:**
1. Validate and parse input.
2. Call `expand_prompt()` from `generate_images.py` to get structured multi-view prompts.
3. Call `generate_views()` from `generate_images.py` to produce 4 reference images.
4. Call `generate_mesh()` from `generate_mesh.py` to create a textured 3D mesh.
5. Call `auto_rig()` from `auto_rig.py` to rig and optimize the mesh.
6. Read the final `.glb` from disk, generate a thumbnail, return both.

**Error handling:**
- Each stage wraps its work in a try/except. On failure, raise a descriptive exception that propagates to Runpod as a job error.
- Use `runpod.serverless.progress_update(job, {"stage": "...", "progress": N})` to report progress to the client during long operations.
- Temporary files go in a job-specific temp directory (`/tmp/jobs/{job_id}/`) which is cleaned up on completion.

**Key design decisions:**
- No external storage configuration needed. Runpod handles large output upload automatically.
- All inter-module communication is via filesystem paths (images saved to disk, mesh saved to disk, etc.). This keeps interfaces simple and debuggable.
- The handler is synchronous within a single job — Runpod handles concurrency at the container level.

---

### `generate_images.py` — Flux Multi-View Generation

**Purpose:** Takes the user's text description, expands it into structured prompts, and generates 4 consistent character reference images (front, left side, right side, back).

**Public functions:**

```python
def expand_prompt(description: str, style: str) -> list[str]:
    """Expand a short character description into 4 detailed view-specific prompts.

    Uses a local small LLM or template-based approach to generate consistent,
    detailed prompts for each view angle. Each prompt includes:
    - The full character description
    - View-specific angle instruction
    - Style modifiers
    - Technical quality modifiers (white background, full body, character sheet)

    Args:
        description: User's natural language character description.
        style: Art style hint (e.g., "fantasy", "sci-fi").

    Returns:
        List of 4 prompt strings: [front, left, right, back].
    """

def generate_views(
    prompts: list[str],
    output_dir: str,
    seed: int | None = None,
) -> list[str]:
    """Generate 4 reference images from the expanded prompts using Flux.

    Args:
        prompts: 4 view-specific prompts from expand_prompt().
        output_dir: Directory to save generated images.
        seed: Optional seed for reproducibility. If None, a random seed is used.
            The same seed is used for all 4 views to maximize consistency.

    Returns:
        List of 4 file paths to the generated PNG images.
    """
```

**Implementation details:**

- **Model:** Flux GGUF quantized (Q4 or Q5 — balance between quality and VRAM). Load with `diffusers` or the `flux` library with GGUF support.
- **Prompt expansion strategy:** Use a template-based approach (no separate LLM dependency). Build structured prompts by combining:
  - Base description from user
  - View angle prefix: `"front view of"`, `"left side view of"`, `"right side view of"`, `"back view of"`
  - Style suffix: based on `style` parameter
  - Technical suffix: `"white background, full body, character design sheet, high quality, detailed"`
- **Consistency:** Use the same seed across all 4 generations. This is the primary mechanism for visual consistency across views.
- **Resolution:** Generate at 1024x1024 — this is the sweet spot for Flux quality vs. speed.
- **VRAM management:** Load model once at module level (on first call), keep in GPU memory between jobs. Use `torch.cuda.empty_cache()` after generation batch to free intermediate tensors before mesh generation.

**Model loading pattern:**
```python
_pipeline: FluxPipeline | None = None

def _get_pipeline() -> FluxPipeline:
    """Lazy-load the Flux pipeline. Cached across jobs."""
    global _pipeline
    if _pipeline is None:
        _pipeline = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
        ).to("cuda")
    return _pipeline
```

---

### `generate_mesh.py` — Hunyuan3D Mesh Generation

**Purpose:** Takes 4 reference images and generates a textured 3D mesh using Hunyuan3D 2.1's multi-view mode.

**Public functions:**

```python
def generate_mesh(
    image_paths: list[str],
    output_path: str,
    poly_target: int = 30000,
) -> str:
    """Generate a textured 3D mesh from multi-view reference images.

    Uses Hunyuan3D 2.1 in multi-view conditioning mode. The model takes
    the 4 reference images and produces a textured mesh.

    Args:
        image_paths: Paths to 4 reference images [front, left, right, back].
        output_path: Path to write the output .glb file.
        poly_target: Target polygon count. Used to configure Hunyuan3D's
            mesh extraction resolution.

    Returns:
        Path to the generated .glb file (same as output_path).
    """
```

**Implementation details:**

- **Model:** Hunyuan3D 2.1 from Tencent. Use the official `Hunyuan3D` inference pipeline.
- **Input:** 4 reference images. Hunyuan3D 2.1 supports multi-view conditioning natively — feed all 4 images with their camera angle labels.
- **Output:** Raw mesh (likely `.obj` or `.glb`). The model outputs geometry + texture.
- **Mesh extraction:** Use marching cubes at a resolution calibrated to approximate `poly_target`. Higher resolution = more polygons.
- **Texture:** Hunyuan3D generates texture maps (albedo at minimum). Bake these into the output mesh.
- **VRAM management:** Unload Flux from GPU before loading Hunyuan3D. These models cannot coexist on a single L40S (48GB). Pattern:

```python
def _swap_to_mesh_model():
    """Unload image model, load mesh model."""
    from generate_images import unload_model
    unload_model()  # Frees Flux from VRAM
    torch.cuda.empty_cache()
    _load_hunyuan()  # Loads Hunyuan3D into VRAM
```

- **Add `unload_model()` to `generate_images.py`:**
```python
def unload_model() -> None:
    """Unload Flux pipeline from GPU memory."""
    global _pipeline
    if _pipeline is not None:
        _pipeline = None
        torch.cuda.empty_cache()
```

---

### `auto_rig.py` — Blender Headless Rigging

**Purpose:** Takes the raw mesh from Hunyuan3D, adds a humanoid skeleton compatible with Unity's Humanoid avatar system, optionally decimates the mesh, and exports as `.glb`.

**Public functions:**

```python
def auto_rig(
    input_mesh_path: str,
    output_glb_path: str,
    poly_target: int = 30000,
) -> str:
    """Rig a mesh with a humanoid skeleton and export as .glb.

    Runs Blender in headless mode via subprocess. The Blender Python script:
    1. Imports the mesh
    2. Applies decimation if triangle count exceeds poly_target
    3. Creates a humanoid armature with Unity-compatible bone names
    4. Parents the mesh to the armature with automatic weights
    5. Exports as .glb with embedded textures

    Args:
        input_mesh_path: Path to the input mesh (.glb or .obj).
        output_glb_path: Path to write the rigged .glb file.
        poly_target: Target polygon count. Decimate if mesh exceeds this.

    Returns:
        Path to the rigged .glb file (same as output_glb_path).
    """
```

**Implementation approach:**

This module is a Python wrapper that calls Blender as a subprocess. The actual rigging logic lives in an embedded Blender Python script.

```python
def auto_rig(input_mesh_path: str, output_glb_path: str, poly_target: int = 30000) -> str:
    result = subprocess.run(
        [
            "blender", "--background", "--python-expr", _BLENDER_SCRIPT,
            "--", input_mesh_path, output_glb_path, str(poly_target),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Blender auto-rig failed: {result.stderr}")
    return output_glb_path
```

**Blender script responsibilities:**

1. **Import mesh:** `bpy.ops.import_scene.gltf()` or `bpy.ops.wm.obj_import()` depending on input format.

2. **Decimation:** If face count > `poly_target`, apply Decimate modifier with ratio = `poly_target / current_count`.

3. **Create humanoid armature:** Build a skeleton programmatically with Unity-compatible bone names. The bone hierarchy must use these exact names for Unity's Humanoid avatar auto-detection:

```
Hips
├── Spine
│   ├── Chest
│   │   ├── UpperChest
│   │   │   ├── Neck
│   │   │   │   └── Head
│   │   │   ├── LeftShoulder
│   │   │   │   └── LeftUpperArm
│   │   │   │       └── LeftLowerArm
│   │   │   │           └── LeftHand
│   │   │   └── RightShoulder
│   │   │       └── RightUpperArm
│   │   │           └── RightLowerArm
│   │   │               └── RightHand
├── LeftUpperLeg
│   └── LeftLowerLeg
│       └── LeftFoot
│           └── LeftToes
└── RightUpperLeg
    └── RightLowerLeg
        └── RightFoot
            └── RightToes
```

4. **Position bones:** Analyze the mesh bounding box and vertex groups to estimate joint positions. Use heuristics:
   - Hips at ~45% of mesh height
   - Spine at ~55%
   - Chest at ~65%
   - Neck at ~80%
   - Head at ~85-95%
   - Shoulder width from mesh width at chest height
   - Legs: split lower half symmetrically

5. **Parent mesh to armature:** Use `bpy.ops.object.parent_set(type='ARMATURE_AUTO')` for automatic weight painting.

6. **Export:** `bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB', export_animations=False)`

**Why not Rigify/Auto-Rig Pro?** These are addon-based and harder to install in a Docker container. A simple programmatic skeleton with automatic weights is more reliable in a headless environment and sufficient for idle/pose animations.

---

### `Dockerfile` — Container Image

**Base image:** Start from a Runpod base image that includes CUDA and Python.

```dockerfile
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install Blender (headless)
RUN apt-get update && apt-get install -y blender && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen

# Copy application code
COPY handler.py generate_images.py generate_mesh.py auto_rig.py ./

# Pre-download model weights at build time (baked into image)
# This avoids cold-start downloads on first request.
RUN python -c "from generate_images import _get_pipeline; _get_pipeline()"
RUN python -c "from generate_mesh import _load_hunyuan; _load_hunyuan()"

CMD ["python", "-u", "handler.py"]
```

**Key decisions:**
- Model weights are baked into the Docker image. This makes the image large (~30-50GB) but eliminates cold-start latency from downloading weights.
- Blender is installed via apt — the Ubuntu 22.04 repo version is sufficient for our headless scripting needs.
- Use `uv` for Python dependency management, matching the project convention.

---

### `pyproject.toml` — Project Configuration

```toml
[project]
name = "xr-character-backend"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "runpod>=1.7.0",
    "torch>=2.4.0",
    "diffusers>=0.30.0",
    "transformers>=4.40.0",
    "accelerate>=0.30.0",
    "safetensors>=0.4.0",
    "Pillow>=10.0.0",
    "numpy>=1.26.0",
    "trimesh>=4.0.0",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
```

Note: Hunyuan3D dependencies will need to be added based on the specific installation method (pip package vs. cloned repo). The `diffusers` integration may cover it, or a separate `hunyuan3d` package may be needed.

---

## VRAM Management Strategy

The L40S has 48GB VRAM. The two main models cannot coexist in memory:

| Model | Approx VRAM |
|-------|-------------|
| Flux GGUF (Q4) | ~12-16 GB |
| Hunyuan3D 2.1 | ~20-30 GB |
| Blender | CPU only |

**Swap strategy:**
1. Load Flux, generate 4 images (~60-90s).
2. Unload Flux (`del pipeline; torch.cuda.empty_cache()`).
3. Load Hunyuan3D, generate mesh (~120-180s).
4. Unload Hunyuan3D.
5. Run Blender (CPU, subprocess) for rigging (~30-60s).

Total estimated time per character: **3-5 minutes**.

Models are lazy-loaded on first use and explicitly unloaded between stages. This is managed by each module exposing `unload_model()`.

---

## File I/O Convention

All inter-stage files are written to a job-specific temp directory:

```
/tmp/jobs/{job_id}/
├── prompt_front.txt
├── prompt_left.txt
├── prompt_right.txt
├── prompt_back.txt
├── view_front.png
├── view_left.png
├── view_right.png
├── view_back.png
├── raw_mesh.glb
├── rigged_mesh.glb        ← final output
└── thumbnail.png
```

The handler creates this directory at job start and cleans it up after reading the final output.

---

## Error Handling Strategy

Each module raises descriptive exceptions on failure. The handler catches these and returns structured errors to Runpod:

```python
class PipelineError(Exception):
    """Base exception for pipeline errors."""
    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(f"[{stage}] {message}")

class PromptExpansionError(PipelineError): ...
class ImageGenerationError(PipelineError): ...
class MeshGenerationError(PipelineError): ...
class RiggingError(PipelineError): ...
```

The handler maps these to Runpod job errors:
```python
try:
    result = run_pipeline(job_input)
    return result
except PipelineError as e:
    raise  # Runpod captures the exception as a job error
```

---

## Progress Reporting

The handler reports progress to the client at each stage boundary:

```python
runpod.serverless.progress_update(job, {"stage": "expanding_prompt", "progress": 0})
# ... expand prompt ...
runpod.serverless.progress_update(job, {"stage": "generating_images", "progress": 20})
# ... generate images ...
runpod.serverless.progress_update(job, {"stage": "generating_mesh", "progress": 50})
# ... generate mesh ...
runpod.serverless.progress_update(job, {"stage": "rigging", "progress": 80})
# ... rig mesh ...
# return result (progress implicitly 100)
```

The Unity client can poll for these updates to show meaningful loading indicators.

---

## Testing Strategy

**Local testing (no GPU):**
- `handler.py` can be run directly with `python handler.py` — Runpod's SDK supports local test mode where you pass a test input via environment variable.
- Each module can be tested independently by calling its public functions.

**GPU testing:**
- Build the Docker image and run it locally with `--gpus all`.
- Use Runpod's test endpoint feature after deployment.

**Test input:**
```python
TEST_INPUT = {
    "input": {
        "description": "A tall elven warrior with silver hair and leather armor",
        "style": "fantasy",
        "poly_target": 30000,
    }
}
```

---

## Summary of Interfaces

| Module | Input | Output | Runs on |
|--------|-------|--------|---------|
| `handler.py` | Runpod job dict | `.glb` bytes + thumbnail | CPU |
| `generate_images.py` | description, style | 4 PNG image paths | GPU (Flux) |
| `generate_mesh.py` | 4 image paths | mesh `.glb` path | GPU (Hunyuan3D) |
| `auto_rig.py` | mesh path, poly_target | rigged `.glb` path | CPU (Blender) |
