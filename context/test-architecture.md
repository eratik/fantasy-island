# Test Architecture — RunpodBackend

## Overview

This document defines the test strategy for the `RunpodBackend/` Python codebase. The pipeline has four modules (`handler.py`, `generate_images.py`, `generate_mesh.py`, `auto_rig.py`) that run sequentially on a GPU server. Tests are split into two tiers:

- **Unit tests** — run locally on any machine (no GPU, no Blender, no model weights). All heavy dependencies are mocked.
- **Integration tests** — run on a GPU machine with Blender installed. Exercise real models and the real Blender subprocess.

## Directory Structure

```
RunpodBackend/
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures, mock factories, pytest config
│   ├── fixtures/                # Static test data
│   │   ├── tiny_mesh.glb        # Minimal valid GLB (a cube with 12 tris)
│   │   ├── rigged_mesh.glb      # Minimal valid GLB with armature + 22 bones
│   │   └── view_front.png       # 64x64 white-background test image
│   ├── test_handler.py          # Unit tests for handler.py
│   ├── test_generate_images.py  # Unit tests for generate_images.py
│   ├── test_generate_mesh.py    # Unit tests for generate_mesh.py
│   ├── test_auto_rig.py         # Unit tests for auto_rig.py
│   ├── test_glb_validator.py    # Tests for the GLB validation utility itself
│   └── integration/
│       ├── __init__.py
│       ├── test_auto_rig_blender.py      # Real Blender subprocess
│       ├── test_generate_images_gpu.py   # Real Flux inference
│       ├── test_generate_mesh_gpu.py     # Real Hunyuan3D inference
│       └── test_pipeline_e2e.py          # Full pipeline, text → rigged .glb
├── handler.py
├── generate_images.py
├── generate_mesh.py
├── auto_rig.py
└── pyproject.toml
```

## pytest Configuration

Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "gpu: requires CUDA GPU (deselect with '-m not gpu')",
    "blender: requires Blender installed (deselect with '-m not blender')",
    "integration: full integration test (deselect with '-m not integration')",
    "slow: takes >30s to run",
]
```

### Running Tests

```bash
# All unit tests (no GPU, no Blender needed):
uv run pytest tests/ -m "not gpu and not blender and not integration"

# Just unit tests (shorthand — same as above since unit tests have no markers):
uv run pytest tests/ --ignore=tests/integration

# Integration: auto-rig only (needs Blender, no GPU):
uv run pytest tests/integration/test_auto_rig_blender.py -m blender

# Integration: GPU tests (needs CUDA):
uv run pytest tests/integration/ -m gpu

# Full pipeline end-to-end (needs GPU + Blender):
uv run pytest tests/integration/test_pipeline_e2e.py -m integration

# Everything:
uv run pytest tests/
```

## Dependencies

Add to `pyproject.toml` under a `[dependency-groups]` section:

```toml
[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-mock>=3.14",
]
```

No additional test dependencies are needed. The project already depends on `Pillow`, `trimesh`, and `numpy`, which are sufficient for fixture generation and GLB validation.

## Unit Test Strategy

Unit tests mock all heavy dependencies (GPU models, Blender subprocess) and test the logic of each module in isolation.

### `test_handler.py` — Handler / Orchestrator

Mocks: `expand_prompt`, `generate_views`, `generate_mesh`, `auto_rig`, `unload_model` (both), `runpod.serverless.progress_update`.

| Test | What it verifies |
|------|-----------------|
| `test_parse_input_valid` | `_parse_input` returns correct `JobInput` with all fields |
| `test_parse_input_defaults` | Missing `style` defaults to `"fantasy"`, `poly_target` to 30000 |
| `test_parse_input_empty_description` | Raises `ValueError` for empty/missing description |
| `test_parse_input_poly_target_out_of_range` | Raises `ValueError` for poly_target < 1000 or > 200000 |
| `test_parse_input_poly_target_cast` | String poly_target is cast to int |
| `test_handler_success` | Full handler call with mocked pipeline returns `{glb, thumbnail, generation_time_seconds}` |
| `test_handler_propagates_pipeline_error` | Each `PipelineError` subclass propagates correctly |
| `test_handler_progress_updates` | `progress_update` is called at each stage boundary with correct stage names |
| `test_handler_cleans_up_job_dir` | Job temp dir is removed even on failure |
| `test_make_job_dir_creates_nested` | `_make_job_dir` creates `/tmp/jobs/{id}/` |
| `test_encode_glb_roundtrip` | `_encode_glb` produces valid base64 that decodes to original bytes |
| `test_generate_thumbnail` | Produces a 512x512 (max) PNG from a larger input image |

### `test_generate_images.py` — Flux Multi-View Generation

Mocks: `FluxPipeline.from_pretrained`, `torch.cuda.is_available`, `torch.cuda.empty_cache`.

| Test | What it verifies |
|------|-----------------|
| `test_expand_prompt_structure` | Returns exactly 4 prompts |
| `test_expand_prompt_view_prefixes` | Each prompt starts with the correct view prefix |
| `test_expand_prompt_contains_description` | User description appears in every prompt |
| `test_expand_prompt_style_suffix` | Known styles apply their suffix; unknown styles get default |
| `test_expand_prompt_all_styles` | Each key in `_STYLE_SUFFIXES` produces the expected suffix |
| `test_generate_views_saves_4_pngs` | 4 PNG files are written to `output_dir` with correct names |
| `test_generate_views_wrong_prompt_count` | Raises `ValueError` if len(prompts) != 4 |
| `test_generate_views_uses_same_seed` | The `torch.Generator.manual_seed` is called with the same seed for all 4 views |
| `test_generate_views_random_seed_when_none` | A seed is generated when `seed=None` |
| `test_get_pipeline_no_cuda` | Raises `RuntimeError` when `torch.cuda.is_available()` is False |
| `test_unload_model_clears_pipeline` | After `unload_model()`, `_pipeline` is None and `empty_cache` was called |
| `test_unload_model_noop_when_none` | `unload_model()` does nothing if pipeline was never loaded |

### `test_generate_mesh.py` — Hunyuan3D Mesh Generation

Mocks: `Hunyuan3DDiTFlowMatchingPipeline.from_pretrained`, `torch.cuda.is_available`, `PIL.Image.open`.

| Test | What it verifies |
|------|-----------------|
| `test_generate_mesh_success` | Writes a .glb file to `output_path`, returns that path |
| `test_generate_mesh_wrong_image_count` | Raises `ValueError` for len != 4 |
| `test_generate_mesh_missing_image` | Raises `FileNotFoundError` for nonexistent image path |
| `test_generate_mesh_creates_output_dir` | Parent directory of `output_path` is created if absent |
| `test_poly_target_to_mc_resolution` | Each poly_target range maps to the correct resolution (192/256/320/384) |
| `test_export_glb_native` | `_export_glb` calls `mesh.export()` when available |
| `test_export_glb_trimesh_fallback` | `_export_glb` falls back to trimesh when no `.export()` method |
| `test_export_glb_unknown_mesh_type` | Raises `RuntimeError` for mesh with no `.export()`, `.vertices`, or `.faces` |
| `test_unload_model` | Sets `_pipeline` to None and calls `empty_cache` |

### `test_auto_rig.py` — Blender Headless Rigging

Mocks: `subprocess.run`, `shutil.which`.

| Test | What it verifies |
|------|-----------------|
| `test_auto_rig_success` | Returns `output_glb_path` when Blender exits 0 and output file exists |
| `test_auto_rig_input_not_found` | Raises `FileNotFoundError` for missing input mesh |
| `test_auto_rig_blender_not_found` | Raises `FileNotFoundError` when Blender is not on PATH |
| `test_auto_rig_blender_env_var` | Respects `BLENDER_PATH` environment variable |
| `test_auto_rig_blender_failure` | Raises `RuntimeError` when Blender exits non-zero |
| `test_auto_rig_timeout` | Raises `TimeoutError` when Blender exceeds 300s |
| `test_auto_rig_output_missing_after_success` | Raises `RuntimeError` when Blender exits 0 but output file missing |
| `test_auto_rig_creates_output_dir` | Parent directory of output path is created |
| `test_blender_script_has_all_22_bones` | The `_BLENDER_SCRIPT` string contains all 22 Unity bone names |
| `test_blender_script_bone_hierarchy` | Parse the script to verify parent-child relationships are correct |

## GLB Validation Utility

A shared utility in `tests/conftest.py` that parses `.glb` files and validates Unity-compatible skeleton structure. This is used by both unit tests (on fixture files) and integration tests (on pipeline output).

### Implementation: `validate_glb()`

```python
def validate_glb(glb_path: str) -> dict:
    """Parse a .glb file and validate it has a Unity-compatible humanoid skeleton.

    Returns a dict with validation results:
        {
            "valid": bool,
            "errors": list[str],
            "bone_names": list[str],       # bones found in the GLB
            "missing_bones": list[str],     # required bones not found
            "extra_bones": list[str],       # bones found but not required
            "triangle_count": int,
            "has_mesh": bool,
            "has_skin": bool,              # mesh is skinned (has joint weights)
        }
    """
```

### Required Unity Bone Names (22 total)

```python
UNITY_HUMANOID_BONES = [
    "Hips",
    "Spine",
    "Chest",
    "UpperChest",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftUpperArm",
    "LeftLowerArm",
    "LeftHand",
    "RightShoulder",
    "RightUpperArm",
    "RightLowerArm",
    "RightHand",
    "LeftUpperLeg",
    "LeftLowerLeg",
    "LeftFoot",
    "LeftToes",
    "RightUpperLeg",
    "RightLowerLeg",
    "RightFoot",
    "RightToes",
]
```

### GLB Parsing Approach

Use `trimesh` (already a project dependency) to load the GLB:

```python
import trimesh

scene = trimesh.load(glb_path)
```

For skeleton validation, parse the raw glTF JSON from the GLB binary. The GLB format is:
1. 12-byte header (magic + version + length)
2. JSON chunk (type 0x4E4F534A) containing the glTF scene description
3. Binary chunk (type 0x004E4942) containing buffers

Extract bone names from `gltf_json["skins"][0]["joints"]` cross-referenced with `gltf_json["nodes"]`. Verify:
- All 22 bone names are present
- A skin exists (mesh is bound to skeleton)
- At least one mesh primitive exists
- Triangle count is within expected range

### `test_glb_validator.py`

| Test | What it verifies |
|------|-----------------|
| `test_validate_rigged_fixture` | The `rigged_mesh.glb` fixture passes full validation |
| `test_validate_unrigged_fixture` | The `tiny_mesh.glb` fixture (no armature) reports `has_skin=False`, `missing_bones=all 22` |
| `test_validate_nonexistent_file` | Raises `FileNotFoundError` |
| `test_validate_corrupt_file` | Reports errors for invalid GLB data |
| `test_all_22_bones_checked` | `UNITY_HUMANOID_BONES` list has exactly 22 entries |

## Fixtures

### `conftest.py` — Shared Fixtures

```python
@pytest.fixture
def tmp_job_dir(tmp_path):
    """Create a temporary job directory mirroring /tmp/jobs/{id}/."""
    job_dir = tmp_path / "test-job-001"
    job_dir.mkdir()
    return job_dir

@pytest.fixture
def sample_job_input():
    """Return a valid JobInput dict."""
    return {
        "description": "A tall elven warrior with silver hair and leather armor",
        "style": "fantasy",
        "poly_target": 30000,
    }

@pytest.fixture
def sample_job(sample_job_input):
    """Return a valid Runpod job dict."""
    return {"id": "test-job-001", "input": sample_job_input}

@pytest.fixture
def four_test_images(tmp_job_dir):
    """Create 4 minimal PNG files in the job dir and return their paths."""
    from PIL import Image
    paths = []
    for name in ["view_front.png", "view_left.png", "view_right.png", "view_back.png"]:
        path = tmp_job_dir / name
        img = Image.new("RGBA", (64, 64), (255, 255, 255, 255))
        img.save(str(path))
        paths.append(str(path))
    return paths

@pytest.fixture
def tiny_glb(tmp_path):
    """Create a minimal valid GLB file (cube mesh, no skeleton)."""
    # Use trimesh to generate a minimal GLB
    import trimesh
    mesh = trimesh.creation.box(extents=(1, 1, 2))  # humanoid-ish proportions
    path = tmp_path / "tiny_mesh.glb"
    mesh.export(str(path), file_type="glb")
    return str(path)

@pytest.fixture
def fixture_dir():
    """Return path to the static fixtures directory."""
    return Path(__file__).parent / "fixtures"
```

### Static Fixture Files (`tests/fixtures/`)

- **`tiny_mesh.glb`** — A trimesh-generated cube exported as GLB. Used to test mesh import paths and the "no skeleton" validation case.
- **`rigged_mesh.glb`** — A GLB with a simple cube mesh + a 22-bone humanoid armature matching Unity naming. Created by the Blender integration test and committed as a fixture, OR generated programmatically using trimesh + manual glTF JSON construction.
- **`view_front.png`** — A 64x64 white PNG. Used where a real image is expected but content doesn't matter.

## Integration Test Strategy

All integration tests are marked with `@pytest.mark.integration` and additionally with `@pytest.mark.gpu` or `@pytest.mark.blender` as appropriate.

### `test_auto_rig_blender.py` — Blender Integration

**Requires:** Blender installed (`blender` on PATH or `BLENDER_PATH` set).
**Marker:** `@pytest.mark.blender`

| Test | What it verifies |
|------|-----------------|
| `test_rig_cube_mesh` | Pass `tiny_mesh.glb` through real `auto_rig()`. Validate output GLB has all 22 bones, a skin, and correct bone hierarchy |
| `test_rig_with_decimation` | Pass a high-poly mesh with `poly_target=1000`. Verify triangle count decreased |
| `test_rig_obj_input` | Test with `.obj` format input |
| `test_rig_fbx_input` | Test with `.fbx` format input |
| `test_output_glb_valid_for_unity` | Run `validate_glb()` on the output and assert `valid=True`, `missing_bones=[]` |

### `test_generate_images_gpu.py` — Flux Integration

**Requires:** CUDA GPU, Flux model weights.
**Marker:** `@pytest.mark.gpu`

| Test | What it verifies |
|------|-----------------|
| `test_generate_4_views` | `generate_views()` produces 4 PNG files, each 1024x1024 |
| `test_seed_reproducibility` | Same seed produces identical images |
| `test_unload_frees_vram` | After `unload_model()`, VRAM usage drops significantly |

### `test_generate_mesh_gpu.py` — Hunyuan3D Integration

**Requires:** CUDA GPU, Hunyuan3D model weights.
**Marker:** `@pytest.mark.gpu`

| Test | What it verifies |
|------|-----------------|
| `test_generate_mesh_from_images` | Real images → real mesh → valid GLB on disk |
| `test_mesh_poly_count_matches_target` | Triangle count is within 2x of `poly_target` |

### `test_pipeline_e2e.py` — Full End-to-End

**Requires:** CUDA GPU + Blender.
**Marker:** `@pytest.mark.integration`

| Test | What it verifies |
|------|-----------------|
| `test_full_pipeline` | Text description → handler → base64 GLB output. Decode the GLB, run `validate_glb()`, assert all 22 bones present, mesh has triangles, skin exists |

## Mocking Guidelines

### What to mock (unit tests)

- **ML model pipelines** (`FluxPipeline`, `Hunyuan3DDiTFlowMatchingPipeline`): Mock `from_pretrained` to return a callable that returns a result object with `.images` or `.meshes`.
- **`torch.cuda`**: Mock `is_available()` to return True (or False for error-path tests). Mock `empty_cache()` as a no-op.
- **`subprocess.run`** (Blender): Return a `CompletedProcess` with returncode=0 and create the expected output file via a side_effect.
- **`runpod.serverless.progress_update`**: Mock as a no-op; assert it was called with expected stage names.
- **Filesystem**: Use `tmp_path` (pytest built-in) instead of real `/tmp/jobs/`.

### What NOT to mock (unit tests)

- **`_parse_input`** — Pure logic, no dependencies. Test directly.
- **`expand_prompt`** — Pure string templating. Test directly.
- **`_poly_target_to_mc_resolution`** — Pure mapping function. Test directly.
- **`_encode_glb`** / **`_generate_thumbnail`** — Only depends on Pillow (available in dev). Test with real images.
- **File I/O** — Use `tmp_path` fixtures for real file operations.

### Mock Factory Pattern

In `conftest.py`, provide reusable mock factories:

```python
@pytest.fixture
def mock_flux_pipeline(mocker):
    """Mock the Flux pipeline with a callable that returns fake images."""
    mock_img = Image.new("RGB", (1024, 1024), (128, 128, 128))
    mock_result = mocker.MagicMock()
    mock_result.images = [mock_img]

    mock_pipeline = mocker.MagicMock()
    mock_pipeline.return_value = mock_result
    mock_pipeline.to.return_value = mock_pipeline

    mocker.patch(
        "generate_images.FluxPipeline.from_pretrained",
        return_value=mock_pipeline,
    )
    # Ensure CUDA appears available
    mocker.patch("generate_images.torch.cuda.is_available", return_value=True)
    mocker.patch("generate_images.torch.cuda.empty_cache")

    return mock_pipeline

@pytest.fixture
def mock_blender_success(mocker, tmp_path):
    """Mock subprocess.run to simulate a successful Blender execution."""
    def side_effect(*args, **kwargs):
        # Create the output file that auto_rig expects
        cmd = args[0]
        separator_idx = cmd.index("--")
        output_path = cmd[separator_idx + 2]  # second arg after "--"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake-glb-content")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    mock = mocker.patch("auto_rig.subprocess.run", side_effect=side_effect)
    mocker.patch("auto_rig.shutil.which", return_value="/usr/bin/blender")
    return mock
```

## Blender Script Verification (Unit Test, No Blender Required)

The `_BLENDER_SCRIPT` string in `auto_rig.py` is the most critical piece — it defines the bone names and hierarchy that Unity depends on. Unit tests verify this string statically:

1. **All 22 bone names present:** Parse `_BLENDER_SCRIPT` for `make_bone("BoneName"` calls and assert all 22 names from `UNITY_HUMANOID_BONES` appear.
2. **Correct parent-child hierarchy:** Parse `parent_name="..."` arguments and verify the tree matches the expected hierarchy (e.g., `Spine` is parented to `Hips`, `Chest` to `Spine`, etc.).
3. **Symmetry:** Left and Right variants are both present for all limb bones.

## Error Path Testing

Each module has specific failure modes to test:

| Module | Error case | Expected behavior |
|--------|-----------|-------------------|
| `handler.py` | Empty description | `ValueError` before pipeline starts |
| `handler.py` | Stage failure | Correct `PipelineError` subclass, job dir cleaned up |
| `generate_images.py` | No CUDA | `RuntimeError` from `_get_pipeline()` |
| `generate_images.py` | Wrong prompt count | `ValueError` from `generate_views()` |
| `generate_mesh.py` | Missing image file | `FileNotFoundError` |
| `generate_mesh.py` | Wrong image count | `ValueError` |
| `auto_rig.py` | Blender not installed | `FileNotFoundError` from `_find_blender()` |
| `auto_rig.py` | Blender crash | `RuntimeError` with stderr tail |
| `auto_rig.py` | Blender timeout | `TimeoutError` |
| `auto_rig.py` | Output file missing | `RuntimeError` even on exit code 0 |

## Test Isolation

- Unit tests never import `diffusers`, `hy3dgen`, or `bpy`. All such imports are behind mocks or lazy imports in the source code.
- Each test function gets its own `tmp_path` — no shared mutable state between tests.
- Module-level caches (`_pipeline` globals) are reset in fixtures using `mocker.patch.object` or by directly setting them to `None` in teardown.
- Integration tests that load GPU models use `autouse` session-scoped fixtures to load once and share across the test session.
