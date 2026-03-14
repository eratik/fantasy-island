"""Shared pytest fixtures, mock factories, and GLB validation for RunpodBackend tests."""

from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# GLB Validation
# ---------------------------------------------------------------------------

UNITY_HUMANOID_BONES: list[str] = [
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

assert len(UNITY_HUMANOID_BONES) == 22


def validate_glb(glb_path: str) -> dict:
    """Parse a .glb file and validate it has a Unity-compatible humanoid skeleton.

    Parses the glTF JSON chunk directly from the GLB binary. Does not require
    any optional test dependencies beyond the project's own trimesh dep.

    Returns a dict with:
        valid (bool): True if all checks pass.
        errors (list[str]): Human-readable error descriptions.
        bone_names (list[str]): Bone names found in skeleton joints.
        missing_bones (list[str]): Required Unity bones not found.
        extra_bones (list[str]): Bones found that are not in UNITY_HUMANOID_BONES.
        triangle_count (int): Estimated triangle count from accessor data.
        has_mesh (bool): At least one mesh primitive exists.
        has_skin (bool): At least one skin (joint binding) exists.
    """
    path = Path(glb_path)
    result: dict = {
        "valid": True,
        "errors": [],
        "bone_names": [],
        "missing_bones": [],
        "extra_bones": [],
        "triangle_count": 0,
        "has_mesh": False,
        "has_skin": False,
    }

    if not path.exists():
        raise FileNotFoundError(f"GLB file not found: {glb_path}")

    data = path.read_bytes()

    if len(data) < 12:
        result["valid"] = False
        result["errors"].append("File too small to be a valid GLB")
        return result

    magic, _version, _total_length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        result["valid"] = False
        result["errors"].append(f"Invalid GLB magic: {magic:#010x}")
        return result

    if len(data) < 20:
        result["valid"] = False
        result["errors"].append("GLB too small — no JSON chunk")
        return result

    json_chunk_length, json_chunk_type = struct.unpack_from("<II", data, 12)
    if json_chunk_type != 0x4E4F534A:
        result["valid"] = False
        result["errors"].append("First chunk is not a JSON chunk")
        return result

    json_bytes = data[20 : 20 + json_chunk_length]
    try:
        gltf = json.loads(json_bytes.decode("utf-8").rstrip("\x00 "))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        result["valid"] = False
        result["errors"].append(f"Failed to parse glTF JSON: {exc}")
        return result

    nodes = gltf.get("nodes", [])
    skins = gltf.get("skins", [])
    meshes = gltf.get("meshes", [])
    accessors = gltf.get("accessors", [])

    result["has_mesh"] = len(meshes) > 0
    result["has_skin"] = len(skins) > 0

    if not result["has_skin"]:
        result["valid"] = False
        result["errors"].append("No skins found — mesh is not bound to a skeleton")

    node_names = [n.get("name", "") for n in nodes]
    joint_names: list[str] = []
    for skin in skins:
        for joint_idx in skin.get("joints", []):
            if 0 <= joint_idx < len(node_names):
                name = node_names[joint_idx]
                if name and name not in joint_names:
                    joint_names.append(name)

    result["bone_names"] = sorted(joint_names)
    required = set(UNITY_HUMANOID_BONES)
    found = set(joint_names)
    result["missing_bones"] = sorted(required - found)
    result["extra_bones"] = sorted(found - required)

    if result["missing_bones"]:
        result["valid"] = False
        result["errors"].append(
            f"Missing {len(result['missing_bones'])} Unity bone(s): "
            + ", ".join(result["missing_bones"])
        )

    total = 0
    for mesh in meshes:
        for prim in mesh.get("primitives", []):
            idx = prim.get("indices")
            if idx is not None and 0 <= idx < len(accessors):
                total += accessors[idx].get("count", 0) // 3
    result["triangle_count"] = total

    return result


# ---------------------------------------------------------------------------
# Pytest configuration
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "gpu: requires CUDA GPU (deselect with '-m not gpu')"
    )
    config.addinivalue_line(
        "markers",
        "blender: requires Blender installed (deselect with '-m not blender')",
    )
    config.addinivalue_line(
        "markers",
        "integration: full integration test (deselect with '-m not integration')",
    )
    config.addinivalue_line("markers", "slow: takes >30s to run")


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_job_dir(tmp_path: Path) -> Path:
    """Create a temporary job directory mirroring /tmp/jobs/{id}/."""
    job_dir = tmp_path / "test-job-001"
    job_dir.mkdir()
    return job_dir


@pytest.fixture
def sample_job_input() -> dict:
    """Return a valid JobInput dict."""
    return {
        "description": "A tall elven warrior with silver hair and leather armor",
        "style": "fantasy",
        "poly_target": 30000,
    }


@pytest.fixture
def sample_job(sample_job_input: dict) -> dict:
    """Return a valid Runpod job dict."""
    return {"id": "test-job-001", "input": sample_job_input}


@pytest.fixture
def four_test_images(tmp_job_dir: Path) -> list[str]:
    """Create 4 minimal PNG files in the job dir and return their paths."""
    paths = []
    for name in ["view_front.png", "view_left.png", "view_right.png", "view_back.png"]:
        path = tmp_job_dir / name
        img = Image.new("RGBA", (64, 64), (255, 255, 255, 255))
        img.save(str(path))
        paths.append(str(path))
    return paths


@pytest.fixture
def tiny_glb(tmp_path: Path) -> str:
    """Create a minimal valid GLB file (cube mesh, no skeleton)."""
    import trimesh

    mesh = trimesh.creation.box(extents=(1, 1, 2))
    path = tmp_path / "tiny_mesh.glb"
    mesh.export(str(path), file_type="glb")
    return str(path)


@pytest.fixture
def fixture_dir() -> Path:
    """Return path to the static fixtures directory."""
    return Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Backwards-compatible aliases used by existing unit tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_png_files(four_test_images: list[str]) -> list[str]:
    """Alias for four_test_images — keeps existing unit tests working."""
    return four_test_images


@pytest.fixture
def sample_job_compat(sample_job: dict) -> dict:
    """Alias — kept for internal use."""
    return sample_job


# ---------------------------------------------------------------------------
# GLB builder helper (used to create minimal fixture GLBs in tests)
# ---------------------------------------------------------------------------


def make_minimal_glb(
    path: Path,
    include_skeleton: bool = True,
    bones: list[str] | None = None,
) -> Path:
    """Build a minimal binary .glb with an optional skeleton and write to path."""
    bone_names = bones if bones is not None else UNITY_HUMANOID_BONES

    nodes: list[dict] = []
    joint_indices: list[int] = []

    if include_skeleton:
        for i, name in enumerate(bone_names):
            nodes.append({"name": name})
            joint_indices.append(i)

    gltf: dict = {"asset": {"version": "2.0"}, "nodes": nodes}
    if include_skeleton and joint_indices:
        gltf["skins"] = [{"joints": joint_indices, "name": "Armature"}]

    json_bytes = json.dumps(gltf).encode("utf-8")
    padding = (4 - len(json_bytes) % 4) % 4
    json_bytes += b" " * padding

    json_chunk_length = len(json_bytes)
    total_length = 12 + 8 + json_chunk_length

    header = struct.pack("<III", 0x46546C67, 2, total_length)
    json_chunk_header = struct.pack("<II", json_chunk_length, 0x4E4F534A)

    path.write_bytes(header + json_chunk_header + json_bytes)
    return path


@pytest.fixture
def sample_glb_file(tmp_path: Path) -> Path:
    """Minimal valid .glb with all 22 Unity bones."""
    return make_minimal_glb(tmp_path / "sample.glb")


@pytest.fixture
def sample_glb_no_skeleton(tmp_path: Path) -> Path:
    """Minimal .glb with NO skeleton."""
    return make_minimal_glb(tmp_path / "no_skeleton.glb", include_skeleton=False)


@pytest.fixture
def sample_glb_missing_bones(tmp_path: Path) -> Path:
    """Minimal .glb with only Hips + Spine (missing the other 20 bones)."""
    return make_minimal_glb(
        tmp_path / "missing_bones.glb", bones=["Hips", "Spine"]
    )


# ---------------------------------------------------------------------------
# Mock pipeline fixtures (architecture-spec versions)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_flux_pipeline(mocker):
    """Mock Flux pipeline — patches FluxPipeline.from_pretrained at module level."""
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
    mocker.patch("generate_images.torch.cuda.is_available", return_value=True)
    mocker.patch("generate_images.torch.cuda.empty_cache")

    return mock_pipeline


@pytest.fixture
def mock_blender_success(mocker):
    """Mock subprocess.run to simulate successful Blender — creates output file."""

    def side_effect(*args, **kwargs):
        cmd = args[0]
        separator_idx = cmd.index("--")
        output_path = cmd[separator_idx + 2]  # second arg after "--"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake-glb-content")
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="", stderr=""
        )

    mock = mocker.patch("auto_rig.subprocess.run", side_effect=side_effect)
    mocker.patch("shutil.which", return_value="/usr/bin/blender")
    return mock
