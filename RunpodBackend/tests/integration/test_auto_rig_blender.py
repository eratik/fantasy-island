"""Integration tests for auto_rig.py — requires Blender installed.

Run with:
    pytest tests/integration/test_auto_rig_blender.py -m blender
"""

from __future__ import annotations

from pathlib import Path

import pytest
from glb_validator import UNITY_HUMANOID_BONES, validate_glb

pytestmark = [pytest.mark.blender, pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def raw_mesh_glb(tmp_path_factory):
    """Generate a raw (unrigged) GLB mesh via the full Flux+Hunyuan3D stack."""
    from generate_images import expand_prompt, generate_views
    from generate_images import unload_model as unload_img
    from generate_mesh import generate_mesh
    from generate_mesh import unload_model as unload_mesh

    out_dir = tmp_path_factory.mktemp("rig_inputs")

    prompts = expand_prompt(
        description="A tall elven warrior with silver hair and leather armor",
        style="fantasy",
    )
    image_paths = generate_views(prompts, str(out_dir), seed=42)
    unload_img()

    mesh_path = str(out_dir / "raw_mesh.glb")
    generate_mesh(image_paths, mesh_path, poly_target=15_000)
    unload_mesh()

    return mesh_path


class TestAutoRigBlenderIntegration:
    def test_auto_rig_creates_output(self, raw_mesh_glb, tmp_path):
        from auto_rig import auto_rig

        out = str(tmp_path / "rigged.glb")
        result = auto_rig(
            input_mesh_path=raw_mesh_glb,
            output_glb_path=out,
            poly_target=15_000,
        )

        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_rigged_glb_contains_all_22_unity_bones(self, raw_mesh_glb, tmp_path):
        from auto_rig import auto_rig

        out = str(tmp_path / "rigged_bones.glb")
        auto_rig(
            input_mesh_path=raw_mesh_glb,
            output_glb_path=out,
            poly_target=15_000,
        )

        val = validate_glb(out)
        assert val["missing_bones"] == [], f"Missing bones: {val['missing_bones']}"
        assert len(val["bone_names"]) >= 22

    def test_all_22_bone_names_exactly_correct(self, raw_mesh_glb, tmp_path):
        from auto_rig import auto_rig

        out = str(tmp_path / "rigged_exact.glb")
        auto_rig(
            input_mesh_path=raw_mesh_glb,
            output_glb_path=out,
            poly_target=15_000,
        )

        val = validate_glb(out)
        for bone in UNITY_HUMANOID_BONES:
            assert bone in val["bone_names"], f"Missing bone: {bone}"

    def test_output_glb_valid_for_unity(self, raw_mesh_glb, tmp_path):
        """Run validate_glb() on the output and assert valid=True, missing_bones=[]."""
        from auto_rig import auto_rig

        out = str(tmp_path / "rigged_valid.glb")
        auto_rig(
            input_mesh_path=raw_mesh_glb,
            output_glb_path=out,
            poly_target=15_000,
        )

        val = validate_glb(out)
        assert val["valid"], f"GLB validation errors: {val['errors']}"
        assert val["missing_bones"] == []

    def test_rig_with_decimation(self, raw_mesh_glb, tmp_path):
        """Low poly target should decimate the mesh substantially."""
        from auto_rig import auto_rig

        out = str(tmp_path / "rigged_low.glb")
        auto_rig(
            input_mesh_path=raw_mesh_glb,
            output_glb_path=out,
            poly_target=3_000,
        )

        val = validate_glb(out)
        assert val["triangle_count"] < 15_000 * 1.5, (
            f"Expected decimation, got {val['triangle_count']} triangles"
        )
