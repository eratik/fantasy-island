"""Integration tests for generate_mesh.py — requires CUDA GPU + Hunyuan3D weights.

Run with:
    pytest tests/integration/test_generate_mesh_gpu.py -m gpu
"""

from __future__ import annotations

from pathlib import Path

import pytest
from glb_validator import validate_glb

pytestmark = [pytest.mark.gpu, pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def reference_images(tmp_path_factory):
    """Generate 4 reference images via Flux for mesh generation tests."""
    from generate_images import expand_prompt, generate_views

    out_dir = tmp_path_factory.mktemp("mesh_ref_images")
    prompts = expand_prompt(
        description="A tall elven warrior with silver hair and leather armor",
        style="fantasy",
    )
    return generate_views(prompts, str(out_dir), seed=42)


class TestHunyuan3DMeshGenerationIntegration:
    def test_generate_mesh_creates_glb(self, reference_images, tmp_path):
        from generate_mesh import generate_mesh

        out = str(tmp_path / "mesh.glb")
        result = generate_mesh(reference_images, out, poly_target=10_000)

        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_mesh_poly_count_matches_target(self, reference_images, tmp_path):
        """Triangle count should be within 2× of poly_target."""
        from generate_mesh import generate_mesh

        out = str(tmp_path / "mesh.glb")
        generate_mesh(reference_images, out, poly_target=10_000)

        val = validate_glb(out)
        assert val["has_mesh"], "No mesh found in GLB"
        if val["triangle_count"] > 0:
            assert val["triangle_count"] <= 10_000 * 2, (
                f"Triangle count {val['triangle_count']} exceeds 2× poly_target"
            )

    def test_low_poly_mesh_has_fewer_triangles_than_high(self, reference_images, tmp_path):
        from generate_mesh import generate_mesh

        out_low = str(tmp_path / "mesh_low.glb")
        out_high = str(tmp_path / "mesh_high.glb")

        generate_mesh(reference_images, out_low, poly_target=5_000)
        generate_mesh(reference_images, out_high, poly_target=50_000)

        val_low = validate_glb(out_low)
        val_high = validate_glb(out_high)

        assert val_low["triangle_count"] < val_high["triangle_count"]

    def test_unload_model_after_mesh_generation(self, reference_images, tmp_path):
        import torch

        from generate_mesh import generate_mesh, unload_model

        out = str(tmp_path / "mesh_unload.glb")
        generate_mesh(reference_images, out, poly_target=5_000)

        mem_before = torch.cuda.memory_allocated()
        unload_model()
        mem_after = torch.cuda.memory_allocated()

        assert mem_after <= mem_before
