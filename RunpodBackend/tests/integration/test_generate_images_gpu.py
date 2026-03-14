"""Integration tests for generate_images.py — requires CUDA GPU + Flux weights.

Run with:
    pytest tests/integration/test_generate_images_gpu.py -m gpu
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def flux_output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("flux_output")


class TestFluxImageGenerationIntegration:
    """End-to-end tests that actually run the Flux pipeline."""

    def test_expand_prompt_returns_4_prompts(self):
        from generate_images import expand_prompt

        prompts = expand_prompt(
            description="A tall elven warrior with silver hair and leather armor",
            style="fantasy",
        )
        assert len(prompts) == 4
        for p in prompts:
            assert isinstance(p, str)
            assert len(p) > 20

    def test_generate_views_creates_4_png_files(self, flux_output_dir):
        from generate_images import expand_prompt, generate_views

        prompts = expand_prompt(
            description="A short dwarf warrior with a battle axe",
            style="fantasy",
        )
        paths = generate_views(prompts, str(flux_output_dir), seed=42)

        assert len(paths) == 4
        for p in paths:
            path = Path(p)
            assert path.exists(), f"Expected PNG file: {p}"
            assert path.suffix == ".png"
            assert path.stat().st_size > 0

    def test_generated_images_are_valid_pngs(self, flux_output_dir):
        from PIL import Image

        from generate_images import expand_prompt, generate_views

        prompts = expand_prompt("A ranger with a bow", "fantasy")
        paths = generate_views(prompts, str(flux_output_dir), seed=99)

        for p in paths:
            with Image.open(p) as img:
                assert img.format == "PNG"
                assert img.width > 0
                assert img.height > 0

    def test_generate_views_uses_correct_resolution(self, flux_output_dir):
        from PIL import Image

        from generate_images import _IMAGE_SIZE, expand_prompt, generate_views

        prompts = expand_prompt("A paladin in golden armor", "fantasy")
        paths = generate_views(prompts, str(flux_output_dir), seed=7)

        for p in paths:
            with Image.open(p) as img:
                assert img.width == _IMAGE_SIZE
                assert img.height == _IMAGE_SIZE

    def test_unload_model_frees_gpu(self, tmp_path):
        import torch

        from generate_images import expand_prompt, generate_views, unload_model

        prompts = expand_prompt("A necromancer in dark robes", "fantasy")
        generate_views(prompts, str(tmp_path / "flux_unload"), seed=1)

        mem_before = torch.cuda.memory_allocated()
        unload_model()
        mem_after = torch.cuda.memory_allocated()

        assert mem_after <= mem_before, "Expected GPU memory to decrease after unload"
