"""Unit tests for generate_images.py.

All tests run without GPU — the Flux pipeline is fully mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

_VIEWS = ("front", "left", "right", "back")


# ---------------------------------------------------------------------------
# expand_prompt
# ---------------------------------------------------------------------------

class TestExpandPrompt:
    def test_returns_four_prompts(self):
        from generate_images import expand_prompt

        result = expand_prompt("a knight", "fantasy")
        assert len(result) == 4

    def test_prompts_are_strings(self):
        from generate_images import expand_prompt

        result = expand_prompt("a knight", "fantasy")
        for p in result:
            assert isinstance(p, str)
            assert len(p) > 0

    def test_each_view_angle_present(self):
        from generate_images import expand_prompt

        result = expand_prompt("a knight", "fantasy")
        expected_prefixes = ["front view", "left side view", "right side view", "back view"]
        for prompt, prefix in zip(result, expected_prefixes):
            assert prefix in prompt, f"Expected '{prefix}' in: {prompt!r}"

    def test_description_in_each_prompt(self):
        from generate_images import expand_prompt

        desc = "a tall wizard with a long beard"
        result = expand_prompt(desc, "fantasy")
        for prompt in result:
            assert desc in prompt, f"Description not found in: {prompt!r}"

    def test_known_style_suffix_applied(self):
        from generate_images import expand_prompt

        result = expand_prompt("a warrior", "sci-fi")
        for prompt in result:
            assert "sci-fi" in prompt.lower() or "futuristic" in prompt.lower()

    def test_unknown_style_falls_back_to_default(self):
        from generate_images import expand_prompt

        result = expand_prompt("a warrior", "steampunk")
        # Should not raise; returns 4 prompts
        assert len(result) == 4

    def test_technical_suffix_present(self):
        from generate_images import expand_prompt

        result = expand_prompt("a knight", "fantasy")
        for prompt in result:
            # The technical suffix always includes "white background"
            assert "white background" in prompt

    def test_order_is_front_left_right_back(self):
        from generate_images import expand_prompt

        result = expand_prompt("a mage", "anime")
        assert "front" in result[0]
        assert "left" in result[1]
        assert "right" in result[2]
        assert "back" in result[3]


# ---------------------------------------------------------------------------
# generate_views
# ---------------------------------------------------------------------------

class TestGetPipeline:
    def test_raises_without_cuda(self, mocker):
        import generate_images

        mocker.patch.object(generate_images.torch.cuda, "is_available", return_value=False)
        generate_images._pipeline = None  # ensure cold start

        with pytest.raises(RuntimeError, match="CUDA"):
            generate_images._get_pipeline()


class TestGenerateViews:
    def _make_mock_pipeline(self, mocker, tmp_path):
        from PIL import Image

        mock_img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        mock_result = MagicMock()
        mock_result.images = [mock_img]
        mock_pipeline = MagicMock(return_value=mock_result)
        return mock_pipeline

    def test_raises_on_wrong_prompt_count(self, tmp_path):
        from generate_images import generate_views

        with pytest.raises(ValueError, match="Expected 4 prompts"):
            generate_views(["only one prompt"], str(tmp_path))

    def test_raises_on_too_many_prompts(self, tmp_path):
        from generate_images import generate_views

        with pytest.raises(ValueError, match="Expected 4 prompts"):
            generate_views(["p"] * 5, str(tmp_path))

    def _setup_mock_pipeline(self, mocker):
        """Patch _get_pipeline and torch to avoid GPU usage."""
        from PIL import Image

        import generate_images

        mock_img = Image.new("RGB", (64, 64))
        mock_result = MagicMock()
        mock_result.images = [mock_img]
        mock_pipeline = MagicMock(return_value=mock_result)

        mocker.patch.object(generate_images, "_get_pipeline", return_value=mock_pipeline)
        mocker.patch("generate_images.torch.Generator", return_value=MagicMock())
        mocker.patch("generate_images.torch.cuda.empty_cache")
        return mock_pipeline

    def test_returns_four_paths(self, mocker, tmp_path):
        import generate_images

        self._setup_mock_pipeline(mocker)

        result = generate_images.generate_views(["p1", "p2", "p3", "p4"], str(tmp_path), seed=42)
        assert len(result) == 4

    def test_output_files_exist(self, mocker, tmp_path):
        import generate_images

        self._setup_mock_pipeline(mocker)

        paths = generate_images.generate_views(["p1", "p2", "p3", "p4"], str(tmp_path), seed=42)
        for p in paths:
            assert Path(p).exists(), f"Expected file to exist: {p}"

    def test_output_filenames_contain_view_names(self, mocker, tmp_path):
        import generate_images

        self._setup_mock_pipeline(mocker)

        paths = generate_images.generate_views(["p"] * 4, str(tmp_path), seed=1)
        names = [Path(p).name for p in paths]
        for view in _VIEWS:
            assert any(view in n for n in names), f"No file for view '{view}' in {names}"

    def test_seed_is_used(self, mocker, tmp_path):
        from PIL import Image

        import generate_images

        mock_img = Image.new("RGB", (64, 64))
        mock_result = MagicMock()
        mock_result.images = [mock_img]
        mock_pipeline = MagicMock(return_value=mock_result)
        mocker.patch.object(generate_images, "_get_pipeline", return_value=mock_pipeline)
        mocker.patch("generate_images.torch.cuda.empty_cache")

        mock_gen = MagicMock()
        mock_gen.manual_seed = MagicMock(return_value=mock_gen)
        mocker.patch("generate_images.torch.Generator", return_value=mock_gen)

        generate_images.generate_views(["p"] * 4, str(tmp_path), seed=12345)
        mock_gen.manual_seed.assert_called_with(12345)

    def test_pipeline_called_once_per_view(self, mocker, tmp_path):
        import generate_images

        mock_pipeline = self._setup_mock_pipeline(mocker)

        generate_images.generate_views(["p"] * 4, str(tmp_path), seed=1)
        assert mock_pipeline.call_count == 4

    def test_output_dir_created(self, mocker, tmp_path):
        import generate_images

        self._setup_mock_pipeline(mocker)

        new_dir = tmp_path / "nested" / "output"
        generate_images.generate_views(["p"] * 4, str(new_dir), seed=1)
        assert new_dir.exists()

    def test_flux_error_raises_runtime_error(self, mocker, tmp_path):
        import generate_images

        mock_pipeline = MagicMock(side_effect=RuntimeError("CUDA OOM"))
        mocker.patch.object(generate_images, "_get_pipeline", return_value=mock_pipeline)
        mocker.patch("generate_images.torch.Generator", return_value=MagicMock())
        mocker.patch("generate_images.torch.cuda.empty_cache")

        with pytest.raises(RuntimeError, match="Flux image generation failed"):
            generate_images.generate_views(["p"] * 4, str(tmp_path), seed=1)


# ---------------------------------------------------------------------------
# unload_model
# ---------------------------------------------------------------------------

class TestUnloadModel:
    def test_unload_clears_pipeline(self, mocker):
        import generate_images

        generate_images._pipeline = MagicMock()
        mocker.patch("torch.cuda.empty_cache")

        from generate_images import unload_model
        unload_model()

        assert generate_images._pipeline is None

    def test_unload_is_safe_when_already_none(self, mocker):
        import generate_images

        generate_images._pipeline = None
        mocker.patch("torch.cuda.empty_cache")

        from generate_images import unload_model
        unload_model()  # Should not raise
        assert generate_images._pipeline is None
