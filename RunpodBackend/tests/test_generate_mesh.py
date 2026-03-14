"""Unit tests for generate_mesh.py.

All tests run without GPU — the Hunyuan3D pipeline is fully mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# _poly_target_to_mc_resolution
# ---------------------------------------------------------------------------

class TestPolyTargetToMcResolution:
    def _call(self, poly_target: int) -> int:
        from generate_mesh import _poly_target_to_mc_resolution
        return _poly_target_to_mc_resolution(poly_target)

    def test_low_poly_gives_low_resolution(self):
        assert self._call(5_000) == 192

    def test_boundary_10000_gives_192(self):
        assert self._call(10_000) == 192

    def test_medium_poly_gives_256(self):
        assert self._call(20_000) == 256

    def test_boundary_30000_gives_256(self):
        assert self._call(30_000) == 256

    def test_high_poly_gives_320(self):
        assert self._call(40_000) == 320

    def test_boundary_50000_gives_320(self):
        assert self._call(50_000) == 320

    def test_very_high_poly_gives_384(self):
        assert self._call(100_000) == 384

    def test_returns_int(self):
        result = self._call(30_000)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# generate_mesh input validation
# ---------------------------------------------------------------------------

class TestGenerateMeshValidation:
    def test_raises_on_wrong_image_count(self, tmp_path):
        from generate_mesh import generate_mesh

        with pytest.raises(ValueError, match="Expected 4 image paths"):
            generate_mesh(["a.png", "b.png"], str(tmp_path / "out.glb"))

    def test_raises_on_too_many_images(self, tmp_path):
        from generate_mesh import generate_mesh

        with pytest.raises(ValueError, match="Expected 4 image paths"):
            generate_mesh(["a"] * 5, str(tmp_path / "out.glb"))

    def test_raises_on_missing_image_file(self, tmp_path):
        from generate_mesh import generate_mesh

        paths = [str(tmp_path / f"view_{v}.png") for v in ("front", "left", "right", "back")]
        # Don't create the files — should raise FileNotFoundError

        with pytest.raises(FileNotFoundError, match="Reference image not found"):
            generate_mesh(paths, str(tmp_path / "out.glb"))

    def test_raises_if_only_some_images_missing(self, tmp_path, sample_png_files):
        from generate_mesh import generate_mesh

        # Replace last path with a non-existent file
        paths = sample_png_files[:3] + [str(tmp_path / "missing.png")]

        with pytest.raises(FileNotFoundError):
            generate_mesh(paths, str(tmp_path / "out.glb"))


# ---------------------------------------------------------------------------
# generate_mesh pipeline invocation
# ---------------------------------------------------------------------------

class TestGenerateMeshPipeline:
    def _mock_pipeline(self, mocker, tmp_path, output_path: Path):
        """Patch _load_hunyuan to return a mock that writes a dummy .glb."""
        import generate_mesh

        def fake_pipeline(**kwargs):
            # Write a tiny file so generate_mesh sees the output exist
            output_path.write_bytes(b"GLB_DUMMY")
            mock_mesh = MagicMock()
            mock_mesh.export = MagicMock(side_effect=lambda p: output_path.write_bytes(b"GLB"))
            mock_result = MagicMock()
            mock_result.meshes = [mock_mesh]
            return mock_result

        mock_pipe = MagicMock(side_effect=fake_pipeline)
        mocker.patch.object(generate_mesh, "_load_hunyuan", return_value=mock_pipe)
        return mock_pipe

    def test_returns_output_path(self, mocker, tmp_path, sample_png_files):
        import generate_mesh

        out = tmp_path / "mesh.glb"
        mock_mesh = MagicMock()
        mock_mesh.export = MagicMock(side_effect=lambda p: out.write_bytes(b"GLB"))
        mock_result = MagicMock()
        mock_result.meshes = [mock_mesh]
        mock_pipe = MagicMock(return_value=mock_result)
        mocker.patch.object(generate_mesh, "_load_hunyuan", return_value=mock_pipe)

        returned = generate_mesh.generate_mesh(sample_png_files, str(out))
        assert returned == str(out)

    def test_output_dir_created(self, mocker, tmp_path, sample_png_files):
        import generate_mesh

        out = tmp_path / "nested" / "deep" / "mesh.glb"
        mock_mesh = MagicMock()
        mock_mesh.export = MagicMock(side_effect=lambda p: out.write_bytes(b"GLB"))
        mock_result = MagicMock()
        mock_result.meshes = [mock_mesh]
        mock_pipe = MagicMock(return_value=mock_result)
        mocker.patch.object(generate_mesh, "_load_hunyuan", return_value=mock_pipe)

        generate_mesh.generate_mesh(sample_png_files, str(out))
        assert out.parent.exists()

    def test_mc_resolution_passed_to_pipeline(self, mocker, tmp_path, sample_png_files):
        import generate_mesh

        out = tmp_path / "mesh.glb"
        mock_mesh = MagicMock()
        mock_mesh.export = MagicMock(side_effect=lambda p: out.write_bytes(b"GLB"))
        mock_result = MagicMock()
        mock_result.meshes = [mock_mesh]
        mock_pipe = MagicMock(return_value=mock_result)
        mocker.patch.object(generate_mesh, "_load_hunyuan", return_value=mock_pipe)

        generate_mesh.generate_mesh(sample_png_files, str(out), poly_target=5_000)

        call_kwargs = mock_pipe.call_args.kwargs
        assert call_kwargs.get("mc_resolution") == 192  # low poly → 192

    def test_pipeline_error_raises_runtime_error(self, mocker, tmp_path, sample_png_files):
        import generate_mesh

        out = tmp_path / "mesh.glb"
        mock_pipe = MagicMock(side_effect=Exception("CUDA OOM"))
        mocker.patch.object(generate_mesh, "_load_hunyuan", return_value=mock_pipe)

        with pytest.raises(RuntimeError, match="Hunyuan3D mesh generation failed"):
            generate_mesh.generate_mesh(sample_png_files, str(out))

    def test_images_loaded_as_rgba(self, mocker, tmp_path, sample_png_files):
        """Verify each image is opened and converted to RGBA."""
        from PIL import Image as PILImage

        import generate_mesh

        out = tmp_path / "mesh.glb"
        captured_images = []

        def fake_pipeline(image, extra_images, **kwargs):
            captured_images.extend([image] + extra_images)
            mock_mesh = MagicMock()
            mock_mesh.export = MagicMock(side_effect=lambda p: out.write_bytes(b"GLB"))
            mock_result = MagicMock()
            mock_result.meshes = [mock_mesh]
            return mock_result

        mock_pipe = MagicMock(side_effect=fake_pipeline)
        mocker.patch.object(generate_mesh, "_load_hunyuan", return_value=mock_pipe)

        generate_mesh.generate_mesh(sample_png_files, str(out))

        for img in captured_images:
            assert isinstance(img, PILImage.Image)
            assert img.mode == "RGBA"


# ---------------------------------------------------------------------------
# unload_model
# ---------------------------------------------------------------------------

class TestUnloadMeshModel:
    def test_unload_clears_pipeline(self, mocker):
        import generate_mesh

        generate_mesh._pipeline = MagicMock()
        mocker.patch("torch.cuda.empty_cache")

        from generate_mesh import unload_model
        unload_model()

        assert generate_mesh._pipeline is None

    def test_unload_safe_when_none(self, mocker):
        import generate_mesh

        generate_mesh._pipeline = None
        mocker.patch("torch.cuda.empty_cache")

        from generate_mesh import unload_model
        unload_model()  # Should not raise
