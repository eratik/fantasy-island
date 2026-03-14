"""Unit tests for handler.py.

All tests run without GPU — all pipeline functions are mocked.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# _parse_input
# ---------------------------------------------------------------------------

class TestParseInput:
    def _call(self, raw: dict):
        from handler import _parse_input
        return _parse_input(raw)

    def test_valid_full_input(self):
        result = self._call({
            "description": "A fierce dragon rider",
            "style": "fantasy",
            "poly_target": 25_000,
        })
        assert result["description"] == "A fierce dragon rider"
        assert result["style"] == "fantasy"
        assert result["poly_target"] == 25_000

    def test_missing_description_raises(self):
        with pytest.raises(ValueError, match="'description' is required"):
            self._call({"style": "fantasy"})

    def test_empty_description_raises(self):
        with pytest.raises(ValueError, match="'description' is required"):
            self._call({"description": "   "})

    def test_description_is_stripped(self):
        result = self._call({"description": "  A warrior  "})
        assert result["description"] == "A warrior"

    def test_default_style_is_fantasy(self):
        result = self._call({"description": "A mage"})
        assert result["style"] == "fantasy"

    def test_default_poly_target_is_30000(self):
        result = self._call({"description": "A mage"})
        assert result["poly_target"] == 30_000

    def test_poly_target_too_low_raises(self):
        with pytest.raises(ValueError, match="poly_target"):
            self._call({"description": "X", "poly_target": 999})

    def test_poly_target_too_high_raises(self):
        with pytest.raises(ValueError, match="poly_target"):
            self._call({"description": "X", "poly_target": 200_001})

    def test_poly_target_boundary_low_ok(self):
        result = self._call({"description": "X", "poly_target": 1_000})
        assert result["poly_target"] == 1_000

    def test_poly_target_boundary_high_ok(self):
        result = self._call({"description": "X", "poly_target": 200_000})
        assert result["poly_target"] == 200_000

    def test_poly_target_as_string_coerced(self):
        result = self._call({"description": "X", "poly_target": "50000"})
        assert result["poly_target"] == 50_000


# ---------------------------------------------------------------------------
# _encode_glb
# ---------------------------------------------------------------------------

class TestEncodeGlb:
    def test_returns_base64_string(self, tmp_path):
        from handler import _encode_glb

        f = tmp_path / "test.glb"
        f.write_bytes(b"\x00\x01\x02\x03")
        result = _encode_glb(f)
        assert isinstance(result, str)
        assert base64.b64decode(result) == b"\x00\x01\x02\x03"


# ---------------------------------------------------------------------------
# _generate_thumbnail
# ---------------------------------------------------------------------------

class TestGenerateThumbnail:
    def test_returns_base64_string(self, tmp_path, sample_png_files):
        from handler import _generate_thumbnail

        out = tmp_path / "thumb.png"
        result = _generate_thumbnail(sample_png_files, out)
        assert isinstance(result, str)
        raw = base64.b64decode(result)
        assert raw[:4] == b"\x89PNG"

    def test_thumbnail_file_created(self, tmp_path, sample_png_files):
        from handler import _generate_thumbnail

        out = tmp_path / "thumb.png"
        _generate_thumbnail(sample_png_files, out)
        assert out.exists()

    def test_thumbnail_max_size_512(self, tmp_path, sample_png_files):
        from handler import _generate_thumbnail

        out = tmp_path / "thumb.png"
        _generate_thumbnail(sample_png_files, out)
        with Image.open(out) as img:
            assert img.width <= 512
            assert img.height <= 512

    def test_uses_front_view(self, tmp_path, sample_png_files):
        """Thumbnail should be derived from image_paths[0] (front view)."""
        from handler import _generate_thumbnail

        # Create a distinct red front image
        red_img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        front_path = tmp_path / "front.png"
        red_img.save(str(front_path))

        paths = [str(front_path)] + sample_png_files[1:]
        out = tmp_path / "thumb.png"
        _generate_thumbnail(paths, out)

        with Image.open(out) as img:
            pixel = img.getpixel((0, 0))
            # Should be reddish (from the red front image)
            assert pixel[0] > pixel[1] and pixel[0] > pixel[2]


# ---------------------------------------------------------------------------
# _make_job_dir / _cleanup_job_dir
# ---------------------------------------------------------------------------

class TestJobDirHelpers:
    def test_make_job_dir_creates_directory(self, tmp_path, monkeypatch):
        import handler
        monkeypatch.setattr(handler, "JOBS_TEMP_DIR", tmp_path / "jobs")

        from handler import _make_job_dir
        job_dir = _make_job_dir("abc-123")
        assert job_dir.exists()
        assert job_dir.name == "abc-123"

    def test_cleanup_removes_directory(self, tmp_path):
        from handler import _cleanup_job_dir

        d = tmp_path / "job"
        d.mkdir()
        (d / "file.txt").write_text("hello")

        _cleanup_job_dir(d)
        assert not d.exists()

    def test_cleanup_does_not_raise_if_missing(self, tmp_path):
        from handler import _cleanup_job_dir

        missing = tmp_path / "nonexistent"
        _cleanup_job_dir(missing)  # Should not raise


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------

class TestRunPipeline:
    def _mock_all_stages(self, mocker, tmp_path, sample_png_files):
        """Patch all pipeline functions and simulate successful execution."""
        import handler

        # Mock expand_prompt → 4 prompts
        mocker.patch.object(handler, "expand_prompt", return_value=["p1", "p2", "p3", "p4"])

        # Mock generate_views → sample PNG paths
        mocker.patch.object(handler, "generate_views", return_value=sample_png_files)

        # Mock unload functions
        mocker.patch.object(handler, "unload_image_model")
        mocker.patch.object(handler, "unload_mesh_model")

        # Mock generate_mesh → writes a dummy GLB
        def fake_generate_mesh(image_paths, output_path, poly_target):
            Path(output_path).write_bytes(b"GLB_RAW")
            return output_path

        mocker.patch.object(handler, "generate_mesh", side_effect=fake_generate_mesh)

        # Mock auto_rig → writes a dummy rigged GLB
        def fake_auto_rig(input_mesh_path, output_glb_path, poly_target):
            Path(output_glb_path).write_bytes(b"GLB_RIGGED")
            return output_glb_path

        mocker.patch.object(handler, "auto_rig", side_effect=fake_auto_rig)

        # Mock runpod progress updates
        mocker.patch("runpod.serverless.progress_update")

        # Patch JOBS_TEMP_DIR so job dirs go to tmp_path
        mocker.patch.object(handler, "JOBS_TEMP_DIR", tmp_path)

    def test_returns_expected_keys(self, mocker, tmp_path, sample_png_files, sample_job):
        self._mock_all_stages(mocker, tmp_path, sample_png_files)

        from handler import _parse_input, run_pipeline
        job_input = _parse_input(sample_job["input"])
        result = run_pipeline(sample_job, job_input)

        assert "glb" in result
        assert "thumbnail" in result
        assert "generation_time_seconds" in result

    def test_glb_is_base64(self, mocker, tmp_path, sample_png_files, sample_job):
        self._mock_all_stages(mocker, tmp_path, sample_png_files)

        from handler import _parse_input, run_pipeline
        job_input = _parse_input(sample_job["input"])
        result = run_pipeline(sample_job, job_input)

        raw = base64.b64decode(result["glb"])
        assert raw == b"GLB_RIGGED"

    def test_generation_time_is_positive_float(
        self, mocker, tmp_path, sample_png_files, sample_job
    ):
        self._mock_all_stages(mocker, tmp_path, sample_png_files)

        from handler import _parse_input, run_pipeline
        job_input = _parse_input(sample_job["input"])
        result = run_pipeline(sample_job, job_input)

        assert isinstance(result["generation_time_seconds"], float)
        assert result["generation_time_seconds"] >= 0

    def test_progress_updates_called(self, mocker, tmp_path, sample_png_files, sample_job):
        self._mock_all_stages(mocker, tmp_path, sample_png_files)

        mock_progress = mocker.patch("runpod.serverless.progress_update")

        from handler import _parse_input, run_pipeline
        job_input = _parse_input(sample_job["input"])
        run_pipeline(sample_job, job_input)

        assert mock_progress.call_count >= 3

    def test_stage_order_is_correct(self, mocker, tmp_path, sample_png_files, sample_job):
        """expand_prompt → generate_views → generate_mesh → auto_rig."""
        import handler

        calls = []

        mocker.patch.object(
            handler, "expand_prompt",
            side_effect=lambda **kw: calls.append("expand") or ["p"] * 4,
        )
        mocker.patch.object(
            handler, "generate_views",
            side_effect=lambda **kw: calls.append("views") or sample_png_files,
        )
        mocker.patch.object(
            handler, "unload_image_model",
            side_effect=lambda: calls.append("unload_img"),
        )
        mocker.patch.object(
            handler, "unload_mesh_model",
            side_effect=lambda: calls.append("unload_mesh"),
        )

        def fake_mesh(image_paths, output_path, poly_target):
            calls.append("mesh")
            Path(output_path).write_bytes(b"GLB")
            return output_path

        def fake_rig(input_mesh_path, output_glb_path, poly_target):
            calls.append("rig")
            Path(output_glb_path).write_bytes(b"GLB")
            return output_glb_path

        mocker.patch.object(handler, "generate_mesh", side_effect=fake_mesh)
        mocker.patch.object(handler, "auto_rig", side_effect=fake_rig)
        mocker.patch("runpod.serverless.progress_update")
        mocker.patch.object(handler, "JOBS_TEMP_DIR", tmp_path)

        from handler import _parse_input, run_pipeline
        job_input = _parse_input(sample_job["input"])
        run_pipeline(sample_job, job_input)

        assert calls.index("expand") < calls.index("views")
        assert calls.index("views") < calls.index("mesh")
        assert calls.index("mesh") < calls.index("rig")


# ---------------------------------------------------------------------------
# run_pipeline — error propagation
# ---------------------------------------------------------------------------

class TestRunPipelineErrors:
    def _patch_base(self, mocker, tmp_path, sample_png_files):
        import handler
        mocker.patch.object(handler, "JOBS_TEMP_DIR", tmp_path)
        mocker.patch("runpod.serverless.progress_update")
        return handler

    def test_expand_prompt_failure_raises_prompt_expansion_error(
        self, mocker, tmp_path, sample_png_files, sample_job
    ):
        handler = self._patch_base(mocker, tmp_path, sample_png_files)
        mocker.patch.object(handler, "expand_prompt", side_effect=RuntimeError("LLM failed"))

        from handler import PromptExpansionError, _parse_input, run_pipeline
        job_input = _parse_input(sample_job["input"])

        with pytest.raises(PromptExpansionError):
            run_pipeline(sample_job, job_input)

    def test_generate_views_failure_raises_image_generation_error(
        self, mocker, tmp_path, sample_png_files, sample_job
    ):
        handler = self._patch_base(mocker, tmp_path, sample_png_files)
        mocker.patch.object(handler, "expand_prompt", return_value=["p"] * 4)
        mocker.patch.object(handler, "generate_views", side_effect=RuntimeError("CUDA OOM"))

        from handler import ImageGenerationError, _parse_input, run_pipeline
        job_input = _parse_input(sample_job["input"])

        with pytest.raises(ImageGenerationError):
            run_pipeline(sample_job, job_input)

    def test_generate_mesh_failure_raises_mesh_generation_error(
        self, mocker, tmp_path, sample_png_files, sample_job
    ):
        handler = self._patch_base(mocker, tmp_path, sample_png_files)
        mocker.patch.object(handler, "expand_prompt", return_value=["p"] * 4)
        mocker.patch.object(handler, "generate_views", return_value=sample_png_files)
        mocker.patch.object(handler, "unload_image_model")
        mocker.patch.object(handler, "generate_mesh", side_effect=RuntimeError("mesh fail"))

        from handler import MeshGenerationError, _parse_input, run_pipeline
        job_input = _parse_input(sample_job["input"])

        with pytest.raises(MeshGenerationError):
            run_pipeline(sample_job, job_input)

    def test_auto_rig_failure_raises_rigging_error(
        self, mocker, tmp_path, sample_png_files, sample_job
    ):
        handler = self._patch_base(mocker, tmp_path, sample_png_files)
        mocker.patch.object(handler, "expand_prompt", return_value=["p"] * 4)
        mocker.patch.object(handler, "generate_views", return_value=sample_png_files)
        mocker.patch.object(handler, "unload_image_model")
        mocker.patch.object(handler, "unload_mesh_model")

        def fake_mesh(image_paths, output_path, poly_target):
            Path(output_path).write_bytes(b"GLB")
            return output_path

        mocker.patch.object(handler, "generate_mesh", side_effect=fake_mesh)
        mocker.patch.object(handler, "auto_rig", side_effect=RuntimeError("Blender crash"))

        from handler import RiggingError, _parse_input, run_pipeline
        job_input = _parse_input(sample_job["input"])

        with pytest.raises(RiggingError):
            run_pipeline(sample_job, job_input)

    def test_handler_cleans_up_job_dir(self, mocker, tmp_path, sample_png_files, sample_job):
        """Job temp dir is removed even when a pipeline stage fails."""
        handler = self._patch_base(mocker, tmp_path, sample_png_files)
        mocker.patch.object(handler, "expand_prompt", side_effect=RuntimeError("fail"))

        from handler import PromptExpansionError, _parse_input, run_pipeline
        job_input = _parse_input(sample_job["input"])

        with pytest.raises(PromptExpansionError):
            run_pipeline(sample_job, job_input)

        # All job subdirs under JOBS_TEMP_DIR should have been removed
        remaining = list(tmp_path.iterdir())
        assert remaining == [], f"Job dirs not cleaned up after failure: {remaining}"


# ---------------------------------------------------------------------------
# handler (entry point)
# ---------------------------------------------------------------------------

class TestHandler:
    def test_invalid_input_raises_value_error(self):
        from handler import handler

        job = {"id": "x", "input": {}}  # missing description
        with pytest.raises(ValueError, match="Invalid job input"):
            handler(job)

    def test_valid_job_calls_run_pipeline(self, mocker, tmp_path, sample_png_files):
        import handler as handler_module

        mocker.patch.object(handler_module, "JOBS_TEMP_DIR", tmp_path)
        mocker.patch("runpod.serverless.progress_update")
        mocker.patch.object(handler_module, "expand_prompt", return_value=["p"] * 4)
        mocker.patch.object(handler_module, "generate_views", return_value=sample_png_files)
        mocker.patch.object(handler_module, "unload_image_model")
        mocker.patch.object(handler_module, "unload_mesh_model")

        def fake_mesh(image_paths, output_path, poly_target):
            Path(output_path).write_bytes(b"GLB")
            return output_path

        def fake_rig(input_mesh_path, output_glb_path, poly_target):
            Path(output_glb_path).write_bytes(b"GLB")
            return output_glb_path

        mocker.patch.object(handler_module, "generate_mesh", side_effect=fake_mesh)
        mocker.patch.object(handler_module, "auto_rig", side_effect=fake_rig)

        from handler import handler
        result = handler({
            "id": "test-123",
            "input": {"description": "A brave knight"},
        })

        assert "glb" in result
        assert "thumbnail" in result
        assert "generation_time_seconds" in result
