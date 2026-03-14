"""Integration tests for the full end-to-end character generation pipeline.

Runs: text description → Flux images → Hunyuan3D mesh → Blender rigging → .glb

Requires: GPU (CUDA), Blender on PATH or BLENDER_PATH, all model weights.

Run with:
    pytest tests/integration/test_pipeline_e2e.py -m integration
"""

from __future__ import annotations

import base64

import pytest
from glb_validator import validate_glb

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.blender, pytest.mark.slow]


class TestFullPipelineIntegration:
    """End-to-end pipeline: text → rigged .glb via handler()."""

    def _run_handler(
        self, description: str, style: str = "fantasy", poly_target: int = 15_000
    ) -> dict:
        from handler import handler

        job = {
            "id": "integration-test-001",
            "input": {
                "description": description,
                "style": style,
                "poly_target": poly_target,
            },
        }
        return handler(job)

    def test_full_pipeline(self, tmp_path):
        """Text description → handler → base64 GLB. Validate all 22 bones present."""
        result = self._run_handler(
            "A tall elven warrior with silver hair and leather armor"
        )

        assert "glb" in result
        assert "thumbnail" in result
        assert "generation_time_seconds" in result

        raw = base64.b64decode(result["glb"])
        glb_path = tmp_path / "output.glb"
        glb_path.write_bytes(raw)

        val = validate_glb(str(glb_path))
        assert val["valid"], f"GLB validation errors: {val['errors']}"
        assert val["missing_bones"] == [], f"Missing bones: {val['missing_bones']}"
        assert val["has_mesh"], "Output has no mesh"
        assert val["has_skin"], "Output mesh is not skinned"

    def test_glb_output_is_base64(self):
        result = self._run_handler("A fierce orc warrior with a giant club")
        raw = base64.b64decode(result["glb"])
        assert len(raw) > 0

    def test_thumbnail_is_valid_base64_png(self):
        result = self._run_handler("A sneaky rogue with daggers")
        raw = base64.b64decode(result["thumbnail"])
        assert raw[:4] == b"\x89PNG", "Thumbnail is not a PNG"

    def test_generation_time_is_reasonable(self):
        result = self._run_handler("A simple character")
        elapsed = result["generation_time_seconds"]
        assert isinstance(elapsed, float)
        assert 0 < elapsed < 600, f"Pipeline time {elapsed}s is out of expected range"

    def test_fantasy_style_pipeline(self):
        result = self._run_handler("A dark elf assassin", style="fantasy")
        assert "glb" in result

    def test_sci_fi_style_pipeline(self):
        result = self._run_handler("A space marine in power armor", style="sci-fi")
        assert "glb" in result

    def test_job_temp_dir_cleaned_up(self, monkeypatch, tmp_path):
        """Verify the job temp directory is removed after the pipeline completes."""
        import handler as handler_module

        monkeypatch.setattr(handler_module, "JOBS_TEMP_DIR", tmp_path / "jobs")

        self._run_handler("A knight")
        jobs_dir = tmp_path / "jobs"

        if jobs_dir.exists():
            remaining = list(jobs_dir.iterdir())
            assert remaining == [], f"Job dirs not cleaned up: {remaining}"
