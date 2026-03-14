"""Unit tests for auto_rig.py.

All tests run without Blender — subprocess.run is mocked.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# _find_blender
# ---------------------------------------------------------------------------


class TestFindBlender:
    def test_uses_blender_path_env(self, monkeypatch, tmp_path):
        blender_exe = tmp_path / "blender"
        blender_exe.touch()
        monkeypatch.setenv("BLENDER_PATH", str(blender_exe))

        from auto_rig import _find_blender

        result = _find_blender()
        assert result == str(blender_exe)

    def test_falls_back_to_which(self, mocker, monkeypatch):
        monkeypatch.delenv("BLENDER_PATH", raising=False)
        mocker.patch("shutil.which", return_value="/usr/bin/blender")

        from auto_rig import _find_blender

        result = _find_blender()
        assert result == "/usr/bin/blender"

    def test_raises_if_not_found(self, mocker, monkeypatch):
        monkeypatch.delenv("BLENDER_PATH", raising=False)
        mocker.patch("shutil.which", return_value=None)

        from auto_rig import _find_blender

        with pytest.raises(FileNotFoundError, match="Blender executable not found"):
            _find_blender()

    def test_auto_rig_blender_env_var(self, mocker, monkeypatch, tmp_path):
        """auto_rig respects BLENDER_PATH environment variable."""
        blender_exe = tmp_path / "myblender"
        blender_exe.touch()
        monkeypatch.setenv("BLENDER_PATH", str(blender_exe))

        input_mesh = tmp_path / "mesh.glb"
        input_mesh.write_bytes(b"GLB")
        out_path = tmp_path / "out.glb"
        out_path.write_bytes(b"GLB")

        mock_run = mocker.patch(
            "auto_rig.subprocess.run",
            return_value=MagicMock(returncode=0, stdout="", stderr=""),
        )

        from auto_rig import auto_rig

        auto_rig(input_mesh_path=str(input_mesh), output_glb_path=str(out_path))
        cmd = mock_run.call_args.args[0]
        assert cmd[0] == str(blender_exe)

    def test_auto_rig_blender_not_found(self, mocker, monkeypatch, tmp_path):
        """Raises FileNotFoundError when Blender is not on PATH."""
        monkeypatch.delenv("BLENDER_PATH", raising=False)
        mocker.patch("shutil.which", return_value=None)

        input_mesh = tmp_path / "mesh.glb"
        input_mesh.write_bytes(b"GLB")

        from auto_rig import auto_rig

        with pytest.raises(FileNotFoundError):
            auto_rig(input_mesh_path=str(input_mesh), output_glb_path=str(tmp_path / "out.glb"))


# ---------------------------------------------------------------------------
# auto_rig — input validation
# ---------------------------------------------------------------------------


class TestAutoRigValidation:
    def test_raises_if_input_not_found(self, tmp_path):
        from auto_rig import auto_rig

        missing = str(tmp_path / "nonexistent.glb")
        out = str(tmp_path / "out.glb")

        with pytest.raises(FileNotFoundError, match="Input mesh not found"):
            auto_rig(input_mesh_path=missing, output_glb_path=out)

    def test_auto_rig_creates_output_dir(self, mock_blender_success, tmp_path):
        """auto_rig creates the output directory if it does not exist."""
        input_mesh = tmp_path / "mesh.glb"
        input_mesh.write_bytes(b"GLB")
        out_path = tmp_path / "nested" / "output.glb"

        from auto_rig import auto_rig

        auto_rig(input_mesh_path=str(input_mesh), output_glb_path=str(out_path))
        assert out_path.parent.exists()


# ---------------------------------------------------------------------------
# auto_rig — subprocess invocation
# ---------------------------------------------------------------------------


class TestAutoRigSubprocess:
    def test_auto_rig_success(self, mock_blender_success, tmp_path):
        """Returns output_glb_path when Blender exits 0 and output file exists."""
        input_mesh = tmp_path / "mesh.glb"
        input_mesh.write_bytes(b"GLB")
        out_path = tmp_path / "out.glb"

        from auto_rig import auto_rig

        result = auto_rig(input_mesh_path=str(input_mesh), output_glb_path=str(out_path))
        assert result == str(out_path)
        assert out_path.exists()

    def test_blender_called_with_background_flag(self, mocker, tmp_path):
        input_mesh = tmp_path / "mesh.glb"
        input_mesh.write_bytes(b"GLB")
        out_path = tmp_path / "out.glb"
        out_path.write_bytes(b"GLB")

        mock_run = mocker.patch(
            "auto_rig.subprocess.run",
            return_value=MagicMock(returncode=0, stdout="", stderr=""),
        )
        mocker.patch("auto_rig._find_blender", return_value="/usr/bin/blender")

        from auto_rig import auto_rig

        auto_rig(input_mesh_path=str(input_mesh), output_glb_path=str(out_path))
        cmd = mock_run.call_args.args[0]
        assert "--background" in cmd

    def test_blender_called_with_python_expr(self, mocker, tmp_path):
        input_mesh = tmp_path / "mesh.glb"
        input_mesh.write_bytes(b"GLB")
        out_path = tmp_path / "out.glb"
        out_path.write_bytes(b"GLB")

        mock_run = mocker.patch(
            "auto_rig.subprocess.run",
            return_value=MagicMock(returncode=0, stdout="", stderr=""),
        )
        mocker.patch("auto_rig._find_blender", return_value="/usr/bin/blender")

        from auto_rig import auto_rig

        auto_rig(input_mesh_path=str(input_mesh), output_glb_path=str(out_path))
        cmd = mock_run.call_args.args[0]
        assert "--python-expr" in cmd

    def test_input_output_poly_passed_to_blender(self, mocker, tmp_path):
        input_mesh = tmp_path / "mesh.glb"
        input_mesh.write_bytes(b"GLB")
        out_path = tmp_path / "out.glb"
        out_path.write_bytes(b"GLB")

        mock_run = mocker.patch(
            "auto_rig.subprocess.run",
            return_value=MagicMock(returncode=0, stdout="", stderr=""),
        )
        mocker.patch("auto_rig._find_blender", return_value="/usr/bin/blender")

        from auto_rig import auto_rig

        auto_rig(
            input_mesh_path=str(input_mesh),
            output_glb_path=str(out_path),
            poly_target=15_000,
        )
        cmd = mock_run.call_args.args[0]
        assert str(input_mesh) in cmd
        assert str(out_path) in cmd
        assert "15000" in cmd

    def test_timeout_is_set(self, mocker, tmp_path):
        input_mesh = tmp_path / "mesh.glb"
        input_mesh.write_bytes(b"GLB")
        out_path = tmp_path / "out.glb"
        out_path.write_bytes(b"GLB")

        mock_run = mocker.patch(
            "auto_rig.subprocess.run",
            return_value=MagicMock(returncode=0, stdout="", stderr=""),
        )
        mocker.patch("auto_rig._find_blender", return_value="/usr/bin/blender")

        from auto_rig import auto_rig

        auto_rig(input_mesh_path=str(input_mesh), output_glb_path=str(out_path))
        kwargs = mock_run.call_args.kwargs
        assert "timeout" in kwargs
        assert kwargs["timeout"] == 300

    def test_returns_output_path_on_success(self, mocker, tmp_path):
        input_mesh = tmp_path / "mesh.glb"
        input_mesh.write_bytes(b"GLB")
        out_path = tmp_path / "out.glb"
        out_path.write_bytes(b"GLB")

        mocker.patch(
            "auto_rig.subprocess.run",
            return_value=MagicMock(returncode=0, stdout="", stderr=""),
        )
        mocker.patch("auto_rig._find_blender", return_value="/usr/bin/blender")

        from auto_rig import auto_rig

        result = auto_rig(input_mesh_path=str(input_mesh), output_glb_path=str(out_path))
        assert result == str(out_path)


# ---------------------------------------------------------------------------
# auto_rig — error cases
# ---------------------------------------------------------------------------


class TestAutoRigErrors:
    def test_auto_rig_blender_failure(self, mocker, tmp_path):
        """Raises RuntimeError when Blender exits non-zero."""
        input_mesh = tmp_path / "mesh.glb"
        input_mesh.write_bytes(b"GLB")
        out_path = tmp_path / "out.glb"

        mocker.patch(
            "auto_rig.subprocess.run",
            return_value=MagicMock(returncode=1, stdout="", stderr="ERROR: import failed"),
        )
        mocker.patch("auto_rig._find_blender", return_value="/usr/bin/blender")

        from auto_rig import auto_rig

        with pytest.raises(RuntimeError, match="Blender auto-rig failed"):
            auto_rig(input_mesh_path=str(input_mesh), output_glb_path=str(out_path))

    def test_auto_rig_timeout(self, mocker, tmp_path):
        """Raises TimeoutError when Blender exceeds 300s."""
        input_mesh = tmp_path / "mesh.glb"
        input_mesh.write_bytes(b"GLB")
        out_path = tmp_path / "out.glb"

        mocker.patch(
            "auto_rig.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="blender", timeout=300),
        )
        mocker.patch("auto_rig._find_blender", return_value="/usr/bin/blender")

        from auto_rig import auto_rig

        with pytest.raises(TimeoutError, match="timed out"):
            auto_rig(input_mesh_path=str(input_mesh), output_glb_path=str(out_path))

    def test_auto_rig_output_missing_after_success(self, mocker, tmp_path):
        """Raises RuntimeError when Blender exits 0 but output file missing."""
        input_mesh = tmp_path / "mesh.glb"
        input_mesh.write_bytes(b"GLB")
        out_path = tmp_path / "out.glb"

        mocker.patch(
            "auto_rig.subprocess.run",
            return_value=MagicMock(returncode=0, stdout="", stderr=""),
        )
        mocker.patch("auto_rig._find_blender", return_value="/usr/bin/blender")

        from auto_rig import auto_rig

        with pytest.raises(RuntimeError, match="output file not found"):
            auto_rig(input_mesh_path=str(input_mesh), output_glb_path=str(out_path))


# ---------------------------------------------------------------------------
# Bone name correctness (derived from the embedded Blender script)
# ---------------------------------------------------------------------------

EXPECTED_BONES = [
    "Hips", "Spine", "Chest", "UpperChest", "Neck", "Head",
    "LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand",
    "RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
    "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes",
]


class TestBlenderScriptBoneNames:
    def test_all_22_bones_in_script(self):
        """All 22 Unity Humanoid bones appear in the embedded Blender script.

        The script uses f-string loops for symmetric bones, e.g.:
            make_bone(f"{side}Shoulder", ...)
        We check literal spine bones and the suffix patterns for limbs.
        """
        from auto_rig import _BLENDER_SCRIPT

        # Spine chain: literal names
        literal_bones = ["Hips", "Spine", "Chest", "UpperChest", "Neck", "Head"]
        # Symmetric bones: generated via f"{side}BoneSuffix"
        symmetric_suffixes = [
            "Shoulder", "UpperArm", "LowerArm", "Hand",
            "UpperLeg", "LowerLeg", "Foot", "Toes",
        ]

        for bone in literal_bones:
            assert f'"{bone}"' in _BLENDER_SCRIPT or f"'{bone}'" in _BLENDER_SCRIPT, (
                f"Literal bone '{bone}' not found in _BLENDER_SCRIPT"
            )

        for suffix in symmetric_suffixes:
            assert suffix in _BLENDER_SCRIPT, (
                f"Bone suffix '{suffix}' (for Left/Right{suffix}) not in _BLENDER_SCRIPT"
            )

        assert '"Left"' in _BLENDER_SCRIPT or "'Left'" in _BLENDER_SCRIPT
        assert '"Right"' in _BLENDER_SCRIPT or "'Right'" in _BLENDER_SCRIPT

    def test_exactly_22_bones(self):
        assert len(EXPECTED_BONES) == 22

    def test_left_right_symmetry(self):
        left_bones = [b for b in EXPECTED_BONES if b.startswith("Left")]
        right_bones = [b for b in EXPECTED_BONES if b.startswith("Right")]
        assert len(left_bones) == len(right_bones)

    def test_spine_chain_complete(self):
        spine_bones = ["Hips", "Spine", "Chest", "UpperChest", "Neck", "Head"]
        for bone in spine_bones:
            assert bone in EXPECTED_BONES

    def test_blender_script_bone_hierarchy(self):
        """Verify correct parent-child relationships in _BLENDER_SCRIPT.

        Checks each spine chain bone's parent_name by searching for the
        exact make_bone call block, and verifies arm/leg parents via literal
        string presence.
        """
        from auto_rig import _BLENDER_SCRIPT

        # Spine chain: each bone's make_bone call contains a specific parent_name.
        # We search for the make_bone("BoneName" block and look for parent_name=
        # in the next line(s). Use simple substring checks for robustness.
        spine_parent_assertions = [
            # (bone_name_in_script, expected_parent_name_text)
            ('"Spine"', 'parent_name="Hips"'),
            ('"Chest"', 'parent_name="Spine"'),
            ('"UpperChest"', 'parent_name="Chest"'),
            ('"Neck"', 'parent_name="UpperChest"'),
            ('"Head"', 'parent_name="Neck"'),
        ]
        for bone_token, parent_token in spine_parent_assertions:
            # Find the make_bone call for this bone
            bone_idx = _BLENDER_SCRIPT.index(bone_token)
            # Look ahead up to 300 chars for the parent_name= argument
            snippet = _BLENDER_SCRIPT[bone_idx : bone_idx + 300]
            assert parent_token in snippet, (
                f"Expected {parent_token!r} near make_bone({bone_token})"
            )

        # Arms: shoulder bones (generated via f-string loop) use parent_name="UpperChest"
        assert 'parent_name="UpperChest"' in _BLENDER_SCRIPT, (
            "Expected shoulder bones to have parent_name='UpperChest'"
        )
        # Legs: upper leg bones use parent_name="Hips"
        assert 'parent_name="Hips"' in _BLENDER_SCRIPT, (
            "Expected upper leg bones to have parent_name='Hips'"
        )
