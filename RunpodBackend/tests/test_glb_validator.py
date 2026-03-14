"""Unit tests for the GLB validation utility (validate_glb / UNITY_HUMANOID_BONES).

No GPU or Blender required — all tests use in-memory or tmp_path GLB fixtures.
"""

from __future__ import annotations

import pytest
from conftest import UNITY_HUMANOID_BONES, make_minimal_glb, validate_glb


class TestUnityHumanoidBones:
    def test_exactly_22_bones(self):
        assert len(UNITY_HUMANOID_BONES) == 22

    def test_no_duplicates(self):
        assert len(UNITY_HUMANOID_BONES) == len(set(UNITY_HUMANOID_BONES))

    def test_spine_chain_present(self):
        spine = ["Hips", "Spine", "Chest", "UpperChest", "Neck", "Head"]
        for bone in spine:
            assert bone in UNITY_HUMANOID_BONES

    def test_left_right_symmetry(self):
        left_bones = [b for b in UNITY_HUMANOID_BONES if b.startswith("Left")]
        right_bones = [b for b in UNITY_HUMANOID_BONES if b.startswith("Right")]
        # Every Left bone has a Right counterpart
        assert len(left_bones) == len(right_bones) == 8
        for bone in left_bones:
            assert "Right" + bone[4:] in right_bones


class TestValidateGlb:
    def test_validate_rigged_fixture(self, sample_glb_file):
        """A GLB with all 22 bones passes full validation."""
        val = validate_glb(str(sample_glb_file))
        assert val["valid"] is True
        assert val["errors"] == []
        assert val["has_skin"] is True
        assert val["missing_bones"] == []
        assert len(val["bone_names"]) == 22

    def test_validate_unrigged_fixture(self, sample_glb_no_skeleton):
        """A GLB with no armature reports has_skin=False, all 22 bones missing."""
        val = validate_glb(str(sample_glb_no_skeleton))
        assert val["valid"] is False
        assert val["has_skin"] is False
        assert len(val["missing_bones"]) == 22

    def test_validate_missing_bones(self, sample_glb_missing_bones):
        """A GLB with only Hips+Spine reports the other 20 bones missing."""
        val = validate_glb(str(sample_glb_missing_bones))
        assert val["valid"] is False
        assert "Hips" in val["bone_names"]
        assert "Spine" in val["bone_names"]
        assert len(val["missing_bones"]) == 20

    def test_validate_nonexistent_file(self, tmp_path):
        """Raises FileNotFoundError for a path that does not exist."""
        with pytest.raises(FileNotFoundError):
            validate_glb(str(tmp_path / "does_not_exist.glb"))

    def test_validate_corrupt_file(self, tmp_path):
        """Reports errors for invalid GLB data."""
        bad = tmp_path / "corrupt.glb"
        bad.write_bytes(b"NOT A VALID GLB FILE AT ALL")
        val = validate_glb(str(bad))
        assert val["valid"] is False
        assert len(val["errors"]) > 0

    def test_validate_empty_file(self, tmp_path):
        """Reports errors for a zero-byte file."""
        empty = tmp_path / "empty.glb"
        empty.write_bytes(b"")
        val = validate_glb(str(empty))
        assert val["valid"] is False

    def test_validate_tiny_glb(self, tiny_glb):
        """A mesh-only GLB (no skeleton) reports has_skin=False."""
        val = validate_glb(tiny_glb)
        assert val["has_skin"] is False
        assert val["valid"] is False  # no skeleton = invalid for Unity humanoid

    def test_extra_bones_reported(self, tmp_path):
        """Bones beyond the 22 required are reported in extra_bones."""
        extra = UNITY_HUMANOID_BONES + ["LeftPinky", "RightPinky"]
        glb = make_minimal_glb(tmp_path / "extra.glb", bones=extra)
        val = validate_glb(str(glb))
        assert "LeftPinky" in val["extra_bones"]
        assert "RightPinky" in val["extra_bones"]
        # Should still be valid — extra bones don't break Unity humanoid
        assert val["valid"] is True
        assert val["missing_bones"] == []

    def test_bone_names_sorted(self, sample_glb_file):
        """bone_names list is sorted alphabetically."""
        val = validate_glb(str(sample_glb_file))
        assert val["bone_names"] == sorted(val["bone_names"])

    def test_all_22_bones_checked(self, tmp_path):
        """Validate each bone individually — removing any one makes it invalid."""
        for missing_bone in UNITY_HUMANOID_BONES:
            remaining = [b for b in UNITY_HUMANOID_BONES if b != missing_bone]
            glb = make_minimal_glb(tmp_path / f"missing_{missing_bone}.glb", bones=remaining)
            val = validate_glb(str(glb))
            assert not val["valid"], f"Expected invalid when '{missing_bone}' is absent"
            assert missing_bone in val["missing_bones"]
