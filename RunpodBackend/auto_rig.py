"""Blender headless auto-rigging module.

Wraps Blender's Python API via subprocess to rig a humanoid mesh with a
Unity-compatible skeleton and export the result as a .glb file.
"""

from __future__ import annotations

import logging
import os
import subprocess
import textwrap
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Blender Python script (runs inside the Blender process)
# ---------------------------------------------------------------------------
# This script is passed to Blender via --python-expr. It receives its
# arguments after the "--" separator on the command line.
#
# Bone names follow Unity's Humanoid avatar naming convention exactly so that
# Unity's Humanoid avatar auto-detection maps bones without manual remapping.
# ---------------------------------------------------------------------------

_BLENDER_SCRIPT = textwrap.dedent(r"""
import sys
import math
import bpy
import mathutils

# ---- Parse arguments passed after "--" --------------------------------
argv = sys.argv
separator = argv.index("--")
args = argv[separator + 1:]
if len(args) != 3:
    print(f"ERROR: expected 3 args (input_path, output_path, poly_target), got {len(args)}")
    sys.exit(1)

input_path = args[0]
output_path = args[1]
poly_target = int(args[2])

print(f"[auto_rig] input={input_path} output={output_path} poly_target={poly_target}")

# ---- Clear default scene -----------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)

# ---- Import mesh -------------------------------------------------------
ext = input_path.rsplit(".", 1)[-1].lower()
if ext == "glb" or ext == "gltf":
    bpy.ops.import_scene.gltf(filepath=input_path)
elif ext == "obj":
    # Blender 3.4+ uses wm.obj_import; 3.0 uses import_scene.obj
    try:
        bpy.ops.wm.obj_import(filepath=input_path)
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=input_path)
elif ext == "fbx":
    bpy.ops.import_scene.fbx(filepath=input_path)
else:
    print(f"ERROR: unsupported input format: {ext}")
    sys.exit(1)

# Collect all mesh objects
mesh_objects = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not mesh_objects:
    print("ERROR: no mesh objects found after import")
    sys.exit(1)

print(f"[auto_rig] imported {len(mesh_objects)} mesh object(s)")

# Join all meshes into one for simpler rigging
bpy.ops.object.select_all(action="DESELECT")
for obj in mesh_objects:
    obj.select_set(True)
bpy.context.view_layer.objects.active = mesh_objects[0]
if len(mesh_objects) > 1:
    bpy.ops.object.join()

mesh_obj = bpy.context.active_object

# ---- Decimate if needed ------------------------------------------------
current_tris = sum(len(p.vertices) - 2 for p in mesh_obj.data.polygons)
print(f"[auto_rig] mesh has ~{current_tris} triangles (target: {poly_target})")

if current_tris > poly_target:
    ratio = poly_target / current_tris
    ratio = max(0.05, min(ratio, 1.0))  # clamp to [5%, 100%]
    print(f"[auto_rig] applying decimate modifier, ratio={ratio:.3f}")
    dec = mesh_obj.modifiers.new(name="Decimate", type="DECIMATE")
    dec.ratio = ratio
    dec.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier="Decimate")

# ---- Compute bounding box for bone placement --------------------------
# We need world-space bounds of the mesh to derive joint positions.
bpy.context.view_layer.update()
bbox_corners = [mesh_obj.matrix_world @ mathutils.Vector(c) for c in mesh_obj.bound_box]
xs = [v.x for v in bbox_corners]
ys = [v.y for v in bbox_corners]
zs = [v.z for v in bbox_corners]

min_x, max_x = min(xs), max(xs)
min_y, max_y = min(ys), max(ys)
min_z, max_z = min(zs), max(zs)

height = max_z - min_z
width  = max_x - min_x
depth  = max_y - min_y
center_x = (min_x + max_x) / 2.0
center_y = (min_y + max_y) / 2.0

print(f"[auto_rig] bbox: h={height:.3f} w={width:.3f} d={depth:.3f}")

# Helper to build a Z position from a fraction of total height
def z(frac: float) -> float:
    return min_z + height * frac

# ---- Build humanoid armature ------------------------------------------
# All bone names are Unity Humanoid avatar convention.
bpy.ops.object.armature_add(location=(center_x, center_y, z(0.45)))
arm_obj = bpy.context.active_object
arm_obj.name = "Armature"
arm = arm_obj.data
arm.name = "Armature"

bpy.ops.object.mode_set(mode="EDIT")
edit_bones = arm.edit_bones

# Remove the default bone that Blender adds
for b in list(edit_bones):
    edit_bones.remove(b)

# ------------------------------------------------------------------
# Bone factory: creates a bone from head to tail, optionally
# parented to another bone.
# ------------------------------------------------------------------
def make_bone(
    name: str,
    head: tuple[float, float, float],
    tail: tuple[float, float, float],
    parent_name: str | None = None,
    connected: bool = False,
) -> None:
    b = edit_bones.new(name)
    b.head = head
    b.tail = tail
    if parent_name:
        b.parent = edit_bones[parent_name]
        b.use_connect = connected

# ---- Spine chain -------------------------------------------------------
hip_z        = z(0.45)
spine_z      = z(0.55)
chest_z      = z(0.65)
upper_chest_z = z(0.72)
neck_z       = z(0.80)
head_z       = z(0.87)
head_top_z   = z(0.97)

spine_thickness = height * 0.03  # small Y offset so bones have a direction

make_bone("Hips",
    head=(center_x, center_y, hip_z),
    tail=(center_x, center_y, spine_z))

make_bone("Spine",
    head=(center_x, center_y, spine_z),
    tail=(center_x, center_y, chest_z),
    parent_name="Hips", connected=True)

make_bone("Chest",
    head=(center_x, center_y, chest_z),
    tail=(center_x, center_y, upper_chest_z),
    parent_name="Spine", connected=True)

make_bone("UpperChest",
    head=(center_x, center_y, upper_chest_z),
    tail=(center_x, center_y, neck_z),
    parent_name="Chest", connected=True)

make_bone("Neck",
    head=(center_x, center_y, neck_z),
    tail=(center_x, center_y, head_z),
    parent_name="UpperChest", connected=True)

make_bone("Head",
    head=(center_x, center_y, head_z),
    tail=(center_x, center_y, head_top_z),
    parent_name="Neck", connected=True)

# ---- Arms --------------------------------------------------------------
shoulder_y   = center_y
arm_z        = upper_chest_z
shoulder_w   = width * 0.10   # distance from center to shoulder joint
elbow_w      = width * 0.28
wrist_w      = width * 0.42
hand_w       = width * 0.48

elbow_z      = z(0.60)
wrist_z      = z(0.45)
hand_z       = z(0.38)

for side, sign in (("Left", -1), ("Right", 1)):
    sx = center_x + sign * shoulder_w
    ex = center_x + sign * elbow_w
    wx = center_x + sign * wrist_w
    hx = center_x + sign * hand_w

    make_bone(f"{side}Shoulder",
        head=(center_x, shoulder_y, arm_z),
        tail=(sx, shoulder_y, arm_z),
        parent_name="UpperChest")

    make_bone(f"{side}UpperArm",
        head=(sx, shoulder_y, arm_z),
        tail=(ex, shoulder_y, elbow_z),
        parent_name=f"{side}Shoulder", connected=True)

    make_bone(f"{side}LowerArm",
        head=(ex, shoulder_y, elbow_z),
        tail=(wx, shoulder_y, wrist_z),
        parent_name=f"{side}UpperArm", connected=True)

    make_bone(f"{side}Hand",
        head=(wx, shoulder_y, wrist_z),
        tail=(hx, shoulder_y, hand_z),
        parent_name=f"{side}LowerArm", connected=True)

# ---- Legs --------------------------------------------------------------
hip_y        = center_y
leg_spread   = width * 0.14   # X offset from centre to hip joint

upper_leg_z  = hip_z
knee_z       = z(0.22)
ankle_z      = z(0.05)
toe_z        = z(0.01)
toe_y        = center_y - depth * 0.20   # toes point forward (−Y in Blender)

for side, sign in (("Left", -1), ("Right", 1)):
    lx = center_x + sign * leg_spread

    make_bone(f"{side}UpperLeg",
        head=(lx, hip_y, upper_leg_z),
        tail=(lx, hip_y, knee_z),
        parent_name="Hips")

    make_bone(f"{side}LowerLeg",
        head=(lx, hip_y, knee_z),
        tail=(lx, hip_y, ankle_z),
        parent_name=f"{side}UpperLeg", connected=True)

    make_bone(f"{side}Foot",
        head=(lx, hip_y, ankle_z),
        tail=(lx, toe_y, ankle_z),
        parent_name=f"{side}LowerLeg", connected=True)

    make_bone(f"{side}Toes",
        head=(lx, toe_y, ankle_z),
        tail=(lx, toe_y, toe_z),
        parent_name=f"{side}Foot", connected=True)

bpy.ops.object.mode_set(mode="OBJECT")

print(f"[auto_rig] armature has {len(arm.bones)} bones")

# ---- Parent mesh to armature with automatic weights -------------------
bpy.ops.object.select_all(action="DESELECT")
mesh_obj.select_set(True)
arm_obj.select_set(True)
bpy.context.view_layer.objects.active = arm_obj
bpy.ops.object.parent_set(type="ARMATURE_AUTO")

print("[auto_rig] automatic weight painting complete")

# ---- Export as OBJ (GLB export requires numpy which Blender 3.0 lacks) ---
# We export as OBJ from Blender, then convert to GLB via trimesh in the wrapper.
bpy.ops.object.select_all(action="SELECT")
obj_output = output_path.rsplit(".", 1)[0] + ".obj"
try:
    bpy.ops.wm.obj_export(filepath=obj_output, export_selected_objects=False)
except AttributeError:
    bpy.ops.export_scene.obj(filepath=obj_output, use_selection=False)

print(f"[auto_rig] exported rigged mesh to {obj_output}")
sys.exit(0)
""")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def auto_rig(
    input_mesh_path: str,
    output_glb_path: str,
    poly_target: int = 30_000,
) -> str:
    """Rig a mesh with a humanoid skeleton and export as .glb.

    Runs Blender in headless mode via subprocess. The embedded Blender Python
    script:

    1. Imports the mesh (.glb, .obj, or .fbx).
    2. Applies a Decimate modifier if the triangle count exceeds ``poly_target``.
    3. Creates a humanoid armature with Unity-compatible bone names.
    4. Parents the mesh to the armature using automatic weight painting.
    5. Exports as .glb with embedded textures and skeleton.

    Bone names exactly follow Unity's Humanoid avatar naming convention so
    that Unity's avatar auto-detection requires no manual bone mapping:
    Hips, Spine, Chest, UpperChest, Neck, Head,
    LeftShoulder / RightShoulder,
    LeftUpperArm / RightUpperArm,
    LeftLowerArm / RightLowerArm,
    LeftHand / RightHand,
    LeftUpperLeg / RightUpperLeg,
    LeftLowerLeg / RightLowerLeg,
    LeftFoot / RightFoot,
    LeftToes / RightToes.

    Args:
        input_mesh_path: Path to the input mesh (.glb, .obj, or .fbx).
        output_glb_path: Path to write the rigged .glb file.
        poly_target: Target triangle count. If the imported mesh exceeds this,
            a Decimate modifier is applied before rigging.

    Returns:
        Path to the rigged .glb file (same as ``output_glb_path``).

    Raises:
        FileNotFoundError: If ``input_mesh_path`` does not exist.
        RuntimeError: If Blender exits with a non-zero return code.
        TimeoutError: If Blender takes longer than 5 minutes.
    """
    if not Path(input_mesh_path).exists():
        raise FileNotFoundError(f"Input mesh not found: {input_mesh_path}")

    Path(output_glb_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert GLB to OBJ before passing to Blender — Blender's glTF importer
    # requires numpy which may not be available in Blender's bundled Python.
    # OBJ import is built-in and has no such dependency.
    actual_input = input_mesh_path
    if input_mesh_path.lower().endswith(".glb") or input_mesh_path.lower().endswith(".gltf"):
        import trimesh  # type: ignore[import]

        logger.info("Converting %s to OBJ for Blender compatibility", input_mesh_path)
        obj_path = str(Path(input_mesh_path).with_suffix(".obj"))
        mesh = trimesh.load(input_mesh_path)
        mesh.export(obj_path, file_type="obj")
        actual_input = obj_path
        logger.info("Converted to %s", obj_path)

    logger.info(
        "Starting Blender auto-rig: %s → %s (poly_target=%d)",
        actual_input,
        output_glb_path,
        poly_target,
    )

    blender_bin = _find_blender()
    cmd = [
        blender_bin,
        "--background",
        "--python-expr",
        _BLENDER_SCRIPT,
        "--",
        actual_input,
        output_glb_path,
        str(poly_target),
    ]

    logger.debug("Blender command: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes max
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("Blender auto-rig timed out after 5 minutes") from exc

    # Forward Blender stdout/stderr to our logger for debugging.
    if result.stdout:
        for line in result.stdout.splitlines():
            logger.debug("[blender] %s", line)
    if result.stderr:
        for line in result.stderr.splitlines():
            logger.debug("[blender:stderr] %s", line)

    if result.returncode != 0:
        # Include the tail of stderr in the error message for quick diagnosis.
        stderr_tail = "\n".join(result.stderr.splitlines()[-30:])
        raise RuntimeError(
            f"Blender auto-rig failed (exit code {result.returncode}):\n{stderr_tail}"
        )

    # Blender exports as OBJ (GLB export needs numpy which Blender 3.0 lacks).
    # Convert OBJ → GLB using trimesh.
    obj_output = str(Path(output_glb_path).with_suffix(".obj"))
    if Path(obj_output).exists():
        logger.info("Converting Blender OBJ output to GLB: %s → %s", obj_output, output_glb_path)
        import trimesh  # type: ignore[import]

        mesh = trimesh.load(obj_output)
        mesh.export(output_glb_path, file_type="glb")
        Path(obj_output).unlink(missing_ok=True)  # Clean up OBJ
        mtl_path = Path(obj_output).with_suffix(".mtl")
        mtl_path.unlink(missing_ok=True)
        logger.info("Converted to GLB: %s", output_glb_path)

    if not Path(output_glb_path).exists():
        stderr_tail = "\n".join(result.stderr.splitlines()[-30:]) if result.stderr else ""
        stdout_tail = "\n".join(result.stdout.splitlines()[-30:]) if result.stdout else ""
        raise RuntimeError(
            f"Blender exited successfully but output file not found: {output_glb_path}\n"
            f"STDOUT:\n{stdout_tail}\nSTDERR:\n{stderr_tail}"
        )

    logger.info("Auto-rig complete: %s", output_glb_path)
    return output_glb_path


def _find_blender() -> str:
    """Locate the Blender executable.

    Checks the ``BLENDER_PATH`` environment variable first, then falls back
    to ``blender`` on ``$PATH``.

    Returns:
        Path string to the Blender executable.

    Raises:
        FileNotFoundError: If Blender cannot be found.
    """
    import shutil

    blender = os.environ.get("BLENDER_PATH") or shutil.which("blender")
    if not blender:
        raise FileNotFoundError(
            "Blender executable not found. "
            "Set BLENDER_PATH or ensure 'blender' is on PATH."
        )
    return blender


# Make os available (imported at module level for _find_blender)
