"""End-to-end test harness for the Runpod serverless endpoint.

Submits jobs to a live Runpod serverless endpoint and validates the results.
Supports testing individual pipeline stages (when the endpoint supports stage isolation)
or the full pipeline end-to-end.

Usage:
    # Full pipeline test (default)
    uv run python e2e_test.py

    # Save output artifacts to a directory
    uv run python e2e_test.py --output-dir ./test_output

    # Custom description
    uv run python e2e_test.py --description "A robot warrior with chrome armor"

    # Custom style and poly target
    uv run python e2e_test.py --style sci-fi --poly-target 15000

    # Set timeout (default 600s / 10 min)
    uv run python e2e_test.py --timeout 900

Requires:
    RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID in .env or environment variables.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv is optional — env vars can be set directly


# ---------------------------------------------------------------------------
# Runpod serverless client
# ---------------------------------------------------------------------------

def _get_config() -> tuple[str, str]:
    """Read API key and endpoint ID from environment.

    Returns:
        Tuple of (api_key, endpoint_id).

    Raises:
        SystemExit: If required env vars are missing.
    """
    api_key = os.getenv("RUNPOD_API_KEY", "").strip()
    endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID", "").strip()

    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)
    if not endpoint_id:
        print("ERROR: RUNPOD_ENDPOINT_ID not set. Add it to .env or export it.")
        print("  Deploy the serverless endpoint first, then set this to the endpoint ID.")
        sys.exit(1)

    return api_key, endpoint_id


def submit_job(api_key: str, endpoint_id: str, payload: dict) -> str:
    """Submit an async job to the Runpod serverless endpoint.

    Args:
        api_key: Runpod API key.
        endpoint_id: Runpod serverless endpoint ID.
        payload: Job input payload.

    Returns:
        Job ID string.
    """
    import urllib.request

    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    data = json.dumps({"input": payload}).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode())

    if "id" not in result:
        print(f"ERROR: Unexpected response from /run: {result}")
        sys.exit(1)

    return result["id"]


def poll_status(
    api_key: str, endpoint_id: str, job_id: str, timeout: int = 600
) -> dict:
    """Poll the Runpod endpoint for job completion.

    Uses exponential backoff: 2s → 5s → 10s → 15s intervals.
    Only charges while the worker is active — polling is free.

    Args:
        api_key: Runpod API key.
        endpoint_id: Runpod serverless endpoint ID.
        job_id: Job ID to poll.
        timeout: Max seconds to wait before giving up.

    Returns:
        Final job result dict.
    """
    import urllib.request

    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    req_headers = {"Authorization": f"Bearer {api_key}"}

    intervals = [2, 2, 5, 5, 10, 10, 15]  # backoff schedule
    start = time.monotonic()
    poll_count = 0

    while True:
        elapsed = time.monotonic() - start
        if elapsed > timeout:
            print(f"\nERROR: Job timed out after {timeout}s")
            sys.exit(1)

        req = urllib.request.Request(url, headers=req_headers)
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode())

        status = result.get("status", "UNKNOWN")

        if status == "COMPLETED":
            print(f"\n  Job completed in {elapsed:.1f}s")
            return result.get("output", {})

        if status == "FAILED":
            error = result.get("error", "Unknown error")
            print(f"\n  ERROR: Job failed — {error}")
            sys.exit(1)

        if status == "CANCELLED":
            print("\n  ERROR: Job was cancelled")
            sys.exit(1)

        # Progress display
        progress_info = ""
        if status == "IN_PROGRESS":
            output = result.get("output", {})
            if isinstance(output, dict):
                stage = output.get("stage", "")
                progress = output.get("progress", "")
                if stage:
                    progress_info = f" [{stage} {progress}%]"

        interval = intervals[min(poll_count, len(intervals) - 1)]
        print(
            f"  [{elapsed:5.0f}s] Status: {status}{progress_info} "
            f"(next check in {interval}s)",
            end="\r",
        )
        time.sleep(interval)
        poll_count += 1


# ---------------------------------------------------------------------------
# GLB validation (standalone, no external deps)
# ---------------------------------------------------------------------------

UNITY_HUMANOID_BONES = {
    "Hips", "Spine", "Chest", "UpperChest", "Neck", "Head",
    "LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand",
    "RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
    "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes",
}


def validate_glb(glb_bytes: bytes) -> dict:
    """Validate a .glb binary and check for Unity Humanoid compatibility.

    Parses the glTF JSON chunk to inspect nodes, skins, meshes, and textures.
    No external dependencies required.

    Args:
        glb_bytes: Raw .glb file bytes.

    Returns:
        Dict with validation results:
            - valid_glb: bool
            - has_skeleton: bool
            - bone_names: list[str]
            - missing_bones: list[str]
            - unity_compatible: bool
            - has_textures: bool
            - triangle_estimate: int
            - file_size_mb: float
            - errors: list[str]
    """
    result = {
        "valid_glb": False,
        "has_skeleton": False,
        "bone_names": [],
        "missing_bones": [],
        "unity_compatible": False,
        "has_textures": False,
        "triangle_estimate": 0,
        "file_size_mb": round(len(glb_bytes) / (1024 * 1024), 2),
        "errors": [],
    }

    # Parse GLB header
    if len(glb_bytes) < 12:
        result["errors"].append("File too small to be a valid GLB")
        return result

    magic, version, length = struct.unpack_from("<III", glb_bytes, 0)
    if magic != 0x46546C67:  # "glTF"
        result["errors"].append(f"Invalid GLB magic: {magic:#x}")
        return result

    if version != 2:
        result["errors"].append(f"Unsupported glTF version: {version}")
        return result

    # Parse JSON chunk
    if len(glb_bytes) < 20:
        result["errors"].append("Missing JSON chunk")
        return result

    chunk_length, chunk_type = struct.unpack_from("<II", glb_bytes, 12)
    if chunk_type != 0x4E4F534A:  # "JSON"
        result["errors"].append("First chunk is not JSON")
        return result

    json_data = glb_bytes[20: 20 + chunk_length].decode("utf-8")
    try:
        gltf = json.loads(json_data)
    except json.JSONDecodeError as e:
        result["errors"].append(f"Invalid JSON in GLB: {e}")
        return result

    result["valid_glb"] = True

    # Check for skeleton (skins)
    skins = gltf.get("skins", [])
    nodes = gltf.get("nodes", [])

    if skins:
        result["has_skeleton"] = True
        # Collect bone names from skin joints
        for skin in skins:
            for joint_idx in skin.get("joints", []):
                if joint_idx < len(nodes):
                    name = nodes[joint_idx].get("name", f"unnamed_{joint_idx}")
                    result["bone_names"].append(name)

    # Check Unity compatibility
    bone_set = set(result["bone_names"])
    result["missing_bones"] = sorted(UNITY_HUMANOID_BONES - bone_set)
    result["unity_compatible"] = len(result["missing_bones"]) == 0

    # Check textures
    textures = gltf.get("textures", [])
    images = gltf.get("images", [])
    result["has_textures"] = len(textures) > 0 or len(images) > 0

    # Estimate triangle count from accessors
    meshes = gltf.get("meshes", [])
    accessors = gltf.get("accessors", [])
    total_indices = 0
    for mesh in meshes:
        for prim in mesh.get("primitives", []):
            indices_idx = prim.get("indices")
            if indices_idx is not None and indices_idx < len(accessors):
                total_indices += accessors[indices_idx].get("count", 0)
    result["triangle_estimate"] = total_indices // 3 if total_indices > 0 else 0

    return result


def print_validation(v: dict) -> None:
    """Pretty-print GLB validation results."""
    print("\n--- GLB Validation ---")
    print(f"  File size:        {v['file_size_mb']:.2f} MB")
    print(f"  Valid GLB:        {'PASS' if v['valid_glb'] else 'FAIL'}")
    print(f"  Has skeleton:     {'PASS' if v['has_skeleton'] else 'FAIL'}")
    print(f"  Has textures:     {'PASS' if v['has_textures'] else 'FAIL'}")
    print(f"  Triangle count:   ~{v['triangle_estimate']:,}")
    print(f"  Unity compatible: {'PASS' if v['unity_compatible'] else 'FAIL'}")

    if v["bone_names"]:
        print(f"  Bones found ({len(v['bone_names'])}): {', '.join(sorted(v['bone_names']))}")
    if v["missing_bones"]:
        print(f"  MISSING bones:    {', '.join(v['missing_bones'])}")
    if v["errors"]:
        print(f"  Errors:           {'; '.join(v['errors'])}")
    print("----------------------")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_e2e_test(args: argparse.Namespace) -> None:
    """Run the end-to-end test against a live Runpod endpoint."""
    api_key, endpoint_id = _get_config()

    payload = {
        "description": args.description,
        "style": args.style,
        "poly_target": args.poly_target,
    }

    print("=" * 60)
    print("XR Character Creator — E2E Test")
    print("=" * 60)
    print(f"  Endpoint:    {endpoint_id}")
    print(f"  Description: {payload['description'][:60]}...")
    print(f"  Style:       {payload['style']}")
    print(f"  Poly target: {payload['poly_target']:,}")
    print(f"  Timeout:     {args.timeout}s")
    print()

    # Submit job
    print("Submitting job...")
    job_id = submit_job(api_key, endpoint_id, payload)
    print(f"  Job ID: {job_id}")
    print()

    # Poll for completion (only charges while worker is active)
    print("Waiting for completion (serverless — you're only charged during processing)...")
    output = poll_status(api_key, endpoint_id, job_id, timeout=args.timeout)

    # Extract results
    gen_time = output.get("generation_time_seconds", 0)
    print(f"  Generation time: {gen_time:.1f}s")

    glb_b64 = output.get("glb", "")
    thumbnail_b64 = output.get("thumbnail", "")

    if not glb_b64:
        print("ERROR: No GLB data in response")
        sys.exit(1)

    glb_bytes = base64.b64decode(glb_b64)
    print(f"  GLB size: {len(glb_bytes) / (1024 * 1024):.2f} MB")

    # Validate GLB
    validation = validate_glb(glb_bytes)
    print_validation(validation)

    # Save artifacts if requested
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save GLB
        glb_path = out_dir / "character.glb"
        glb_path.write_bytes(glb_bytes)
        print(f"\n  Saved GLB:       {glb_path}")

        # Save thumbnail
        if thumbnail_b64:
            thumb_bytes = base64.b64decode(thumbnail_b64)
            thumb_path = out_dir / "thumbnail.png"
            thumb_path.write_bytes(thumb_bytes)
            print(f"  Saved thumbnail: {thumb_path}")

        # Save validation report
        report_path = out_dir / "validation.json"
        report = {
            "job_id": job_id,
            "payload": payload,
            "generation_time_seconds": gen_time,
            "validation": {
                k: v for k, v in validation.items()
                if k != "bone_names"  # keep report concise
            },
            "bone_names": validation["bone_names"],
        }
        report_path.write_text(json.dumps(report, indent=2))
        print(f"  Saved report:    {report_path}")

    # Final verdict
    print()
    all_pass = (
        validation["valid_glb"]
        and validation["has_skeleton"]
        and validation["unity_compatible"]
    )
    if all_pass:
        print("RESULT: PASS — GLB is valid, has skeleton, Unity-compatible")
    else:
        print("RESULT: FAIL — see validation details above")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E2E test for XR Character Creator Runpod endpoint"
    )
    parser.add_argument(
        "--description",
        default="A tall elven warrior woman with silver hair, emerald eyes, "
                "leather armor with gold trim, pointed ears, athletic build",
        help="Character description to generate",
    )
    parser.add_argument(
        "--style",
        default="fantasy",
        choices=["fantasy", "sci-fi", "realistic", "anime", "cartoon"],
        help="Art style (default: fantasy)",
    )
    parser.add_argument(
        "--poly-target",
        type=int,
        default=30000,
        help="Target polygon count (default: 30000)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save GLB, thumbnail, and validation report",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Max seconds to wait for job completion (default: 600)",
    )
    args = parser.parse_args()
    run_e2e_test(args)


if __name__ == "__main__":
    main()
