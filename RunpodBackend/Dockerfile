# XR Character Creator — Runpod Serverless Backend
#
# Build stages:
#   1. Base CUDA image with system deps and Blender
#   2. Python deps via uv
#   3. Model weight pre-download (baked into image for zero cold-start latency)
#
# Resulting image is large (~30-50 GB) but starts instantly on Runpod.

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
# blender: headless 3D rigging (auto_rig.py)
# libgl1 / libglib2.0-0: required by OpenCV / PIL for image I/O
# git: needed to clone Hunyuan3D repo if pip package is unavailable
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        blender \
        libgl1 \
        libglib2.0-0 \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python toolchain — uv for fast, reproducible installs
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir uv

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
WORKDIR /app

# Copy dependency manifests first so Docker can cache this layer.
COPY pyproject.toml uv.lock ./

# Install all Python deps from the lockfile (no network after this layer).
RUN uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# Hunyuan3D — install from source (no stable PyPI package yet)
# ---------------------------------------------------------------------------
# We clone at a pinned commit for reproducibility. Update the SHA as needed.
RUN git clone https://github.com/Tencent/Hunyuan3D-2.git /opt/hunyuan3d && \
    cd /opt/hunyuan3d && \
    pip install --no-cache-dir -e ".[inference]"

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
COPY handler.py generate_images.py generate_mesh.py auto_rig.py ./

# ---------------------------------------------------------------------------
# Pre-download model weights (baked into image)
# ---------------------------------------------------------------------------
# Models are downloaded at build time so the container starts instantly.
#
# HF_TOKEN is passed via Docker BuildKit secret — it is NEVER written into
# any image layer and will not appear in `docker history` output.
#
# Build command:
#   docker build --secret id=hf_token,env=HF_TOKEN .
#
# The secret is mounted as a file at /run/secrets/hf_token for the duration
# of each RUN step only, then discarded.

# Flux GGUF weights (~12-16 GB)
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token 2>/dev/null || true) \
    python -c "\
from generate_images import _get_pipeline; \
print('Pre-loading Flux pipeline...'); \
_get_pipeline(); \
print('Flux pre-load complete.')"

# Hunyuan3D 2.1 weights (~20-30 GB)
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token 2>/dev/null || true) \
    python -c "\
from generate_mesh import _load_hunyuan; \
print('Pre-loading Hunyuan3D pipeline...'); \
_load_hunyuan(); \
print('Hunyuan3D pre-load complete.')"

# ---------------------------------------------------------------------------
# Runtime environment
# ---------------------------------------------------------------------------
# HF_TOKEN and RUNPOD_API_KEY are injected at runtime via Runpod env vars.
# They are NOT baked into the image.
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Blender executable path (used by auto_rig._find_blender)
ENV BLENDER_PATH=/usr/bin/blender

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
CMD ["python", "-u", "handler.py"]
