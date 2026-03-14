# XR Character Creator

Unity XR app (XREAL One Pro) + Python cloud backend (Runpod) that generates rigged 3D characters from natural language descriptions.

## Tech Stack

### Client — Unity (C#)
- **Engine:** Unity 6000.x with Android build support
- **XR:** XREAL SDK 3.0 (AR Foundation + XR Interaction Toolkit)
- **Runtime model loading:** GLTFast (com.unity.cloud.gltfast)
- **Animation:** Humanoid avatar system with Mixamo clips

### Server — Python (Runpod Serverless)
- **Image generation:** Flux (GGUF quantized) — multi-view character reference sheets
- **3D generation:** Hunyuan3D 2.1 — mesh from multi-view images
- **Rigging:** Blender headless (Python API) — auto-rig with humanoid skeleton
- **Package manager:** uv
- **Formatter/Linter:** ruff
- **Python version:** 3.11+

## Project Structure

```
fantasy-island/
├── context/                    # Project briefs and reference docs
├── RunpodBackend/              # Python serverless backend
│   ├── handler.py              # Main Runpod handler (orchestrator)
│   ├── generate_images.py      # Flux multi-view generation
│   ├── generate_mesh.py        # Hunyuan3D pipeline
│   ├── auto_rig.py             # Blender headless rigging
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── .env.example
└── Assets/                     # Unity project root
    ├── Scripts/
    ├── Animations/             # Mixamo .fbx clips
    ├── Prefabs/
    └── Scenes/
```

## Key Commands

### Python Backend (RunpodBackend/)
```bash
cd RunpodBackend
uv sync                        # Install dependencies
uv run ruff check .            # Lint
uv run ruff format .           # Format
uv run python handler.py       # Local test run
docker build -t xr-character . # Build container
```

### Unity
Unity project is managed through the Unity Editor — open the root folder as a Unity project.

## Code Conventions

### Python
- Use ruff for formatting and linting (line length 100)
- Type hints on all function signatures
- Docstrings on public functions (Google style)
- Environment variables for all secrets and configuration (never hardcode)

### C# (Unity)
- PascalCase for classes, methods, properties
- camelCase for local variables and parameters
- `_camelCase` for private fields
- One class per file, filename matches class name
- Use `async/await` for all network and loading operations
- Null-check with `== null` (not `is null`) due to Unity's custom null handling

## Architecture Notes

- The backend pipeline is: Text → LLM prompt expansion → Flux (4 views) → Hunyuan3D → Blender auto-rig → .glb
- The client communicates with Runpod via REST (UnityWebRequest)
- Generated models are uploaded to cloud storage; client downloads via URL
- Animation retargeting requires the Blender rig to output Unity-compatible humanoid bone names
- Target mesh complexity: 10k-50k triangles for mobile XR

## Environment Variables (Backend)

See `RunpodBackend/.env.example` for required variables. Key ones:
- `RUNPOD_API_KEY` — Runpod authentication
- `HF_TOKEN` — HuggingFace token for model downloads

Generated assets (`.glb` files) are returned directly from the serverless handler — Runpod automatically uploads large outputs to its blob storage and returns a presigned URL. No external storage setup needed.
