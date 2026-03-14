# XR Character Creator — Project Brief

## Overview
Build a Unity XR application for XREAL One Pro glasses that lets a user describe a character in natural language, sends that description to a cloud AI pipeline on Runpod, and receives back a fully rigged 3D character model that appears in the user's physical space with idle animation and voice-commanded poses.

## Hardware & Development Environment
- **Development machine:** M2 Max MacBook Pro, 32GB unified memory
- **XR device:** XREAL One Pro with X1 spatial computing chip
- **Android target:** BeamPro or compatible Samsung phone
- **Cloud GPU:** Runpod serverless (L40S, 48GB VRAM)

## Architecture

### Client (Unity on Android → XREAL One Pro)
- Unity 2022.3 LTS or newer
- XREAL SDK 3.0 (AR Foundation + XR Interaction Toolkit)
- GLTFast for runtime .glb loading
- Simple UI: text input field or voice input for character description
- Animator controller with Humanoid avatar setup
- Pre-loaded animation library (idle sway, common poses, gestures)
- HTTP client (UnityWebRequest) to call Runpod endpoint
- Plane detection to place character on real-world floor
- Loading indicator while generation is in progress

### Server (Runpod Serverless — Docker Container)
- **Flux (GGUF quantized)** — generates multi-angle character reference images from text prompt
  - Front, left side, right side, back views
  - Clean white/neutral background for each view
  - Consistent character appearance across all views via prompt engineering
- **Hunyuan3D 2.1 (multi-view mode)** — takes the reference images and generates:
  - Textured 3D mesh (.glb)
  - PBR maps: albedo, normal, roughness, metallic
  - Target poly count suitable for mobile XR rendering (~10k-50k polys)
- **Blender headless (Python API)** — auto-rigs the mesh:
  - Humanoid skeleton compatible with Unity's Humanoid avatar system
  - Standard bone naming convention
  - Optional: apply decimation if mesh is too heavy for mobile
- Returns a single .glb file with embedded textures and skeleton

### API Contract

**Request:**
```json
POST /generate-character
{
  "description": "A tall elven warrior woman with silver hair, emerald eyes, leather armor with gold trim, pointed ears, athletic build",
  "style": "fantasy",
  "poly_target": 30000
}
```

**Response:**
```json
{
  "status": "completed",
  "model_url": "https://storage.runpod.io/..../character.glb",
  "thumbnail_url": "https://storage.runpod.io/..../preview.png",
  "generation_time_seconds": 240
}
```

## Implementation Phases

### Phase 1 — Runpod Backend (Week 1)
1. Create Docker container with Flux + Hunyuan3D 2.1 + Blender
2. Write the Python handler that orchestrates the pipeline:
   - Take text description as input
   - Use an LLM (small local model or API call) to expand the description into a structured multi-view prompt
   - Generate 4 reference images with Flux (front, left, right, back)
   - Feed images into Hunyuan3D multi-view mode
   - Run Blender auto-rig script on the output mesh
   - Upload final .glb to Runpod storage or S3
   - Return URL
3. Deploy as Runpod serverless endpoint
4. Test via curl/Postman to verify end-to-end

### Phase 2 — Unity Client Foundation (Week 1-2)
1. Create new Unity project with XREAL SDK 3.0
2. Set up AR Foundation: plane detection, XR Origin
3. Import GLTFast package for runtime .glb loading
4. Build basic scene: AR camera, detected planes, placement indicator
5. Create CharacterManager script:
   - HTTP POST to Runpod endpoint
   - Download .glb from returned URL
   - Import mesh at runtime using GLTFast
   - Place on detected floor plane
6. Create Animator controller:
   - Import idle/breathing animation from Mixamo (pre-downloaded .fbx clips)
   - Set up Humanoid avatar retargeting
   - Default state: idle sway
7. Test in Unity editor with a pre-made .glb first, then test with live Runpod calls

### Phase 3 — Voice Commands & Pose Library (Week 2-3)
1. Integrate Whisper (tiny model) for speech-to-text
   - Can run on-device or use a lightweight API
2. Build command interpreter:
   - Map spoken commands to animation triggers
   - "wave" → play wave animation
   - "sit" → play sit animation
   - "turn around" → play 360 turn animation
   - "pose" → play hero pose animation
3. Expand Mixamo animation library:
   - Download 20-30 common animations as .fbx
   - Import into Unity, configure as Humanoid clips
   - All clips auto-retarget to any Humanoid avatar
4. Add smooth transitions between animations (crossfade)

### Phase 4 — Polish & Deploy (Week 3-4)
1. Build and deploy APK to Android device
2. Test full flow through XREAL One Pro glasses
3. Optimize mesh loading time and rendering performance
4. Add character persistence (save/load generated characters)
5. Add ability to have multiple characters in scene simultaneously
6. Tune lighting and shadows for AR realism

## Key Technical Decisions

### Runtime .glb Loading in Unity
Use GLTFast (com.unity.cloud.gltfast) — it's Unity's official glTF importer, supports runtime loading on Android, handles PBR materials automatically, and is actively maintained.

```csharp
// Simplified runtime loading example
var gltf = new GltfImport();
var success = await gltf.Load(glbUrl);
if (success) {
    var characterGO = new GameObject("Character");
    await gltf.InstantiateMainSceneAsync(characterGO.transform);
    // Configure Animator, place on floor plane, etc.
}
```

### Animation Retargeting
Unity's Humanoid animation system handles this automatically IF the rig uses standard humanoid bone names. The Blender auto-rig step must output bones that Unity recognizes. The Rigify addon or Auto-Rig Pro in Blender can do this. Alternatively, use a simple bone-mapping script.

### Mesh Optimization for Mobile XR
AI-generated meshes can be heavy. Target guidelines:
- 10k-30k triangles for characters viewed at arm's length
- 30k-50k for hero characters viewed up close
- Use Blender's decimate modifier in the pipeline if needed
- Texture resolution: 1024x1024 or 2048x2048 max for mobile

## File Structure
```
XRCharacterCreator/
├── Assets/
│   ├── Scripts/
│   │   ├── CharacterManager.cs      // Handles Runpod API + loading
│   │   ├── CharacterPlacer.cs       // AR plane detection + placement
│   │   ├── VoiceCommandHandler.cs   // Speech-to-text + command mapping
│   │   ├── AnimationController.cs   // Animation state management
│   │   └── RunpodClient.cs          // HTTP client for Runpod API
│   ├── Animations/
│   │   ├── IdleSway.fbx
│   │   ├── Wave.fbx
│   │   ├── Sit.fbx
│   │   └── ... (Mixamo clips)
│   ├── Prefabs/
│   │   ├── CharacterRoot.prefab     // Template with Animator
│   │   └── PlacementIndicator.prefab
│   └── Scenes/
│       └── MainXR.unity
├── RunpodBackend/
│   ├── Dockerfile
│   ├── handler.py                   // Main Runpod handler
│   ├── generate_images.py           // Flux multi-view generation
│   ├── generate_mesh.py             // Hunyuan3D pipeline
│   ├── auto_rig.py                  // Blender headless rigging
│   └── requirements.txt
└── README.md
```

## Dependencies & Accounts Needed
- Unity Hub + Unity 2022.3+ with Android build support
- XREAL SDK 3.0 (download from developer.xreal.com)
- Runpod account with ~$25 credit to start
- HuggingFace account (for downloading model weights)
- Mixamo account (free, for animation clips)
- GLTFast Unity package (via Package Manager)

## Success Criteria for Prototype
- [ ] User types a character description
- [ ] Character appears as a 3D model in XR within 5 minutes
- [ ] Character stands on a detected real-world surface
- [ ] Character has subtle idle animation (breathing/sway)
- [ ] Character responds to at least 3 voice commands with different animations
- [ ] Works end-to-end through XREAL One Pro glasses
