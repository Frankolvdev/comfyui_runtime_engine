RUNTIMEENGINE09 — Disable DynamicVRAM during Modal GPU snapshot warmup

Base: comfyui_runtime_engine-main (3).zip

Production files changed:
- src/comfyui_runtime_engine/modal/persistent.py
- src/comfyui_runtime_engine/embedded/warmup.py

Behavior:
- Embedded ComfyUI used for the Modal GPU snapshot now always receives
  --disable-dynamic-vram (unless already supplied).
- This aligns snapshot warmup with TRYON's normal Modal runtime policy.
- Keeps force_full_load=True as a final promotion/safety check.
- Fails with detailed loaded_bytes / total_model_bytes diagnostics if a configured
  resident is still not fully resident on CUDA.

Validation performed:
8 passed.
