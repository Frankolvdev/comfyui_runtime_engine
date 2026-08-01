# MegaZIP 04 — Stable Environment and Clean Embedded Shutdown

This increment stabilizes the local environment before enabling CUDA.

## Added

- One combined pip resolution for all ComfyUI and custom-node requirements.
- Automatic `pip check` after synchronization.
- Critical import audit.
- Writable `models`, `input`, `output`, `temp`, and `user` preparation.
- `env verify` and `env prepare` commands.
- Graceful shutdown window for ComfyUI-Manager background tasks.

## Commands

```powershell
comfy-runtime --config runtime-engine.toml env prepare
comfy-runtime --config runtime-engine.toml env verify
comfy-runtime --config runtime-engine.toml server-probe --hold-seconds 2
```

A missing optional import such as Triton on CPU Windows is reported but does not make `env verify` fail unless `pip check` or workspace permissions fail.
