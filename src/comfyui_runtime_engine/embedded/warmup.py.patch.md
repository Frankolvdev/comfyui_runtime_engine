# Required integration patch for `EmbeddedWarmupRunner.run`

MegaZIP 14A includes a complete replacement script (`APLICAR_MEGAZIP_14A.py`)
that applies the following conservative extension to the existing v0.13 runner:

- add keyword arguments:
  - `persist_after_ready: bool = False`
  - `persistent_ready_callback = None`
- after building the successful report:
  - invoke `persistent_ready_callback(report)` when supplied;
  - when `persist_after_ready=True`, keep advancing the existing ComfyUI event
    loop until SIGTERM/SIGINT;
- skip `guard.restore()`, server task cancellation and bootstrap close until the
  persistent loop exits;
- preserve the exact existing path for all local commands when the flag is false.

The patch is applied automatically by the included installer script.
