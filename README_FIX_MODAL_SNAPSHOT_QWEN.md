# FIX incremental — Qwen 3 8B residente en snapshot Modal

Este paquete agrega únicamente `text_encoders/qwen_3_8b.safetensors` al snapshot Modal existente.

Cambios:
- Lo añade a `DEFAULT_MODAL_RESIDENT_MODELS`.
- Añade un `CLIPLoader` al workflow de warmup con `type = flux2` y `device = default`.
- Añade un sink independiente para mantener una referencia fuerte al CLIP durante la captura.
- Actualiza únicamente los tests contractuales de MegaZIP 14B para exigir ambos modelos residentes.

No modifica Beam, RunPod, endpoints, cancelaciones, pipeline, proxy HTTP/WebSocket ni la lógica restante de Modal.

Después de aplicar:
1. Ejecutar los tests.
2. Reiniciar el backend.
3. Exportar nuevamente el runtime Modal.
4. Hacer build y deploy.
5. Verificar `resident_count: 2` y `expected_resident_count: 2`.
