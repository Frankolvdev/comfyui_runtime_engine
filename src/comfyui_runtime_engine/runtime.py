from __future__ import annotations

from pathlib import Path

from .adapters import select_adapter
from .config import RuntimeConfig
from .embedded import EmbeddedBootstrap, EmbeddedWarmupRunner
from .environment import EnvironmentManager
from .errors import RuntimeEngineError
from .events import EventSink
from .gpu import IsolatedGPUProbe
from .residency import ResidencyManager
from .warmup import AssetLinkManager, WarmupWorkflow


class RuntimeEngine:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.events = EventSink(
            config.event_log if config.json_events else None
        )
        self.adapter, self.inspection = select_adapter(
            config.comfyui_path,
            config.supported_versions,
            config.strict_version,
        )

    def residency_scan(self):
        return self._residency_manager().scan()

    def residency_plan(self):
        return self._residency_manager().plan()

    def residency_verify(self):
        return self._residency_manager().verify()

    def warmup_verify(self) -> dict[str, object]:
        if self.config.warmup_workflow is None:
            raise RuntimeEngineError(
                "residency.warmup_workflow is not configured"
            )
        workflow = WarmupWorkflow.load(self.config.warmup_workflow)
        residency = self.residency_verify()
        targets = {item.target for item in self.config.warmup_inputs}
        missing_inputs = [
            name for name in workflow.required_input_images()
            if name not in targets
        ]
        report = {
            "workflow": str(workflow.path),
            "node_count": workflow.node_count,
            "residency": residency,
            "required_input_images": workflow.required_input_images(),
            "missing_input_mappings": missing_inputs,
            "purge_nodes": workflow.purge_nodes(),
            "workflow_modified": False,
            "sam3_source_model": str(self.config.sam3_source_model),
            "sam3_expected_model": str(self.config.sam3_expected_model),
            "success": bool(residency["success"]) and not missing_inputs,
        }
        self.events.emit("warmup.verify.completed", **report)
        return report

    def warmup_probe(
        self,
        *,
        hold_seconds: float = 10.0,
        iterations: int = 2,
    ) -> dict[str, object]:
        if iterations < 1 or iterations > 5:
            raise RuntimeEngineError("warmup iterations must be between 1 and 5")
        gpu = IsolatedGPUProbe(self.events).run()
        if not gpu.get("success"):
            raise RuntimeEngineError("GPU verification failed before warmup")

        verification = self.warmup_verify()
        if not verification["success"]:
            raise RuntimeEngineError(f"Warmup verification failed: {verification}")

        asset_result = AssetLinkManager(
            replace_existing=self.config.sam3_replace_existing
        ).ensure_file_link(
            name="sam3.pt",
            source=self.config.sam3_source_model,
            expected=self.config.sam3_expected_model,
            mode=self.config.sam3_link_mode,
        )
        if asset_result.status in {
            "source_missing",
            "existing_target_not_replaced",
        }:
            raise RuntimeEngineError(
                "SAM3 asset preparation is not ready: "
                + str(asset_result.to_dict())
            )

        resident_plan = self.residency_plan()
        resident_paths = tuple(
            Path(item["matches"][0]["absolute_path"])
            for item in resident_plan["resolutions"]
            if item["resolved"]
        )

        gpu_args = tuple(
            argument
            for argument in self.config.embedded_extra_args
            if argument != "--cpu"
        )
        report = EmbeddedWarmupRunner(
            self._embedded_bootstrap(extra_args=gpu_args),
            model_roots=self._effective_model_roots(),
            warmup_inputs=self.config.warmup_inputs,
            resident_paths=resident_paths,
        ).run(
            WarmupWorkflow.load(self.config.warmup_workflow),
            timeout_seconds=self.config.warmup_timeout_seconds,
            hold_seconds=hold_seconds,
            iterations=iterations,
            asset_links={"sam3": asset_result.to_dict()},
        )
        report["gpu_probe"] = gpu
        report["resident_plan"] = resident_plan
        report["resident_models_only"] = list(self.config.resident_models)
        return report

    def _residency_manager(self):
        return ResidencyManager(
            comfyui_root=self.config.comfyui_path,
            model_roots=self._effective_model_roots(),
            resident_models=self.config.resident_models,
            events=self.events,
        )

    def _effective_model_roots(self):
        return (
            self.config.model_roots
            or (self.config.comfyui_path / "models",)
        )

    def _embedded_bootstrap(self, extra_args=None):
        manager = EnvironmentManager(
            self.config.comfyui_path, self.events
        )
        if self.config.ensure_workspace:
            manager.ensure_workspace()
        return EmbeddedBootstrap(
            self.inspection,
            self.events,
            host=self.config.host,
            port=self.config.port,
            startup_timeout_seconds=self.config.startup_timeout_seconds,
            extra_args=(
                self.config.embedded_extra_args
                if extra_args is None
                else extra_args
            ),
            shutdown_grace_seconds=self.config.shutdown_grace_seconds,
        )
