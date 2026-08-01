from __future__ import annotations

from dataclasses import asdict, dataclass
import subprocess

from ..events import EventSink


@dataclass(slots=True)
class GPUDevice:
    index: int
    name: str
    total_memory_bytes: int
    capability: tuple[int, int] | None


class GPUProbe:
    """Non-destructive CUDA diagnostics. It never loads a ComfyUI model."""

    def __init__(self, events: EventSink) -> None:
        self.events = events

    def run(self) -> dict[str, object]:
        report: dict[str, object] = {
            "torch_imported": False,
            "torch_version": None,
            "cuda_compiled": False,
            "cuda_runtime": None,
            "cuda_available": False,
            "device_count": 0,
            "devices": [],
            "nvidia_smi": self._nvidia_smi(),
            "success": False,
        }
        try:
            import torch

            report["torch_imported"] = True
            report["torch_version"] = str(torch.__version__)
            report["cuda_runtime"] = getattr(torch.version, "cuda", None)
            report["cuda_compiled"] = bool(report["cuda_runtime"])
            available = bool(torch.cuda.is_available())
            report["cuda_available"] = available
            if available:
                devices: list[GPUDevice] = []
                count = int(torch.cuda.device_count())
                for index in range(count):
                    properties = torch.cuda.get_device_properties(index)
                    capability = tuple(torch.cuda.get_device_capability(index))
                    devices.append(GPUDevice(
                        index=index,
                        name=str(properties.name),
                        total_memory_bytes=int(properties.total_memory),
                        capability=(int(capability[0]), int(capability[1])),
                    ))
                report["device_count"] = count
                report["devices"] = [asdict(device) for device in devices]
                # Minimal allocation verifies that the CUDA context can really execute.
                tensor = torch.zeros(1, device="cuda")
                report["allocation_test"] = bool(tensor.item() == 0)
                del tensor
                torch.cuda.synchronize()
            report["success"] = bool(report["cuda_available"] and report.get("allocation_test"))
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        self.events.emit("gpu.probe.completed", **report)
        return report

    @staticmethod
    def _nvidia_smi() -> dict[str, object]:
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
            return {
                "available": completed.returncode == 0,
                "return_code": completed.returncode,
                "output": (completed.stdout or completed.stderr or "").strip(),
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
