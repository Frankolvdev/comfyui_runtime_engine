from __future__ import annotations

import json
import subprocess


def _nvidia_smi() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return {
            "available": completed.returncode == 0,
            "return_code": completed.returncode,
            "output": completed.stdout.strip() or completed.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    report: dict[str, object] = {
        "torch_imported": False,
        "torch_version": None,
        "cuda_compiled": False,
        "cuda_runtime": None,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "allocation_test": False,
        "nvidia_smi": _nvidia_smi(),
        "success": False,
    }
    try:
        import torch

        report["torch_imported"] = True
        report["torch_version"] = torch.__version__
        report["cuda_runtime"] = getattr(torch.version, "cuda", None)
        report["cuda_compiled"] = bool(report["cuda_runtime"])
        report["cuda_available"] = bool(torch.cuda.is_available())

        if report["cuda_available"]:
            count = torch.cuda.device_count()
            report["device_count"] = count
            devices: list[dict[str, object]] = []
            for index in range(count):
                props = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "index": index,
                        "name": props.name,
                        "total_memory_bytes": props.total_memory,
                        "capability": list(torch.cuda.get_device_capability(index)),
                    }
                )
            report["devices"] = devices
            tensor = torch.ones((16, 16), device="cuda")
            report["allocation_test"] = bool(float(tensor.sum().item()) == 256.0)
            del tensor
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        report["success"] = bool(
            report["torch_imported"]
            and report["cuda_compiled"]
            and report["cuda_available"]
            and report["device_count"]
            and report["allocation_test"]
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
