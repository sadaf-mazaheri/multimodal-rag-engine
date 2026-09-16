"""Background CPU, RAM and GPU sampling for load tests.

Every dependency here is optional and imported defensively: ``psutil`` is not a
declared project dependency, and most machines running this benchmark have no
GPU. A missing probe yields ``available: False`` with a reason, never an error,
so a load test never fails because a resource could not be observed.

CPU readings are reported two ways, because ``psutil.Process.cpu_percent`` is
per core and exceeds 100 on a multi-core process: ``process_cpu_percent`` is
normalised to the whole machine (0-100) and ``process_cpu_percent_raw`` keeps
psutil's value.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any

from mmrag.production.stats import summarize

_UNSET: Any = object()


def import_psutil() -> Any | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil


def gpu_snapshot() -> tuple[dict[str, Any] | None, str | None]:
    """Current GPU memory (and utilisation when pynvml is present), or a reason."""
    # torch is only consulted if something already imported it: the sampler must
    # not load a multi-hundred-megabyte framework just to learn there is no GPU.
    torch = sys.modules.get("torch")
    if torch is None:
        return None, "torch not loaded in this process"
    try:
        if not torch.cuda.is_available():
            return None, "no CUDA device available to torch"
        device = torch.cuda.current_device()
        snap: dict[str, Any] = {
            "device": torch.cuda.get_device_name(device),
            "memory_allocated_mb": torch.cuda.memory_allocated(device) / 2**20,
            "max_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
        }
    except Exception as exc:  # pragma: no cover - driver-specific
        return None, f"torch.cuda probe failed: {type(exc).__name__}"
    try:  # pragma: no cover - needs an NVIDIA driver
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        snap["utilization_percent"] = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except Exception:
        pass
    return snap, None


class ResourceSampler:
    """Samples resource usage on a daemon thread between ``start`` and ``stop``."""

    def __init__(
        self,
        interval_s: float = 0.5,
        *,
        psutil_module: Any = _UNSET,
        gpu_probe: Any = gpu_snapshot,
    ):
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.interval_s = interval_s
        self._psutil = import_psutil() if psutil_module is _UNSET else psutil_module
        self._gpu_probe = gpu_probe
        self._samples: list[dict[str, float]] = []
        self._gpu: list[dict[str, Any]] = []
        self._gpu_reason: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: Any = None
        self._cores = os.cpu_count() or 1

    def _sample(self) -> None:
        if self._process is not None:
            raw = float(self._process.cpu_percent(None))
            self._samples.append({
                "process_cpu_percent_raw": raw,
                "process_cpu_percent": raw / self._cores,
                "process_rss_mb": self._process.memory_info().rss / 2**20,
                "system_cpu_percent": float(self._psutil.cpu_percent(None)),
                "system_memory_percent": float(self._psutil.virtual_memory().percent),
            })
        if self._gpu_probe is not None:
            snap, reason = self._gpu_probe()
            if snap is not None:
                self._gpu.append(snap)
            else:
                self._gpu_reason = reason

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def start(self) -> ResourceSampler:
        if self._psutil is not None:
            self._process = self._psutil.Process()
            # The first cpu_percent(None) call only primes the counters.
            self._process.cpu_percent(None)
            self._psutil.cpu_percent(None)
        self._thread = threading.Thread(target=self._loop, name="resource-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()  # one closing sample, so a short level still has data
        return self.summary()

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"interval_s": self.interval_s, "n_samples": len(self._samples)}
        if self._psutil is None:
            out.update(available=False, reason="psutil not installed")
        else:
            out["available"] = True
            for key in ("process_cpu_percent", "process_cpu_percent_raw", "process_rss_mb",
                        "system_cpu_percent", "system_memory_percent"):
                s = summarize(sample[key] for sample in self._samples)
                out[key] = {"mean": s["mean"], "max": s["max"]}
            out["cpu_logical_cores"] = self._cores
        if self._gpu:
            out["gpu"] = {
                "available": True,
                "device": self._gpu[-1].get("device"),
                "max_memory_allocated_mb": round(
                    max(g["max_memory_allocated_mb"] for g in self._gpu), 1),
                "utilization_percent": summarize(
                    g["utilization_percent"] for g in self._gpu if "utilization_percent" in g),
            }
        else:
            out["gpu"] = {"available": False, "reason": self._gpu_reason or "not sampled"}
        return out


def environment_resources() -> dict[str, Any]:
    """Static machine facts recorded once per load run."""
    psutil = import_psutil()
    torch = sys.modules.get("torch")
    info: dict[str, Any] = {
        "cpu_logical_cores": os.cpu_count(),
        "cpu_physical_cores": psutil.cpu_count(logical=False) if psutil else None,
        "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 1) if psutil else None,
        "psutil_available": psutil is not None,
        "torch_num_threads": None,
        "cuda_available": None,
    }
    if torch is not None:
        try:
            info["torch_num_threads"] = torch.get_num_threads()
            info["cuda_available"] = bool(torch.cuda.is_available())
        except Exception:  # pragma: no cover - defensive
            pass
    return info
