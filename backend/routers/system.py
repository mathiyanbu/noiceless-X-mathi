"""
System telemetry endpoint: CPU utilization per core, CPU temperature, and memory.
"""

import os
from fastapi import APIRouter
from backend.schemas import SystemResponse, CpuMetrics, MemoryMetrics
from backend.ipc_client import ipc_client, RuntimeUnavailableError
from ai.runtime.realtime_pipeline import SystemMetrics

router = APIRouter(prefix="/api", tags=["system"])
_local_metrics = SystemMetrics()


def _read_memory_metrics() -> MemoryMetrics:
    meminfo_path = "/proc/meminfo"
    if os.path.exists(meminfo_path):
        try:
            total_kb = 0
            avail_kb = 0
            with open(meminfo_path, "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_kb = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        avail_kb = int(line.split()[1])
            if total_kb > 0:
                total_mb = round(total_kb / 1024.0, 1)
                free_mb = round(avail_kb / 1024.0, 1)
                used_mb = round(total_mb - free_mb, 1)
                return MemoryMetrics(total_mb=total_mb, used_mb=used_mb, free_mb=free_mb)
        except Exception:
            pass

    # Host fallback approximation (e.g. during development outside Pi)
    return MemoryMetrics(total_mb=4096.0, used_mb=1280.0, free_mb=2816.0)


@router.get("/system", response_model=SystemResponse)
async def get_system_metrics():
    """
    Returns real hardware metrics: CPU core %, temperature (°C), and memory.
    Reads from the C++ runtime over IPC if connected, or directly from the Linux kernel (/proc, sysfs).
    """
    # 1. Try reading from C++ runtime over IPC
    try:
        res = ipc_client.send_command("get_system")
        if res.get("status") == "ok":
            cpu_data = res.get("cpu", {})
            return SystemResponse(
                cpu=CpuMetrics(
                    overall_pct=float(cpu_data.get("overall_pct", 0.0)),
                    cores_pct=[float(x) for x in cpu_data.get("cores_pct", [])]
                ),
                temperature_c=float(res.get("temperature_c", 0.0)),
                temperature_available=bool(res.get("temperature_available", False)),
                memory=_read_memory_metrics(),
                source="c++_runtime"
            )
    except RuntimeUnavailableError:
        pass

    # 2. Read directly from local system metrics (/proc/stat, /sys/class/thermal)
    _local_metrics.sample()
    return SystemResponse(
        cpu=CpuMetrics(
            overall_pct=_local_metrics.overall_cpu,
            cores_pct=_local_metrics.per_core_cpu
        ),
        temperature_c=_local_metrics.temperature_c,
        temperature_available=_local_metrics.temp_available,
        memory=_read_memory_metrics(),
        source="host_kernel"
    )
