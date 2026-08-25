from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class HardwareProfile:
    cpu_model: str
    ram_gb: float
    gpu_model: str
    gpu_vram_gb: float
    gpu_acceleration_available: bool
    torch_cuda_available: bool
    safe_model_size_gb: float
    max_tokens_estimate: int
    recommended_context_window: int
    os_name: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CudaMemorySnapshot:
    device_index: int
    device_name: str
    total_gb: float
    free_gb: float
    used_gb: float
    source: str
    device_count: int = 1

    @property
    def free_fraction(self) -> float:
        if self.total_gb <= 0:
            return 0.0
        return max(0.0, min(1.0, self.free_gb / self.total_gb))


def _run_command(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _detect_cpu_model() -> str:
    if os.name == "nt":
        output = _run_command(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
            ]
        )
        if output:
            return output.splitlines()[0].strip()

    fallback = platform.processor().strip()
    if fallback:
        return fallback

    uname = platform.uname()
    return f"{uname.system} {uname.machine}".strip()


def _detect_ram_gb() -> float:
    try:
        import psutil  # type: ignore

        total_bytes = psutil.virtual_memory().total
        return round(total_bytes / (1024**3), 1)
    except Exception:
        pass

    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32),
                    ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return round(status.ullTotalPhys / (1024**3), 1)
        except Exception:
            return 8.0

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round((pages * page_size) / (1024**3), 1)
    except Exception:
        return 8.0


def _aggregate_cuda_memory(
    snapshots: list[CudaMemorySnapshot],
    source: str,
) -> Optional[CudaMemorySnapshot]:
    if not snapshots:
        return None
    if len(snapshots) == 1:
        return snapshots[0]

    names = ", ".join(item.device_name for item in snapshots)
    return CudaMemorySnapshot(
        device_index=-1,
        device_name=f"{len(snapshots)} CUDA GPUs ({names})",
        total_gb=round(sum(item.total_gb for item in snapshots), 3),
        free_gb=round(sum(item.free_gb for item in snapshots), 3),
        used_gb=round(sum(item.used_gb for item in snapshots), 3),
        source=source,
        device_count=len(snapshots),
    )


def _detect_nvidia_cuda_memory() -> Optional[CudaMemorySnapshot]:
    output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return None

    snapshots: list[CudaMemorySnapshot] = []
    for raw_line in output.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) < 5:
            continue
        try:
            device_index = int(parts[0])
            total_gb = float(parts[2]) / 1024.0
            free_gb = float(parts[3]) / 1024.0
            used_gb = float(parts[4]) / 1024.0
        except (TypeError, ValueError):
            continue
        if total_gb <= 0:
            continue
        snapshots.append(
            CudaMemorySnapshot(
                device_index=device_index,
                device_name=parts[1] or f"CUDA GPU {device_index}",
                total_gb=round(total_gb, 3),
                free_gb=round(max(0.0, free_gb), 3),
                used_gb=round(max(0.0, used_gb), 3),
                source="nvidia-smi",
            )
        )

    return _aggregate_cuda_memory(snapshots, source="nvidia-smi")


def _detect_torch_cuda_memory() -> Optional[CudaMemorySnapshot]:
    try:
        import torch  # type: ignore
    except Exception:
        return None

    if not torch.cuda.is_available():
        return None

    snapshots: list[CudaMemorySnapshot] = []
    try:
        device_count = int(torch.cuda.device_count())
    except Exception:
        device_count = 1

    for device_index in range(max(1, device_count)):
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            device_name = str(torch.cuda.get_device_name(device_index))
            total_gb = float(total_bytes) / (1024**3)
            free_gb = float(free_bytes) / (1024**3)
        except Exception:
            continue
        if total_gb <= 0:
            continue
        snapshots.append(
            CudaMemorySnapshot(
                device_index=device_index,
                device_name=device_name or f"CUDA GPU {device_index}",
                total_gb=round(total_gb, 3),
                free_gb=round(max(0.0, free_gb), 3),
                used_gb=round(max(0.0, total_gb - free_gb), 3),
                source="torch.cuda",
            )
        )

    return _aggregate_cuda_memory(snapshots, source="torch.cuda")


def detect_cuda_memory_snapshot() -> Optional[CudaMemorySnapshot]:
    """Return aggregate live memory telemetry for all CUDA devices visible to the runtime."""
    return _detect_nvidia_cuda_memory() or _detect_torch_cuda_memory()


def _detect_nvidia_gpu() -> tuple[str, float]:
    snapshot = _detect_nvidia_cuda_memory()
    if snapshot is None:
        return "", 0.0

    return snapshot.device_name, round(snapshot.total_gb, 1)


def _detect_torch_cuda() -> tuple[bool, str, float]:
    try:
        import torch  # type: ignore
    except Exception:
        return False, "", 0.0

    if not torch.cuda.is_available():
        return False, "", 0.0

    try:
        device_name = torch.cuda.get_device_name(0)
        properties = torch.cuda.get_device_properties(0)
        vram_gb = round(properties.total_memory / (1024**3), 1)
        return True, str(device_name), float(vram_gb)
    except Exception:
        return True, "CUDA GPU", 0.0


def _estimate_limits(ram_gb: float, gpu_vram_gb: float) -> tuple[float, int, int]:
    gpu_mode = gpu_vram_gb > 0
    if gpu_mode:
        safe_model = max(2.0, round(gpu_vram_gb * 0.78, 1))
        max_tokens = int(max(4096, min(131072, (ram_gb * 1100) + (gpu_vram_gb * 850))))
        recommended_ctx = int(min(max_tokens, max(8192, min(32768, gpu_vram_gb * 2048))))
    else:
        safe_model = max(2.0, round(ram_gb * 0.22, 1))
        max_tokens = int(max(2048, min(32768, ram_gb * 700)))
        recommended_ctx = int(min(max_tokens, max(4096, ram_gb * 320)))

    return safe_model, max_tokens, recommended_ctx


def detect_hardware_profile() -> HardwareProfile:
    cpu_model = _detect_cpu_model()
    ram_gb = _detect_ram_gb()

    nvidia_name, nvidia_vram_gb = _detect_nvidia_gpu()
    torch_cuda_available, torch_gpu_name, torch_vram_gb = _detect_torch_cuda()

    gpu_model = nvidia_name or torch_gpu_name
    gpu_vram_gb = nvidia_vram_gb if nvidia_vram_gb > 0 else torch_vram_gb
    gpu_acceleration = bool(gpu_model) and (gpu_vram_gb > 0 or torch_cuda_available)

    safe_model_size_gb, max_tokens, recommended_ctx = _estimate_limits(ram_gb, gpu_vram_gb)

    return HardwareProfile(
        cpu_model=cpu_model or "Unknown CPU",
        ram_gb=ram_gb,
        gpu_model=gpu_model or "No dedicated GPU detected",
        gpu_vram_gb=gpu_vram_gb,
        gpu_acceleration_available=gpu_acceleration,
        torch_cuda_available=torch_cuda_available,
        safe_model_size_gb=safe_model_size_gb,
        max_tokens_estimate=max_tokens,
        recommended_context_window=recommended_ctx,
        os_name=f"{platform.system()} {platform.release()}",
    )
