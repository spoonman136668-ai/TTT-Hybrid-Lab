"""Engineering-only process peak memory probe; no scientific acceptance.

Unix ru_maxrss and Windows peak working set are related but not interchangeable
for frozen scientific thresholds. Consumers must compare only matching metrics.
No third-party packages, networking, file writes, or research execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import ctypes
import sys


@dataclass(frozen=True)
class PeakMemory:
    peak_bytes: int
    metric: str
    comparable_to_linux_ru_maxrss: bool


class MemoryProbeBlocked(RuntimeError):
    """Fail closed on an unsupported or invalid operating-system reading."""


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _windows_peak_working_set(*, kernel32=None, psapi=None) -> PeakMemory:
    if kernel32 is None or psapi is None:
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
        except (AttributeError, OSError) as exc:
            raise MemoryProbeBlocked("WINDOWS_PSAPI_UNAVAILABLE") from exc
    try:
        get_current = kernel32.GetCurrentProcess
        get_current.restype = ctypes.c_void_p
        get_info = psapi.GetProcessMemoryInfo
        get_info.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            ctypes.c_ulong,
        )
        get_info.restype = ctypes.c_int
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not get_info(get_current(), ctypes.byref(counters), counters.cb):
            raise MemoryProbeBlocked("WINDOWS_PSAPI_QUERY_FAILED")
        peak = int(counters.PeakWorkingSetSize)
    except (AttributeError, OSError, TypeError) as exc:
        raise MemoryProbeBlocked("WINDOWS_PSAPI_QUERY_FAILED") from exc
    if peak <= 0:
        raise MemoryProbeBlocked("WINDOWS_PEAK_WORKING_SET_INVALID")
    return PeakMemory(peak, "windows_psapi_peak_working_set_bytes", False)


def read_peak_memory(*, platform_name: str | None = None,
                     unix_resource=None, kernel32=None, psapi=None) -> PeakMemory:
    """Read an OS-native peak, but never normalize unlike metrics into science."""
    platform_name = sys.platform if platform_name is None else platform_name
    if platform_name in ("linux", "darwin"):
        if unix_resource is None:
            try:
                import resource as unix_resource
            except ImportError as exc:
                raise MemoryProbeBlocked("UNIX_RUSAGE_UNAVAILABLE") from exc
        try:
            raw = int(unix_resource.getrusage(unix_resource.RUSAGE_SELF).ru_maxrss)
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            raise MemoryProbeBlocked("UNIX_RUSAGE_QUERY_FAILED") from exc
        multiplier = 1024 if platform_name == "linux" else 1
        amount = raw * multiplier
        if amount <= 0:
            raise MemoryProbeBlocked("UNIX_RUSAGE_INVALID")
        return PeakMemory(amount,
                          "linux_ru_maxrss_bytes" if platform_name == "linux"
                          else "darwin_ru_maxrss_bytes",
                          platform_name == "linux")
    if platform_name == "win32":
        return _windows_peak_working_set(kernel32=kernel32, psapi=psapi)
    raise MemoryProbeBlocked("UNSUPPORTED_MEMORY_PLATFORM")
