"""Synthetic, non-scientific tests for OS memory metric custody."""
import ctypes
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from g7_process_peak_memory import (
    MemoryProbeBlocked, _ProcessMemoryCounters, read_peak_memory,
)


class _Fn:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class _Dll:
    def __init__(self, **fns):
        for name, callback in fns.items():
            setattr(self, name, _Fn(callback))


class _RUsage:
    ru_maxrss = 2048


class _Resource:
    RUSAGE_SELF = 0

    @staticmethod
    def getrusage(which):
        assert which == 0
        return _RUsage()


class PeakMemoryTests(unittest.TestCase):
    def test_linux_exact_original_multiplier_and_metric(self):
        result = read_peak_memory(platform_name="linux", unix_resource=_Resource)
        self.assertEqual(result.peak_bytes, 2048 * 1024)
        self.assertEqual(result.metric, "linux_ru_maxrss_bytes")
        self.assertTrue(result.comparable_to_linux_ru_maxrss)

    def test_darwin_does_not_multiply_by_1024(self):
        result = read_peak_memory(platform_name="darwin", unix_resource=_Resource)
        self.assertEqual(result.peak_bytes, 2048)
        self.assertFalse(result.comparable_to_linux_ru_maxrss)

    def test_windows_peak_working_set_distinct_from_unix_rss(self):
        observed = {}

        def get_info(handle, ptr, size):
            observed.update(handle=handle, size=size)
            assert size == ctypes.sizeof(_ProcessMemoryCounters)
            counters = ctypes.cast(ptr, ctypes.POINTER(_ProcessMemoryCounters)).contents
            assert counters.cb == size
            counters.PeakWorkingSetSize = 17_123_456
            counters.WorkingSetSize = 4_000_000
            return 1

        result = read_peak_memory(
            platform_name="win32",
            kernel32=_Dll(GetCurrentProcess=lambda: 123),
            psapi=_Dll(GetProcessMemoryInfo=get_info),
        )
        self.assertEqual(result.peak_bytes, 17_123_456)
        self.assertEqual(observed["handle"], 123)
        self.assertEqual(result.metric, "windows_psapi_peak_working_set_bytes")
        self.assertFalse(result.comparable_to_linux_ru_maxrss)

    def test_windows_failed_api_and_invalid_zero_peak_fail_closed(self):
        for result_code, peak in ((0, 8_000_000), (1, 0)):
            def get_info(_handle, ptr, _size):
                counters = ctypes.cast(ptr, ctypes.POINTER(_ProcessMemoryCounters)).contents
                counters.PeakWorkingSetSize = peak
                return result_code

            with self.subTest(code=result_code, peak=peak):
                with self.assertRaises(MemoryProbeBlocked):
                    read_peak_memory(platform_name="win32",
                                     kernel32=_Dll(GetCurrentProcess=lambda: 1),
                                     psapi=_Dll(GetProcessMemoryInfo=get_info))

    def test_unknown_platform_and_bad_unix_reading_fail_closed(self):
        with self.assertRaisesRegex(MemoryProbeBlocked, "UNSUPPORTED_MEMORY_PLATFORM"):
            read_peak_memory(platform_name="aix")
        class EmptyResource:
            RUSAGE_SELF = 0
            @staticmethod
            def getrusage(_):
                return type("R", (), {"ru_maxrss": 0})()
        with self.assertRaisesRegex(MemoryProbeBlocked, "UNIX_RUSAGE_INVALID"):
            read_peak_memory(platform_name="linux", unix_resource=EmptyResource)


if __name__ == "__main__":
    unittest.main()
