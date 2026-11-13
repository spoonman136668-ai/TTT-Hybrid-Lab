"""Bounded full-regression timeout must preserve partial diagnostic receipts.

These tests do not shorten, skip or substitute the final full regression.
"""
import subprocess

from scripts import qualify_cpu


def test_timeout_keeps_captured_pytest_output_and_budget(monkeypatch):
    def timed_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["python", "-m", "pytest", "tests"], timeout=2400,
            output=b"some test progress\n10 passed\n",
            stderr=b"partial stderr\n",
        )

    monkeypatch.setattr(qualify_cpu.subprocess, "run", timed_out)
    receipt = qualify_cpu.run_command(
        ["python", "-m", "pytest", "tests"], timeout=2400,
    )
    assert receipt["exit_code"] is None
    assert receipt["error_type"] == "TimeoutExpired"
    assert receipt["timeout_seconds"] == 2400
    assert "10 passed" in receipt["stdout_tail"]
    assert "partial stderr" in receipt["stderr_tail"]


def test_timeout_with_no_output_still_fails_closed(monkeypatch):
    def timed_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["pytest"], timeout=1800)

    monkeypatch.setattr(qualify_cpu.subprocess, "run", timed_out)
    receipt = qualify_cpu.run_command(["pytest"], timeout=1800)
    assert receipt["exit_code"] is None
    assert receipt["stdout_tail"] == ""
    assert receipt["stderr_tail"] == ""
