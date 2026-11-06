"""Official pinned TTT-Linear CPU admission and causal adaptation probe.

Uses the actual TTTLinear implementation, never a test double. Establishes
native forward/backward behavior and causal/reset/learning-rate controls only.
No learning, transfer, memory benefit, or scientific superiority is claimed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def fail(code: str) -> None:
    raise AssertionError(code)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ttt-root", required=True, type=Path)
    parser.add_argument("--receipt", type=Path,
                        default=ROOT / "local_runs" / "g5_official_ttt_cpu.json")
    args = parser.parse_args()
    args.receipt = args.receipt.resolve()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    raw = (ROOT / "source_pins.json").read_bytes()
    pins = json.loads(raw)
    pin = pins["sources"]["ttt_linear"]
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                       capture_output=True, text=True, check=False, timeout=15)
    observed_sha = r.stdout.strip() if r.returncode == 0 else None
    expected_sha = os.getenv("EXPECTED_CANDIDATE_SHA")
    identity_pass = bool(observed_sha) and (
        observed_sha == expected_sha if expected_sha else os.getenv("GITHUB_ACTIONS") != "true"
    )
    record = {
        "schema": "ttt-hybrid.official-ttt-cpu.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_sha": observed_sha,
        "expected_candidate_sha": expected_sha,
        "candidate_identity_pass": identity_pass,
        "source_pins_sha256": hashlib.sha256(raw).hexdigest(),
        "required_ttt_sha": pin["sha"],
        "platform": platform.platform(),
        "python": platform.python_version(),
        "qualification_kind": "ENGINEERING_NATIVE_TTT_CPU",
        "scientific_qualification": "UNVERIFIED",
        "authentic_mamba2_cuda": "NOT_RUN",
        "status": "BLOCKED_OR_FAIL",
    }
    start = time.perf_counter()
    try:
        if not identity_pass:
            fail("CANDIDATE_SHA_DRIFT")
        # Running scripts/... sets sys.path[0] to scripts, not the repo root.
        # Resolve both expected roots explicitly; source identity is still
        # enforced by Git SHA/blob and inspect.getfile below.
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(args.ttt_root.resolve()))
        from authentic_preflight import source_check, module_location_check
        source_result = source_check(args.ttt_root, pin)
        module_result = module_location_check("ttt", args.ttt_root)
        record["source_admission"] = source_result
        record["module_origin"] = module_result
        if not source_result["pass"] or not module_result["pass"]:
            fail("OFFICIAL_TTT_SOURCE_ADMISSION_BLOCKED")
        import torch
        import transformers
        from ttt import TTTConfig, TTTLinear
        actual = Path(inspect.getfile(TTTLinear)).resolve()
        expected = (args.ttt_root / "ttt.py").resolve()
        if actual != expected or Path(inspect.getfile(TTTConfig)).resolve() != expected:
            fail("TTT_CLASS_ORIGIN_MISMATCH")
        record["versions"] = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "torch_cuda": torch.version.cuda,
            "python": sys.version,
        }
        torch.set_num_threads(2)
        torch.manual_seed(20261105)
        cfg = TTTConfig(vocab_size=128, hidden_size=64,
                        intermediate_size=128, num_hidden_layers=1,
                        num_attention_heads=4, mini_batch_size=8,
                        ttt_layer_type="linear", ttt_base_lr=1.0,
                        pre_conv=False, use_cache=False,
                        max_position_embeddings=256)
        model = TTTLinear(cfg, layer_idx=0)
        model.eval()
        x = torch.randn(2, 16, 64)
        positions = torch.arange(16).unsqueeze(0).expand(2, -1)

        def forward(values):
            positions_here = torch.arange(values.shape[1]).unsqueeze(0).expand(values.shape[0], -1)
            return model(values, position_ids=positions_here, cache_params=None)

        with torch.no_grad():
            y1 = forward(x)
            y2 = forward(x)
            prefix = forward(x[:, :8])
            changed_x = x.clone()
            changed_x[:, 8:] += 7.0
            changed = forward(changed_x)
            cfg.ttt_base_lr = 0.0
            zero_lr = forward(x)
            cfg.ttt_base_lr = 1.0
            restored = forward(x)
        if list(y1.shape) != list(x.shape) or not torch.isfinite(y1).all().item():
            fail("NONFINITE_OR_WRONG_OUTPUT")
        if not torch.allclose(y1, y2, rtol=1e-5, atol=1e-5):
            fail("EPISODE_CARRYOVER_NEGATIVE_CONTROL")
        if not torch.allclose(y1[:, :8], changed[:, :8], rtol=1e-5, atol=1e-5):
            fail("FUTURE_SUFFIX_LEAKAGE")
        if not torch.allclose(y1[:, :8], prefix, rtol=1e-5, atol=1e-5):
            fail("PREFIX_INVARIANCE")
        if not torch.allclose(y1, restored, rtol=1e-5, atol=1e-5):
            fail("TTT_LR_RESTORE_CHANGED_EPISODE")
        lr_delta = float((y1 - zero_lr).abs().max())
        if not (lr_delta > 1e-7):
            fail("ZERO_LEARNING_RATE_NEGATIVE_CONTROL_NO_EFFECT")

        model.train()
        model.zero_grad(set_to_none=True)
        training_y = forward(x)
        loss = training_y.square().mean()
        if not torch.isfinite(loss).item():
            fail("NONFINITE_TRAINING_LOSS")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        if not gradients or any(g is not None and not torch.isfinite(g).all().item() for g in gradients):
            fail("NONFINITE_OR_MISSING_GRADIENTS")
        nonzero = sum(g is not None and torch.any(g.abs() > 0).item() for g in gradients)
        if nonzero == 0:
            fail("MISSING_NONZERO_GRADIENTS")
        record["metrics"] = {
            "forward_shape": list(y1.shape),
            "forward_finite": True,
            "repeat_forward_equal": True,
            "prefix_invariance": True,
            "future_suffix_negative_control_pass": True,
            "zero_learning_rate_negative_control_pass": True,
            "zero_learning_rate_max_abs_delta": lr_delta,
            "restored_episode_equal": True,
            "nonzero_gradient_parameter_tensors": nonzero,
            "trainable_parameter_tensors": len(gradients),
            "parameters": sum(p.numel() for p in model.parameters()),
            "training_loss": float(loss.detach()),
        }
        record["status"] = "PASS_AUTHENTIC_TTT_CPU_ENGINEERING_ONLY"
        return_code = 0
    except Exception as exc:
        record["status"] = "FAIL_OR_BLOCK_AUTHENTIC_TTT_CPU"
        record["error_type"] = type(exc).__name__
        record["error_message"] = str(exc)[:500]
        record["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
        return_code = 1
    finally:
        record["seconds"] = round(time.perf_counter() - start, 3)
        args.receipt.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": record["status"],
            "candidate_sha": record["candidate_sha"],
            "expected_sha": record["expected_candidate_sha"],
            "error": record.get("error_message"),
            "metrics": record.get("metrics"),
            "receipt": str(args.receipt),
        }))
    return return_code


if __name__ == "__main__":
    sys.exit(main())
