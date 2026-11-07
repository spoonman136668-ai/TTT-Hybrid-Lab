"""G7-S1 preregistered native online-TTT learning versus static matched control.

Only run under the already-qualified frozen R3 venv, with ephemeral signed
Python headers provided by the G7-R35-derived admission wrapper. This is a
bounded *experiment*, not a deployment or claim of learning superiority.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import g7_r5_oneshot as custody

class Blocked(Exception):
    pass

ACTIVE_PHASE = "NOT_STARTED"

def mark_phase(value):
    global ACTIVE_PHASE
    ACTIVE_PHASE = value


def require(condition, reason):
    if not condition:
        raise Blocked(reason)


def episode(seed, batch=2):
    """Four new within-episode bindings, then eight keyed causal queries.

    Values for queries are *only* in preceding support tokens; the query
    never contains its requested value. Fresh seeds are disjoint for FIT/TEST.
    """
    import torch
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    k = (torch.randint(0, 2, (batch, 4, 16),
                       generator=generator, dtype=torch.int64).float() * 2 - 1)
    v = (torch.randint(0, 2, (batch, 4, 16),
                       generator=generator, dtype=torch.int64).float() * 2 - 1)
    x = torch.zeros(batch, 16, 64, dtype=torch.float32)
    shuffled = torch.zeros_like(x)
    order = (0, 1, 2, 3, 3, 2, 1, 0)
    for i in range(4):
        x[:, 2*i, :16] = k[:, i]
        x[:, 2*i+1, 16:32] = v[:, i]
        shuffled[:, 2*i, :16] = k[:, i]
        shuffled[:, 2*i+1, 16:32] = v[:, (i+1) % 4]
        for t in (x, shuffled):
            t[:, 2*i, 32] = 1
            t[:, 2*i+1, 33] = 1
    for j, idx in enumerate(order):
        for t in (x, shuffled):
            t[:, 8+j, :16] = k[:, idx]
            t[:, 8+j, 34] = 1
    y = torch.stack([v[:, i] for i in order], dim=1)
    return x, y, shuffled


def snapshot_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8") + b"\0")
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def evaluate(model, seed, count, *, device):
    import torch
    normal, shuffled = [], []
    model.eval()
    with torch.no_grad():
        for index in range(count):
            # Entirely new association per heldout episode, never FIT.
            x, target, altered = episode(seed*100000 + 900000 + index)
            x = x.to(device)
            target = target.to(device)
            altered = altered.to(device)
            prediction = model(x)[:, 8:, :16]
            wrong = model(altered)[:, 8:, :16]
            require(bool(torch.isfinite(prediction).all()) and
                    bool(torch.isfinite(wrong).all()), "NONFINITE_EVAL")
            normal.append(float((prediction - target).square().mean().item()))
            shuffled.append(float((wrong - target).square().mean().item()))
    return normal, shuffled



def evaluate_support_mechanism(model, seed, count, *, device):
    """Fresh heldout: oracle and zero controls plus causal support interventions.

    Caller has already trained the two original frozen arms; this adds no
    optimizer steps, gradients, source changes, or training-data exposure.
    """
    import torch
    metrics = {
        "zero_baseline_mse": [],
        "oracle_correct_mse": [],
        "oracle_shuffled_mse": [],
        "model_correct_mse": [],
        "model_shuffled_mse": [],
        "model_erased_mse": [],
        "shuffled_prediction_movement_mse": [],
        "erased_prediction_movement_mse": [],
    }
    model.eval()
    with torch.no_grad():
        for index in range(count):
            x, target, shuffled = episode(seed*100000 + 950000 + index)
            erased = x.clone()
            erased[:, 1:8:2, 16:32] = 0
            keys = x[:, 0:8:2, :16]
            queries = x[:, 8:, :16]
            matches = (queries[:, :, None, :] ==
                       keys[:, None, :, :]).all(dim=-1)
            require(bool((matches.sum(dim=-1) == 1).all()),
                    "ORACLE_KEY_MATCH_NOT_UNIQUE")
            require(bool((x[:, 8:, 16:32] == 0).all()) and
                    bool((shuffled[:, 8:, :] == x[:, 8:, :]).all()) and
                    bool((erased[:, 8:, :] == x[:, 8:, :]).all()),
                    "QUERY_VALUE_LEAK_OR_INTERVENTION_DRIFT")
            values = x[:, 1:8:2, 16:32]
            wrong_values = shuffled[:, 1:8:2, 16:32]
            lookup = matches.float()
            oracle = torch.einsum("bqi,biv->bqv", lookup, values)
            wrong_oracle = torch.einsum("bqi,biv->bqv", lookup, wrong_values)
            require(bool(torch.equal(oracle, target)),
                    "ORACLE_RETRIEVAL_OR_TARGET_BROKEN")
            require(bool((erased[:, 1:8:2, 16:32] == 0).all())
                    and bool((erased[:, 0:8:2, :] == x[:, 0:8:2, :]).all()),
                    "ERASED_SUPPORT_CONTROL_INVALID")
            x, target, shuffled, erased = (
                x.to(device), target.to(device),
                shuffled.to(device), erased.to(device)
            )
            correct = model(x)[:, 8:, :16]
            altered = model(shuffled)[:, 8:, :16]
            removed = model(erased)[:, 8:, :16]
            require(bool(torch.isfinite(correct).all())
                    and bool(torch.isfinite(altered).all())
                    and bool(torch.isfinite(removed).all()),
                    "NONFINITE_INTERVENTION_OUTPUT")
            def append(label, value):
                metrics[label].append(float(value))
            append("zero_baseline_mse", float(target.square().mean().item()))
            append("oracle_correct_mse", 0.0)
            append("oracle_shuffled_mse",
                   float((wrong_oracle - target.cpu()).square().mean().item()))
            append("model_correct_mse", (correct - target).square().mean().item())
            append("model_shuffled_mse", (altered - target).square().mean().item())
            append("model_erased_mse", (removed - target).square().mean().item())
            append("shuffled_prediction_movement_mse",
                   (correct - altered).square().mean().item())
            append("erased_prediction_movement_mse",
                   (correct - removed).square().mean().item())
    return {name: statistics.mean(values) for name, values in metrics.items()}



def estimate_bootstrap(base, online, n, seed):
    rng = random.Random(seed)
    ratios = []
    require(len(base) == len(online) and len(base) > 0, "BOOTSTRAP_PAIRING")
    for _ in range(n):
        indexes = [rng.randrange(len(base)) for _ in base]
        b = statistics.mean(base[i] for i in indexes)
        a = statistics.mean(online[i] for i in indexes)
        require(b > 0 and math.isfinite(b+a), "BOOTSTRAP_NUMERIC")
        ratios.append((b-a)/b)
    ratios.sort()
    lo = ratios[int(0.025*len(ratios))]
    hi = ratios[int(0.975*len(ratios))-1]
    return [lo, hi]


def run(args):
    mark_phase("IMPORT_TORCH")
    import torch
    from authentic_hybrid import AuthenticHybrid, load_official_classes

    mark_phase("LOAD_FROZEN_PROTOCOL")
    protocol_bytes = args.protocol.read_bytes()
    spec = json.loads(protocol_bytes)
    require(spec.get("schema") == "ttt-hybrid.g7-s1.pre_registered_association.v1",
            "PROTOCOL_SCHEMA_DRIFT")
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    diagnostic = None
    diagnostic_sha = None
    if args.diagnostic_protocol:
        require(args.diagnostic_protocol.resolve() ==
                (ROOT / "experiments/g7_s4_frozen_mechanism.json").resolve(),
                "S4_DIAGNOSTIC_PROTOCOL_PATH_DENIED")
        raw_diagnostic = args.diagnostic_protocol.read_bytes()
        diagnostic_sha = hashlib.sha256(raw_diagnostic).hexdigest()
        diagnostic = json.loads(raw_diagnostic)
        require(diagnostic["schema"] == "ttt-hybrid.g7-s4.support-causal-diagnostic.v1"
                and diagnostic["predecessor"]["run"] == 42365966599
                and diagnostic["predecessor"]["frozen_s1_protocol_sha256"] ==
                protocol_sha
                and diagnostic["design"]["seeds"] == [101,211,307]
                and diagnostic["design"]["train_steps_per_arm"] == 8
                and diagnostic["design"]["novel_episodes_per_seed"] == 6
                and diagnostic["thresholds"]["zero_baseline_mse"] == 1.0
                and diagnostic["thresholds"]["oracle_shuffled_mse_min"] == 1.5,
                "S4_PRE_REGISTERED_DIAGNOSTIC_DRIFT")
    train = spec["training"]
    require(train["seeds"] == [101, 211, 307] and train["steps"] == 8
            and train["batch_size"] == 2 and train["sequence_length"] == 16
            and spec["episode"]["heldout_episodes_per_seed"] == 6,
            "FROZEN_STUDY_PARAMETERS_DRIFT")
    require([a["ttt_base_lr"] for a in spec["arms"]] == [1.0, 0.0],
            "ARM_IDENTITY_DRIFT")
    require(spec["threshold"]["bootstrap_seed"] == 9821 and
            spec["threshold"]["bootstrap_resamples"] == 1000 and
            spec["threshold"]["mean_relative_reduction_min"] == 0.05,
            "THRESHOLD_DRIFT")
    require(os.geteuid() == 1000 and "microsoft" in
            Path("/proc/sys/kernel/osrelease").read_text().lower(),
            "GPU_WSL_NORMAL_USER_REQUIRED")
    require(custody.git(ROOT, "rev-parse", "HEAD") == args.expected_sha and
            not custody.git(ROOT, "status", "--porcelain", "--untracked-files=all"),
            "SOURCE_SHA_OR_TREE_DRIFT")
    require(custody.sha256(ROOT / "source_pins.json") == custody.R3_PINS_SHA256,
            "SOURCE_PINS_DRIFT")
    mark_phase("VERIFY_PINNED_SOURCE")
    root = (Path.home().resolve() / ".local/share/ttt-hybrid" /
            ("g7-r3-" + custody.R3_SOURCE_SHA[:12]))
    manifest_hash, packages = custody.manifest()
    require(manifest_hash == custody.R3_MANIFEST_SHA256
            and packages.get("torch") == custody.TORCH
            and packages.get("triton") == custody.TRITON, "RUNTIME_PACKAGE_DRIFT")
    pins = json.loads((ROOT / "source_pins.json").read_text())
    for label in ("mamba2", "ttt_linear"):
        custody.source_custody(root / "sources" / label, pins["sources"][label])
    # G7-S2: the strict -I child suppresses PYTHONPATH; explicitly import
    # only the two authenticated official source roots, exactly as the
    # qualified G7-R6 native child does. This does not change science arms.
    sys.path[:0] = [str(root / "sources/mamba2"),
                    str(root / "sources/ttt_linear")]
    require(sys.version_info[:2] == (3,12)
            and torch.__version__ == custody.TORCH
            and torch.version.cuda == "12.8"
            and torch.cuda.is_available()
            and torch.cuda.device_count() == 1
            and tuple(torch.cuda.get_device_capability(0)) == (12,0),
            "CUDA_OR_TORCH_IDENTITY_DRIFT")
    device = torch.device("cuda:0")
    free, _ = torch.cuda.mem_get_info()
    require(free >= spec["budget"]["gpu_free_bytes_min"], "FREE_VRAM_ADMISSION")
    mark_phase("LOAD_OFFICIAL_CLASSES")
    classes = load_official_classes(root / "sources/mamba2",
                                    root / "sources/ttt_linear")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    peak_all = 0
    reports = []
    for seed in train["seeds"]:
        arms = {}
        initial_hashes = []
        for arm in spec["arms"]:
            mark_phase("INITIALIZE_FULL_MODEL")
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = AuthenticHybrid(
                official_classes=classes, width=64, heads=4,
                mini_batch=8, mamba_mem_eff_path=False,
                use_ttt=True, use_depth=True,
            )
            # Frozen deterministic route, preserving same architecture/params.
            with torch.no_grad():
                model.depth.router.weight.zero_()
                model.depth.router.bias.fill_(1.0)
            for p in model.depth.router.parameters():
                p.requires_grad_(False)
            initial_hashes.append(snapshot_digest(model))
            model.ttt.config.ttt_base_lr = arm["ttt_base_lr"]
            model = model.to(device=device, dtype=torch.float32)
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters()
                            if p.requires_grad)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=train["learning_rate"], weight_decay=train["weight_decay"]
            )
            torch.cuda.reset_peak_memory_stats(device)
            losses = []
            model.train()
            for step in range(train["steps"]):
                x, target, _ = episode(seed*100000 + step*100)
                x = x.to(device)
                target = target.to(device)
                optimizer.zero_grad(set_to_none=True)
                mark_phase("TRAIN_FORWARD")
                prediction = model(x)[:, 8:, :16]
                require(prediction.shape == target.shape, "BAD_TRAIN_SHAPE")
                loss = (prediction - target).square().mean()
                require(bool(torch.isfinite(loss)), "NONFINITE_TRAIN_LOSS")
                mark_phase("TRAIN_BACKWARD")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    train["gradient_clipping"],
                )
                require(bool(torch.isfinite(norm)), "NONFINITE_GRADIENT")
                mark_phase("TRAIN_OPTIMIZER")
                optimizer.step()
                losses.append(float(loss.detach().item()))
                torch.cuda.synchronize(device)
                peak = torch.cuda.max_memory_allocated(device)
                require(peak <= spec["budget"]["gpu_torch_peak_bytes_max"],
                        "GPU_PEAK_VRAM_BUDGET_EXCEEDED")
                require(time.monotonic()-started < spec["budget"]["total_seconds_max"],
                        "EXPERIMENT_TIME_BUDGET")
            mark_phase("EVALUATE_HELDOUT")
            heldout, negative = evaluate(
                model, seed, spec["episode"]["heldout_episodes_per_seed"],
                device=device,
            )
            if diagnostic is not None:
                mark_phase("S4_FRESH_SUPPORT_CAUSAL_DIAGNOSTIC")
                new_measures = evaluate_support_mechanism(
                    model, seed,
                    diagnostic["design"]["novel_episodes_per_seed"],
                    device=device)
            else:
                new_measures = None
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            peak_all = max(peak_all, int(peak))
            arms[arm["id"]] = {
                "model_parameters": total,
                "trainable_parameters": trainable,
                "optimizer_steps": len(losses),
                "loss_first": losses[0], "loss_final": losses[-1],
                "heldout_episode_mse": heldout,
                "shuffled_support_episode_mse": negative,
                "support_mechanism_diagnostic": new_measures,
                "peak_torch_allocated_bytes": int(peak),
            }
            del model, optimizer
            torch.cuda.empty_cache()
        require(initial_hashes[0] == initial_hashes[1],
                "INITIAL_WEIGHTS_NOT_MATCHED")
        require(arms["online_ttt"]["model_parameters"] ==
                arms["static_matched"]["model_parameters"] and
                arms["online_ttt"]["trainable_parameters"] ==
                arms["static_matched"]["trainable_parameters"] and
                arms["online_ttt"]["optimizer_steps"] ==
                arms["static_matched"]["optimizer_steps"] == train["steps"],
                "ARM_BUDGET_MISMATCH")
        reports.append({"seed": seed,
                        "initial_weights_sha256": initial_hashes[0],
                        "arms": arms})
    mark_phase("AGGREGATE_HELDOUT_RESULTS")
    base = [v for item in reports
            for v in item["arms"]["static_matched"]["heldout_episode_mse"]]
    online = [v for item in reports
              for v in item["arms"]["online_ttt"]["heldout_episode_mse"]]
    online_shuffled = [v for item in reports
                       for v in item["arms"]["online_ttt"]["shuffled_support_episode_mse"]]
    base_mse, online_mse, shuffled_mse = map(statistics.mean,
                                              (base, online, online_shuffled))
    relative = (base_mse-online_mse)/base_mse
    shuffle_harm = (shuffled_mse-online_mse)/online_mse
    per_seed = []
    for item in reports:
        b = statistics.mean(item["arms"]["static_matched"]["heldout_episode_mse"])
        a = statistics.mean(item["arms"]["online_ttt"]["heldout_episode_mse"])
        per_seed.append((b-a)/b)
    interval = estimate_bootstrap(base, online,
                                  spec["threshold"]["bootstrap_resamples"],
                                  spec["threshold"]["bootstrap_seed"])
    threshold = spec["threshold"]
    prereg_met = (
        relative >= threshold["mean_relative_reduction_min"]
        and all(v > 0 for v in per_seed)
        and interval[0] > 0
        and shuffle_harm >= threshold["shuffled_support_relative_loss_increase_min"]
    )
    support_causal = None
    if diagnostic is not None:
        mark_phase("S4_AGGREGATE_SUPPORT_CAUSAL_RESULTS")
        d = {arm: {} for arm in ("online_ttt", "static_matched")}
        for arm in d:
            for metric in reports[0]["arms"][arm]["support_mechanism_diagnostic"]:
                d[arm][metric] = statistics.mean(
                    item["arms"][arm]["support_mechanism_diagnostic"][metric]
                    for item in reports)
        online_d, static_d = d["online_ttt"], d["static_matched"]
        controls_pass = (
            all(abs(d[arm]["zero_baseline_mse"] - 1.0) < 1e-12 and
                d[arm]["oracle_correct_mse"] <=
                diagnostic["thresholds"]["oracle_correct_mse_max"] and
                d[arm]["oracle_shuffled_mse"] >=
                diagnostic["thresholds"]["oracle_shuffled_mse_min"]
                for arm in d)
        )
        zero_gain = ((1.0 - online_d["model_correct_mse"]) / 1.0)
        online_shuffle_harm = (
            (online_d["model_shuffled_mse"]-online_d["model_correct_mse"]) /
            online_d["model_correct_mse"])
        online_erase_harm = (
            (online_d["model_erased_mse"]-online_d["model_correct_mse"]) /
            online_d["model_correct_mse"])
        per_seed_new = [
            (1.0-item["arms"]["online_ttt"]["support_mechanism_diagnostic"]
             ["model_correct_mse"])
            for item in reports
        ]
        thresholds_pass = (
            controls_pass and
            zero_gain >= diagnostic["thresholds"]["model_relative_mse_below_zero_min"] and
            online_shuffle_harm >= diagnostic["thresholds"]["online_shuffled_relative_harm_min"] and
            online_erase_harm >= diagnostic["thresholds"]["online_erased_relative_harm_min"] and
            all(v > 0 for v in per_seed_new)
        )
        support_causal = {
            "protocol_sha256": diagnostic_sha,
            "status": ("ASSAY_INVALID" if not controls_pass else
                       "DIAGNOSTIC_SIGNAL_REQUIRES_INDEPENDENT_REPLICATION"
                       if thresholds_pass else
                       "ASSAY_VALID_BINDING_NOT_DEMONSTRATED"),
            "oracle_and_zero_controls_pass": controls_pass,
            "all_frozen_mechanism_thresholds_met": thresholds_pass,
            "online_vs_zero_relative_gain": zero_gain,
            "online_shuffle_relative_harm": online_shuffle_harm,
            "online_erase_relative_harm": online_erase_harm,
            "per_seed_online_zero_gains": per_seed_new,
            "arm_metrics": d,
            "scientific_acceptance": "NOT_AUTHORIZED",
        }
    return {
        "schema": "ttt-hybrid.g7-s1.authentic-native-pilot.v1",
        "status": ("PRELIMINARY_POSITIVE_REQUIRES_INDEPENDENT_REPLICATION"
                   if prereg_met else "NULL_OR_NEGATIVE"),
        "candidate_sha": args.expected_sha,
        "frozen_protocol_sha256": protocol_sha,
        "source_pins_sha256": custody.R3_PINS_SHA256,
        "r3_package_manifest_sha256": custody.R3_MANIFEST_SHA256,
        "task": "heldout_episodic_key_value_binding",
        "seeds": train["seeds"],
        "arms": reports,
        "support_causal_diagnostic": support_causal,
        "primary": {
            "static_mse": base_mse,
            "online_mse": online_mse,
            "mean_paired_relative_reduction": relative,
            "paired_bootstrap_95pct_interval": interval,
            "all_seed_relative_reductions": per_seed,
            "online_shuffled_support_mse": shuffled_mse,
            "shuffled_support_relative_harm": shuffle_harm,
            "thresholds_met": prereg_met,
        },
        "gpu_peak_torch_allocated_bytes": peak_all,
        "elapsed_seconds": round(time.monotonic()-started, 3),
        "independent_qualification": "NOT_RUN",
        "deployment": "NOT_AUTHORIZED",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expected-sha", required=True)
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--diagnostic-protocol", type=Path)
    p.add_argument("--receipt", required=True, type=Path)
    a = p.parse_args()
    report = {
        "schema": "ttt-hybrid.g7-s1.authentic-native-pilot.v1",
        "candidate_sha": a.expected_sha,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ENGINEERING_BLOCKED_UNVERIFIED",
        "independent_qualification": "NOT_RUN",
        "deployment": "NOT_AUTHORIZED",
    }
    code = 2
    try:
        require(a.receipt.resolve().parent == (ROOT / "local_runs").resolve()
                and a.receipt.name == "g7_s1_native_science.json"
                and a.protocol.resolve() ==
                (ROOT / "experiments/g7_s1_frozen_protocol.json").resolve(),
                "RECEIPT_OR_PROTOCOL_LOCATION_DENIED")
        report = run(a)
        code = 0
    except (Blocked, custody.Blocked) as exc:
        report["status"] = "ENGINEERING_BLOCKED_" + str(exc)
        report["failure_phase"] = ACTIVE_PHASE
    except Exception as exc:
        report["status"] = "ENGINEERING_BLOCKED_" + type(exc).__name__
        report["error_class"] = type(exc).__name__
        report["failure_phase"] = ACTIVE_PHASE
        raw = str(exc).lower()[:5000]
        rules = {
            "CUBLAS_WORKSPACE_DETERMINISM": "cublas_workspace_config" in raw,
            "TRITON_KERNEL_COMPATIBILITY": "triton" in raw or "ptxas" in raw,
            "AUTOGRAD_INPLACE_MODIFICATION": "inplace" in raw or "in-place" in raw,
            "CUDA_OUT_OF_MEMORY": "out of memory" in raw or "cublas_status_alloc_failed" in raw,
            "NOT_IMPLEMENTED_AUTOGRAD": "derivative for" in raw or "not implemented" in raw,
            "DEVICE_SIDE_ASSERT": "device-side assert" in raw,
            "NATIVE_LIBRARY_MISSING": "cannot open shared object" in raw,
        }
        report["error_categories"] = sorted(k for k, matched in rules.items() if matched) or ["OTHER_RUNTIME"]
        report["error_text_sha256"] = hashlib.sha256(str(exc).encode("utf-8")).hexdigest()
        # No raw tracebacks, user paths, commands or environment contents exported.
    finally:
        if (a.receipt.resolve().parent == (ROOT / "local_runs").resolve()
                and a.receipt.name == "g7_s1_native_science.json"):
            a.receipt.parent.mkdir(parents=True, exist_ok=True)
            a.receipt.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                                 encoding="utf-8")
        print("G7_S1_SCIENCE=" + report["status"], flush=True)
        print("G7_S1_INDEPENDENT=NOT_RUN", flush=True)
        print("G7_S1_DEPLOYMENT=NOT_AUTHORIZED", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
