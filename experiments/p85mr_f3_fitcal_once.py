"""P85MR-F3: independent ONE-SHOT external FIT/CAL source collection.

The historical F1/F2 owner routes are spent. This is NOT a replay of either:
F3 has a new frozen authorization, independent owner identity, new immutable
ONCE marker and receipt path, and no F1/F2 owner or collection entrypoint use.
All external reset/do operations require trusted first-run owner custody.
Never invoked from normal CI or tests; no training, OOD or DEV admission.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

# Only stable, independently tested pure mechanics are reused.
import p85mr_fitcal_owner_firstuse as legacy

F3_BLOB = "92b0c9fc1b80dfcd8914928b83011f84473d9391"
PARENT_SHA = "b4c273251a3a8e24fb22ccd850579111e5a73527"
ORIGINAL_COLLECTOR_BLOB = "f5deeed08624d4dbe878a58e6869c9c4bb062ae3"
PROJECT = "spoonman136668-ai/TTT-Hybrid-Lab"
BRANCH = "research/p85mr-f3-repaired-fitcal-once-20261111"
WORKFLOW = ".github/workflows/p85mr-f3-repaired-fitcal-once.yml"
SOURCE_SHA = "c94e9a4cd3641588b6b8507e71ccee5c27032bdf"
RUNTIME_ALIAS = "p85mr-causalgym-cpu311-r3"
DEPS = {
    "numpy": "1.26.4", "gymnasium": "0.29.1",
    "networkx": "3.4.2", "pydash": "8.0.5", "toposort": "1.10",
    "cloudpickle": "3.0.0", "farama-notifications": "0.0.4",
    "typing-extensions": "4.12.2",
}
MAX_ACTIONS = 18432
MAX_SECONDS = 2400
COHORTS = (("fit", 128), ("cal", 64))


def locked_f3():
    raw = Path(__file__).with_name(
        "p85mr_f3_repaired_fitcal_authorization_freeze.json").read_bytes()
    if legacy.blob(raw) != F3_BLOB:
        raise ValueError("F3_FROZEN_AUTHORIZATION_BLOB_DRIFT")
    f = json.loads(raw)
    if (f["schema"] != "ttt-hybrid.p85mr-f3-independent-repaired-fitcal-source-firstuse-freeze.v1"
        or f["parent"]["sha"] != PARENT_SHA
        or f["parent"]["f2_original_collector_blob"] != ORIGINAL_COLLECTOR_BLOB
        or f["authority"]["effects_started"] is not False
        or f["authority"]["no_replay"] is not True
        or f["execution"]["branch"] != BRANCH
        or f["execution"]["workflow"] != WORKFLOW
        or f["execution"]["run_number"] != 1
        or f["execution"]["run_attempt"] != 1
        or f["runtime"]["path_alias"] != RUNTIME_ALIAS
        or f["runtime"]["package_pins"] != DEPS
        or f["source"]["sha"] != SOURCE_SHA
        or f["source"]["source_blobs"] != legacy.SOURCE_BLOBS
        or f["execution"]["max_actions"] != MAX_ACTIONS
        or f["execution"]["wall_seconds"] != MAX_SECONDS
        or f["scientific"]["cohorts"] != {
            "fit": {"n": 128, "base_seed": 1810000000},
            "cal": {"n": 64, "base_seed": 1810100000},
        }):
        raise ValueError("F3_FROZEN_PROTOCOL_DRIFT")
    if f["runtime"]["wheel_sha256"] != {
        "numpy": "666dbfb6ec68962c033a450943ded891bed2d54e6755e35e5835d63f4f6931d5",
        "gymnasium": "61c3384b5575985bb7f85e43213bcb40f36fcdff388cae6bc229304c71f2843e",
        "networkx": "df5d4365b724cf81b8c6a7312509d0c22386097011ad1abe274afd5e9d3bbc5f",
        "pydash": "b2625f8981862e19911daa07f80ed47b315ce20d9b5eb57aaf97aaf570c3892f",
        "toposort": "cbdbc0d0bee4d2695ab2ceec97fe0679e9c10eab4b2a87a9372b929e70563a87",
        "cloudpickle": "246ee7d0c295602a036e86369c77fecda4ab17b506496730f2f576d9016fd9c7",
        "farama-notifications": "14de931035a41961f7c056361dc7f980762a143d05791ef5794a751a2caf05ae",
        "typing-extensions": "04e5ca0351e0f3f85c6853954072df659d0d13fac324d0072316b67d7794700d",
    }:
        raise ValueError("F3_RUNTIME_WHEEL_PROVENANCE_DRIFT")
    return f


def owner_context(environ):
    expected = {
        "GITHUB_REPOSITORY": PROJECT,
        "GITHUB_ACTOR": "spoonman136668-ai",
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/" + BRANCH,
        "GITHUB_WORKFLOW_REF": PROJECT + "/" + WORKFLOW + "@refs/heads/" + BRANCH,
        "GITHUB_RUN_NUMBER": "1",
        "GITHUB_RUN_ATTEMPT": "1",
        "RUNNER_NAME": "TTT-HYBRID-GPU-WIN",
        "RUNNER_OS": "Windows",
        "RUNNER_ARCH": "X64",
        "P85MR_F3_OWNER_SCOPE": "FIT_CAL_ONLY",
    }
    if any(environ.get(key) != val for key, val in expected.items()):
        raise PermissionError("F3_EXACT_ONCE_OWNER_ROUTE_DENIED")
    run_id, sha = environ.get("GITHUB_RUN_ID", ""), environ.get("GITHUB_SHA", "")
    if not re.fullmatch(r"[0-9]{1,20}", run_id) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise PermissionError("F3_OWNER_RUN_OR_CANDIDATE_SHA_INVALID")
    return {
        "run_id": run_id, "project_sha": sha, "actor": "spoonman136668-ai",
        "runner": "TTT-HYBRID-GPU-WIN", "attempt": 1,
        "source_route": "P85MR-F3-INDEPENDENT-ONCE",
    }


def preflight(source_root, project_root, owner, freeze):
    # Existing source/freeze contracts are read-only provenance, not F1/F2 authority.
    scope = legacy.locked_scope()
    legacy.locked_route()
    legacy.locked_f2_route()
    if legacy.blob(Path(legacy.__file__).read_bytes()) != ORIGINAL_COLLECTOR_BLOB:
        raise ValueError("F3_ORIGINAL_COLLECTOR_CHANGED")
    seed_check = legacy.preflight_partition()
    if seed_check["cross_split_collisions"] != 0 or seed_check["new_unique_seeds"] != 448:
        raise ValueError("F3_SOURCE_SEED_OVERLAP")
    if sys.version_info[:2] != (3, 11) or os.getuid() != 1000:
        raise ValueError("F3_WSL_OWNER_PY311_REQUIRED")
    if (os.name != "posix" or not hasattr(signal, "SIGALRM")
            or "microsoft" not in os.uname().release.lower()):
        raise PermissionError("F3_NOT_APPROVED_WSL_ENVIRONMENT")
    for name, version in DEPS.items():
        if importlib.metadata.version(name) != version:
            raise ValueError("F3_RUNTIME_DEPENDENCY_MISMATCH")
    for module_name in ("numpy", "gymnasium", "networkx", "pydash", "toposort"):
        importlib.import_module(module_name)
    src = Path(source_root).resolve(strict=True)
    project = Path(project_root).resolve(strict=True)
    for root, sha, origins in (
        (src, SOURCE_SHA, (
            "https://github.com/CausalAILab/Causal-Gymnasium.git",
            "https://github.com/CausalAILab/Causal-Gymnasium",
        )),
        (project, owner["project_sha"], (
            "https://github.com/" + PROJECT + ".git",
            "https://github.com/" + PROJECT,
        )),
    ):
        if legacy.git(root, "rev-parse", "HEAD") != sha:
            raise ValueError("F3_CHECKOUT_SHA_MISMATCH")
        if legacy.git(root, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError("F3_DIRTY_CHECKOUT")
        if legacy.git(root, "remote", "get-url", "origin") not in origins:
            raise ValueError("F3_REMOTE_ORIGIN_MISMATCH")
    for relative_path, git_blob in freeze["source"]["source_blobs"].items():
        path = src / relative_path
        if not path.is_file() or legacy.blob(path.read_bytes()) != git_blob:
            raise ValueError("F3_UPSTREAM_SOURCE_BLOB_MISMATCH")
    if any(scope["cohorts"][split]["n"] != count for split, count in COHORTS):
        raise ValueError("F3_FROZEN_COHORT_MISMATCH")
    if (freeze["parent"]["r3_owner_run"] != 43054709366 or
            freeze["parent"]["r3_artifact_sha256"] !=
            "1240602593c84280c7c8b47255a1648eecbf0f32defdad7398afe228caf35b42"):
        raise ValueError("F3_R3_RUNTIME_CUSTODY_DRIFT")
    return src, project, scope


def collect_one_shot(*, source_root, project_root, environ=None):
    freeze = locked_f3()
    owner = owner_context(os.environ if environ is None else environ)
    src, project, scope = preflight(source_root, project_root, owner, freeze)
    # Original F1/F2 collection functions are NEVER called. Their helpers
    # implement previously-qualified factual reset/do and fsynced journaling.
    base = project / "evidence" / "p85mr_fitcal_f3"
    base.mkdir(parents=True, exist_ok=True)
    attempt = base / "ONCE"
    attempt.mkdir(mode=0o700, exist_ok=False)
    legacy._put_new(attempt / "ATTEMPT.json", {
        "status": "F3_ATTEMPTED_NO_REPLAY", "owner_context": owner,
        "scope_sha": F3_BLOB, "upstream_sha": SOURCE_SHA,
        "cohorts": ["fit", "cal"], "max_actions": MAX_ACTIONS,
        "ood_opened": False, "science_pass": False,
    })
    start = time.monotonic()
    previous_signal = signal.getsignal(signal.SIGALRM)
    def expired(_signum, _frame):
        raise TimeoutError("F3_BOUNDED_WALL_CLOCK_EXPIRED")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(MAX_SECONDS)
    total = 0
    summaries = {}
    try:
        sys.dont_write_bytecode = True
        factory, task = legacy._load_scoped_source(src)
        for split, n in COHORTS:
            regime = scope["cohorts"][split]
            accepted, short = [], []
            for index in range(n):
                world = legacy.observed_episode(
                    factory, task, split, index, regime, owner["run_id"],
                    start + MAX_SECONDS,
                    journal=lambda event, s=split: legacy._append_progress(
                        attempt / (s + ".steps.ndjson"), event))
                # Record complete episode before any next potentially effectful reset.
                legacy._append_progress(attempt / (split + ".ndjson"), world)
                total += len(world["transitions"])
                if total > MAX_ACTIONS:
                    raise ValueError("F3_MAX_ACTIONS_EXCEEDED")
                (short if len(world["transitions"]) < 2 else accepted).append(
                    index if len(world["transitions"]) < 2 else world)
            if len(short) / n > 0.15 or not accepted:
                raise ValueError("F3_INSUFFICIENT_VALID_EPISODES")
            capture = {
                "schema": "ttt-hybrid.p80mr.cartpole-do-capture.v1",
                "upstream_repo": legacy.UPSTREAM_REPO,
                "upstream_commit": legacy.UPSTREAM_SHA,
                "upstream_path": legacy.UPSTREAM_PATH,
                "upstream_blob": legacy.UPSTREAM_BLOB,
                "source_id": f"p85mr-f3-{split}-{owner['run_id']}",
                "producer_id": "ttt-hybrid-f3-owner",
                "custody_ref": f"ghrun:{owner['run_id']}",
                "worlds": accepted,
            }
            if legacy.adapt_capture(capture)["custody"]["scientific_pass"] is not False:
                raise ValueError("F3_UNQUALIFIED_SOURCE_ADMISSION_BYPASS")
            digest = legacy._put_new(attempt / (split + ".json"), capture)
            summaries[split] = {
                "scheduled": n, "valid": len(accepted),
                "short_episode_indices": short, "sha256": digest,
            }
        result = {
            "schema": "ttt-hybrid.p85mr-f3-fitcal-unqualified-source-receipt.v1",
            "status": "SOURCE_COLLECTED_INDEPENDENT_WITNESS_PENDING",
            "owner": owner, "upstream_sha": SOURCE_SHA,
            "cohorts": summaries, "executed_actions": total,
            "wall_seconds": time.monotonic() - start, "ood_created": False,
            "external_dev_created": False, "trained_baselines": False,
            "science_pass": False, "original_f1_f2_runs_replayed": False,
        }
        legacy._put_new(attempt / "RESULT.json", result)
        return result
    except BaseException as exc:
        legacy._put_new(attempt / "FAILED.json", {
            "status": "F3_ABORTED_NO_REPLAY", "failure_type": type(exc).__name__,
            "actions_so_far": total, "science_pass": False, "ood_created": False,
        })
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_signal)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkout", required=True)
    parser.add_argument("--project-checkout", required=True)
    parser.add_argument("--approved-once", action="store_true")
    args = parser.parse_args()
    if not args.approved_once:
        raise PermissionError("F3_EFFECTFUL_FLAG_REQUIRED")
    collect_one_shot(source_root=args.source_checkout,
                     project_root=args.project_checkout)


if __name__ == "__main__":
    main()
