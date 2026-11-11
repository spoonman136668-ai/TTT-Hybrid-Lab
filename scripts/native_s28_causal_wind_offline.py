"""S28 source-authenticated OFFLINE CartPoleWind boundary, engineering only.

NEVER imports, constructs, resets or steps a third-party scientific world.
No source download, package installation, training, model selection or
scientific first-use. The factual codec accepts *already produced* transitions
and intentionally discards all environment info, latent wind, graph or oracles.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "experiments" / "s28_offline_causal_wind_adapter_frozen.json"
SCHEMA = "ttt-hybrid.s28.offline-wind-adapter-evidence.v1"
OBS_DIM = 4
RUNTIME_CANDIDATES = ("gymnasium", "numpy", "networkx", "pydash", "toposort")
PROHIBITED_INFO = (
    "wind", "latent", "hidden", "graph", "counterfactual", "ctf",
    "oracle", "future", "unrealized", "intuit", "sample_u",
)


class S28Blocked(ValueError):
    pass


def require(ok: bool, why: str) -> None:
    if not ok:
        raise S28Blocked("S28_" + why)


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=True,
                       allow_nan=False, separators=(",", ":")) + "\n").encode("ascii")


def frozen() -> dict:
    d = json.loads(FROZEN.read_text(encoding="utf-8"))
    up = d["upstream"]
    a, q = d["adapter"], d["qualification"]
    require(
        d["schema"] == "ttt-hybrid.s28.offline-wind-source-adapter-feasibility.v1"
        and d["parent_main_sha"] == "6df5aa3f0dd5a99867098166aeed5b0fc70b4ec2"
        and up["repo"] == "CausalAILab/Causal-Gymnasium"
        and up["commit"] == "c94e9a4cd3641588b6b8507e71ccee5c27032bdf"
        and up["required_blobs"]["causal_gym/envs/cartpole_wind.py"]
            == "d65a6a31941709dc9201259d80dd4ebbcf48b4fa"
        and up["required_blobs"]["LICENSE"]
            == "993e3a4922f003dced0fd061735697c850ea7205"
        and up["recorded_license_marker"] == "@@ -154,49 +194,23 @@"
        and a["no_simulator_construction"] is True
        and a["no_upstream_module_import"] is True
        and a["no_package_install"] is True
        and a["no_network_access_from_adapter"] is True
        and a["no_science_world_reset_or_step"] is True
        and a["can_execute_gym_environment"] is False
        and q["science_first_use_authorized"] is False
        and q["owned_sealed_science_source_registered"] is False
        and q["upstream_source_license_approved"] is False,
        "PROTOCOL_OR_AUTHORITY_DRIFT")
    return d


def _git(*args: str, cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args], capture_output=True,
            text=True, check=False, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise S28Blocked("S28_OFFLINE_GIT_UNAVAILABLE") from e
    require(result.returncode == 0, "OFFLINE_GIT_CHECK_FAILURE")
    return result.stdout.strip()


def git_blob_sha(data: bytes) -> str:
    """Content-address bytes with Git's blob framing, no Git execution."""
    return hashlib.sha1(
        ("blob " + str(len(data)) + "\0").encode("ascii") + data
    ).hexdigest()


def audit_exact_upstream_checkout(source_root: Path, d: dict | None = None) -> dict:
    """READ ONLY: an already supplied detached pinned checkout; never clone."""
    d = frozen() if d is None else d
    root = Path(source_root)
    require(root.is_dir() and not root.is_symlink(), "UPSTREAM_CHECKOUT_MISSING")
    require(_git("rev-parse", "HEAD", cwd=root) == d["upstream"]["commit"],
            "UPSTREAM_COMMIT_NOT_PINNED")
    require(_git("status", "--porcelain", "--untracked-files=all", cwd=root) == "",
            "UPSTREAM_CHECKOUT_DIRTY")
    real_root = root.resolve(strict=True)
    blobs = {}
    for rel, expected in sorted(d["upstream"]["required_blobs"].items()):
        file = root / rel
        require(file.is_file() and not file.is_symlink(),
                "UPSTREAM_REQUIRED_SOURCE_MISSING_OR_LINK")
        resolved = file.resolve(strict=True)
        require(resolved.is_relative_to(real_root), "UPSTREAM_PATH_ESCAPE")
        # Symlinked parent directories would make a checkout mutable without
        # changing an expected file name; forbid them, too.
        part = root
        for component in Path(rel).parts:
            part = part / component
            require(not part.is_symlink(), "UPSTREAM_INTERMEDIATE_SYMLINK")
        blob = git_blob_sha(file.read_bytes())
        require(blob == expected, "UPSTREAM_SOURCE_BLOB_DRIFT:" + rel)
        blobs[rel] = blob

    source = (root / d["upstream"]["path"]).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=d["upstream"]["path"])
    classes = {
        n.name: n for n in tree.body if isinstance(n, ast.ClassDef)
    }
    require(set(d["upstream"]["expected_classes"]).issubset(classes),
            "UPSTREAM_SCM_PCH_CLASSES_MISSING")
    for name, methods in (("CartPoleWindSCM", ("reset", "step", "sample_u")),
                          ("CartPoleWindPCH", ("see", "do", "ctf_do"))):
        defined = {n.name for n in classes[name].body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        require(set(methods).issubset(defined),
                "UPSTREAM_PUBLIC_API_MISSING_" + name)
    imports = {
        alias.name for n in tree.body if isinstance(n, ast.Import)
        for alias in n.names
    }
    require({"numpy", "gymnasium"}.issubset(imports),
            "UPSTREAM_NUMERIC_RUNTIME_IMPORT_DRIFT")
    license_bytes = (root / "LICENSE").read_bytes()
    broken_license = (
        d["upstream"]["recorded_license_marker"].encode("utf-8")
        in license_bytes or len(license_bytes) < 10000
    )
    # The current upstream LICENSE is truncated/contains a literal diff hunk;
    # no assertion of redistribution rights or license-cleared packaging.
    require(broken_license, "UPSTREAM_LICENSE_NEEDS_NEW_INDEPENDENT_REVIEW")
    init_text = (root / "causal_gym/envs/__init__.py").read_text("utf-8")
    require("from .antmaze import" in init_text
            and "from .cartpole_wind import" in init_text,
            "UPSTREAM_PACKAGE_SIDE_EFFECTS_CHANGED")
    return {
        "status": "PASS_EXACT_UPSTREAM_SOURCE_STATIC_ONLY",
        "source_repo": d["upstream"]["repo"],
        "source_commit": d["upstream"]["commit"],
        "checked_required_git_blob_count": len(blobs),
        "actual_required_git_blobs": blobs,
        "source_ast_public_methods_verified": True,
        "upstream_package_initializer_eager_imports_other_worlds": True,
        "license_git_blob": blobs["LICENSE"],
        "license_sha256": hashlib.sha256(license_bytes).hexdigest(),
        "license_file_truncated_or_contains_diff_marker": broken_license,
        "license_or_redistribution_cleared": False,
        "upstream_runtime_imported_or_instantiated": False,
        "upstream_science_worlds_consumed": 0,
    }


def dependency_metadata_only() -> dict:
    """No import of dependency modules and no installer/network calls."""
    found = {}
    for name in RUNTIME_CANDIDATES:
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = None
        found[name] = version
    return {
        "status": "READ_ONLY_ENVIRONMENT_METADATA_NOT_RUNTIME_ADMISSION",
        "installed_distributions": found,
        "runtime_import_or_instantiation_attempted": False,
        "package_install_attempted": False,
        "runtime_ready": False,
        "reason": "MINIMAL_DEPENDENCY_TREE_AND_LICENSE_NOT_QUALIFIED",
    }


def _observation4(observation: Any) -> tuple[float, ...]:
    require(not isinstance(observation, (str, bytes, dict)),
            "OBSERVATION_NOT_NUMERIC_VECTOR")
    try:
        values = tuple(float(v) for v in observation)
    except (TypeError, ValueError, OverflowError) as e:
        raise S28Blocked("S28_OBSERVATION_NOT_NUMERIC_VECTOR") from e
    require(len(values) == OBS_DIM
            and all(math.isfinite(v) for v in values),
            "OBSERVATION_NONFINITE_OR_BAD_WIDTH")
    return values


def _info_untrusted(info: Any) -> None:
    # No info value is copied or used; block keys that name protected inputs
    # instead of silently accepting an accidental leak in source adapters.
    require(isinstance(info, Mapping), "ENV_INFO_NOT_MAPPING")
    for key in info:
        require(isinstance(key, str), "ENV_INFO_KEY_NOT_STRING")
        require(not any(s in key.lower() for s in PROHIBITED_INFO),
                "ENV_INFO_CONTAINS_PRIVILEGED_FIELD")


class FactualWindTransitionLedger:
    """Pure-state codec: caller supplies existing transitions; NO env API."""

    def __init__(self):
        self._current: tuple[float, ...] | None = None
        self._finished = False
        self._records: list[dict] = []

    def start(self, observation: Any, info: Any) -> tuple[float, ...]:
        require(self._current is None, "EPISODE_ALREADY_STARTED")
        _info_untrusted(info)
        self._current = _observation4(observation)
        self._finished = False
        return self._current

    def add_completed_transition(self, action: Any, transition: Any) -> dict:
        require(self._current is not None and not self._finished,
                "STEP_BEFORE_RESET_OR_AFTER_TERMINATION")
        require(type(action) is int and action in (0, 1),
                "ACTION_MUST_BE_EXECUTED_DISCRETE_0_OR_1")
        require(isinstance(transition, (tuple, list)) and len(transition) == 5,
                "SOURCE_STEP_SHAPE_MISMATCH")
        next_obs, reward, terminated, truncated, info = transition
        _info_untrusted(info)
        obs = _observation4(next_obs)
        require(type(reward) in (int, float) and math.isfinite(reward),
                "FACTUAL_REWARD_NONFINITE_OR_NOT_NUMERIC")
        require(type(terminated) is bool and type(truncated) is bool,
                "EPISODE_TERMINATION_NOT_BOOLEAN")
        result = {
            "previous_observation": list(self._current),
            "executed_action": action,
            "next_observation": list(obs),
            "factual_reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
        }
        self._records.append(result)
        self._current = obs
        self._finished = terminated or truncated
        return dict(result)

    def receipt(self) -> dict:
        require(self._current is not None, "NO_EPISODE_STARTED")
        return {
            "schema": "ttt-hybrid.s28.factual-transition-ledger.v1",
            "status": "ENGINEERING_FIXTURE_CODEC_ONLY",
            "events": len(self._records),
            "sha256": hashlib.sha256(canonical_json(self._records)).hexdigest(),
            "records": [dict(x) for x in self._records],
            "world_or_model_executed": False,
            "oracle_or_latent_information_used": False,
            "third_party_source_imported": False,
            "science_first_use": "NOT_RUN",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=None,
                        help="Optional already-checked-out exact upstream Git tree; never cloned")
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    d = frozen()
    if args.source_root is None:
        audit = {
            "status": "NOT_VERIFIED_NO_SUPPLIED_UPSTREAM_CHECKOUT",
            "source_qualified": False,
            "license_or_redistribution_cleared": False,
            "upstream_runtime_imported_or_instantiated": False,
        }
    else:
        audit = audit_exact_upstream_checkout(args.source_root, d)
    report = {
        "schema": SCHEMA,
        "status": "PASS_S28_OFFLINE_INTERFACE_AND_SOURCE_PREPARATION_ONLY",
        "frozen_protocol_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        "upstream": audit,
        "local_dependencies": dependency_metadata_only(),
        "data_interface": d["adapter"]["data_interface"],
        "no_execution_of_external_scm_or_science_first_use": True,
        "license_approval": "BLOCKED_UPSTREAM_LICENSE_INCOMPLETE",
        "ready_to_import_and_run_upstream": False,
        "scientific_first_use": "NOT_AUTHORIZED_NOT_RUN",
        "scientific_pass": False,
        "accepted_model_promoted": False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n",
                            encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "source_status": audit["status"],
        "third_party_license": report["license_approval"],
        "runtime_admitted": report["ready_to_import_and_run_upstream"],
        "science_first_use": report["scientific_first_use"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
