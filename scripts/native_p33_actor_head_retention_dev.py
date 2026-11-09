"""P33: frozen fresh-seed reward-only PPO comparison and immutable DEV weight custody.

Reuses P32's exact source trainer through a short-lived scoped configuration;
does not rerun P32's consumed data or change P20/P30/P32 code.
Project-owner qualified public development only; not protected scientific
first use, model promotion, production, or a new scheduler.
"""
from __future__ import annotations

from contextlib import contextmanager
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from scripts import native_p20_reward_only_ppo_dev as p20
from scripts import native_p30_reward_motion_ppo_dev as p30
from scripts import native_p32_component_damage_dev as p32

SCHEMA = "ttt-hybrid.native-p33.actor-head-retention-result.v1"
CHECKPOINT_SCHEMA = "ttt-hybrid.native-p33.immutable-actor-weights.v1"
PROTOCOL = ROOT / "experiments/native_p33_actor_head_retention_frozen.json"
SEEDS = (4831, 4861, 4903)
TRAIN_BASE = 160000
EVAL_SEEDS = (161011, 161019, 161029, 161037, 161049, 161059, 161071, 161089)
CONDITIONS = ("original", "short_pole_x0.7", "long_pole_x1.3")
TRAIN_ARMS = ("full_actor_recurrent_ppo", "actor_head_only_ppo",
              "critic_only_frozen_policy")
EVAL_ARMS = ("initial_no_update",) + TRAIN_ARMS + (
    "finite_difference_public_two_frame", "seeded_random")
PARENT = "7bc49166b83788efc7bfb9da82d39fc420630c46"
LOCKS = {
    "scripts/native_p20_reward_only_ppo_dev.py":
        "cd9507b3cd86bca0d1507b1e234dcaaf5919e8f4",
    "scripts/native_p30_reward_motion_ppo_dev.py":
        "6d864593732cda1413a6338bc2e42ad9c6a85629",
    "scripts/native_p32_component_damage_dev.py":
        "793ac4153353754c0754c76655663af74cb9c0a5",
}


def freeze_contract() -> dict:
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    t, e, s = p["training"], p["evaluation"], p["source"]
    if (p["schema"] !=
        "ttt-hybrid.native-p33.actor-head-retention-and-checkpoint-frozen.v1"
        or p["phase"] != "PROSPECTIVE_FROZEN_BEFORE_IMPLEMENTATION_OR_PUBLIC_SOURCE_USE"
        or p["parent"]["exact_source_sha"] != PARENT
        or p["parent"]["scientific_P32_broad_gate_passed"] is not False
        or s["protected_scientific_first_use"] is not False
        or s["unchanged_p20_script_git_blob"] != LOCKS[
            "scripts/native_p20_reward_only_ppo_dev.py"]
        or s["unchanged_p30_script_git_blob"] != LOCKS[
            "scripts/native_p30_reward_motion_ppo_dev.py"]
        or s["unchanged_p32_script_git_blob"] != LOCKS[
            "scripts/native_p32_component_damage_dev.py"]
        or t["model_initialization_seeds"] != list(SEEDS)
        or t["original_source_train_reset_base"] != TRAIN_BASE
        or t["arms"] != list(TRAIN_ARMS)
        or t["source_actions_per_arm_seed"] != 49152
        or t["source_optimizer_minibatches_per_arm_seed"] != 576
        or t["planned_total_source_training_actions"] != 442368
        or t["planned_total_optimizer_minibatches"] != 5184
        or e["original_reset_seeds"] != list(EVAL_SEEDS)
        or e["physics_conditions"] != list(CONDITIONS)
        or e["arms"] != list(EVAL_ARMS)
        or p["qualification"]["first_use_reserved_by_project_owner_before_start"] is not True):
        raise RuntimeError("P33_FROZEN_PROTOCOL_DRIFT")
    return p


def _git(*args: str) -> str:
    raw = subprocess.check_output(
        ["git", "-C", str(ROOT), *args], stderr=subprocess.DEVNULL,
        timeout=15)
    return raw.decode("utf-8").strip()


def qualify_local_source() -> str:
    """Source assertion only, never a replacement for project-owner authority."""
    if os.environ.get("NATIVE_P33_DEV_ONLY") != "1":
        raise RuntimeError("P33_PUBLIC_DEV_OPT_IN_REQUIRED")
    freeze_contract()
    p32.frozen()  # Verify old P32 frozen original BEFORE modifying globals.
    sha = _git("rev-parse", "HEAD")
    if (not re.fullmatch("[a-f0-9]{40}", sha)
        or (os.environ.get("GITHUB_SHA") and os.environ["GITHUB_SHA"] != sha)
        or _git("status", "--porcelain", "--untracked-files=all")):
        raise RuntimeError("P33_DIRTY_OR_DRIFTED_CANDIDATE")
    subprocess.check_call(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", PARENT, sha],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
    for name, required in LOCKS.items():
        if _git("hash-object", name) != required:
            raise RuntimeError("P33_PROTECTED_PARENT_BLOB_CHANGED:" + name)
    return sha


@contextmanager
def scoped_fresh_source():
    """Reuse the exact P32 training functions, restore every global on error."""
    prior = (p32.SEEDS, p32.TRAIN_BASE, p32.EVAL_SEEDS,
             p32.TRAIN_ARMS, p32.EVAL_ARMS)
    try:
        p32.SEEDS = SEEDS
        p32.TRAIN_BASE = TRAIN_BASE
        p32.EVAL_SEEDS = EVAL_SEEDS
        p32.TRAIN_ARMS = TRAIN_ARMS
        p32.EVAL_ARMS = EVAL_ARMS
        with p32.scoped_p30_seeds():
            yield
    finally:
        (p32.SEEDS, p32.TRAIN_BASE, p32.EVAL_SEEDS,
         p32.TRAIN_ARMS, p32.EVAL_ARMS) = prior


def _policy_sentinel(seed: int):
    return p32.motion([p20.factual_row([.02, .03, .04, .05], 0, 0.)], seed)


def publish_unaccepted_weights(root: Path, seed: int, model,
                               source_sha: str, train: dict, means: dict) -> dict:
    """Write immutable exclusive candidate bytes; never edit champion pointers."""
    if (seed not in SEEDS or not re.fullmatch("[a-f0-9]{40}", source_sha)
        or train.get("arm") != "actor_head_only_ppo"
        or train.get("model_seed") != seed
        or train.get("cold_exact_logits_and_values") is not True
        or train.get("real_original_train_source_actions") != 49152
        or train.get("original_ppo_minibatch_updates") != 576):
        raise RuntimeError("P33_CANDIDATE_SOURCE_OR_TRAINING_DRIFT")
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("P33_CANDIDATE_ROOT_UNTRUSTED")
    if (root / "accepted").exists() or (root / "latest").exists():
        raise RuntimeError("P33_ACCEPTED_POINTER_FORBIDDEN")
    for key in ("original", "short_pole_x0.7", "long_pole_x1.3"):
        if (key not in means or "actor_head_only_ppo" not in means[key]
            or not isinstance(means[key]["actor_head_only_ppo"], (float, int))):
            raise RuntimeError("P33_HELDOUT_EVIDENCE_MISSING")
    raw = io.BytesIO()
    torch.save(model.state_dict(), raw)
    data = raw.getvalue()
    sha256 = hashlib.sha256(data).hexdigest()
    cold = p30.RewardGRU10()
    cold.load_state_dict(torch.load(
        io.BytesIO(data), weights_only=True, map_location="cpu"), strict=True)
    model.eval()
    cold.eval()
    with torch.no_grad():
        original = model(_policy_sentinel(seed))
        loaded = cold(_policy_sentinel(seed))
    if not all(torch.equal(x, y) for x, y in zip(original, loaded)):
        raise RuntimeError("P33_MODEL_COLD_LOGITS_DRIFT")
    if p20.model_sha(model) != p20.model_sha(cold):
        raise RuntimeError("P33_MODEL_COLD_WEIGHTS_DRIFT")
    weight_name = "p33-actor-head-seed-" + str(seed) + ".pt"
    metadata_name = weight_name + ".manifest.json"
    manifest = {
        "schema": CHECKPOINT_SCHEMA,
        "candidate_source_sha": source_sha,
        "frozen_protocol_path": "experiments/native_p33_actor_head_retention_frozen.json",
        "parent_source_sha": PARENT,
        "model_seed": seed,
        "train_arm": "actor_head_only_ppo",
        "weights_filename": weight_name,
        "weights_sha256": sha256,
        "source_trained_model_sha256": p20.model_sha(cold),
        "real_source_action_trace_sha256":
            train["original_actual_action_consequence_sha256"],
        "original_heldout_reward_means": means,
        "cold_weights_and_logits_bit_exact": True,
        "checkpoint_optimizer_included": False,
        "cold_AdamW_continuation_proven": False,
        "accepted_pointer_modified": False,
        "model_promoted": False,
        "protected_scientific_first_use": False,
        "source_data_used_to_train": "PUBLIC_ORIGINAL_DEV",
    }
    # Never overwrite / replay an already owned candidate, including one
    # where only the weight file survived an interrupted write.
    with (root / weight_name).open("xb") as f:
        f.write(data)
    with (root / metadata_name).open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest


def frozen_development_outcome(records: list[dict]) -> dict:
    if len(records) != len(SEEDS) or [x["model_seed"] for x in records] != list(SEEDS):
        raise RuntimeError("P33_MISSING_OR_DUPLICATE_MODEL_SEED")
    decisions = []
    for row in records:
        means = row["real_source_reward_means_by_condition"]
        good = (
            means["original"]["actor_head_only_ppo"] >= 150
            and all(means[c]["actor_head_only_ppo"] >= 100 and
                    means[c]["actor_head_only_ppo"] >=
                        means[c]["full_actor_recurrent_ppo"] + 25
                    for c in CONDITIONS)
            and row["critic_only_source_actions_equal_initial"] is True)
        decisions.append(good)
    return {
        "per_fresh_seed_frozen_gates": decisions,
        "all_three_seed_actor_only_frozen_gate": all(decisions),
        "strong_conventional_control_reported": all(
            all("finite_difference_public_two_frame" in
                row["real_source_reward_means_by_condition"][c]
                for c in CONDITIONS)
            for row in records),
        "scientific_architecture_superiority": False,
        "protected_scientific_first_use": "NOT_RUN",
    }


def run(s29_root: Path, source_root: Path, cloudpickle_root: Path,
        candidate_root: Path) -> dict:
    contract = freeze_contract()
    source = qualify_local_source()
    candidate_root = Path(candidate_root)
    if candidate_root.exists() or candidate_root.is_symlink():
        raise RuntimeError("P33_CANDIDATE_ROOT_ALREADY_EXISTS_NO_REPLAY")
    if any(candidate_root.resolve().is_relative_to(p.resolve())
           for p in (ROOT, Path(s29_root), Path(source_root),
                     Path(cloudpickle_root))):
        raise RuntimeError("P33_OUTPUT_INSIDE_SOURCE_DENIED")
    began = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    gym, source_admission = p20.admit_gymnasium(
        s29_root, source_root, cloudpickle_root)
    candidate_root.mkdir(parents=True, exist_ok=False)
    records = []
    with scoped_fresh_source():
        for seed in SEEDS:
            initial = p32.make_initial(seed)
            baseline = {condition: [
                p32.evaluate_episode(gym, initial, world, condition, seed,
                                     "initial_no_update")
                for world in EVAL_SEEDS] for condition in CONDITIONS}
            models = {}
            trained = {}
            original_init = p20.model_sha(initial)
            for arm in TRAIN_ARMS:
                model, reference, proof = p32.train_arm(gym, seed, arm)
                if (p20.model_sha(reference) != original_init or
                    proof["initial_source_actor_weights_sha256"] != original_init):
                    raise RuntimeError("P33_TRAINING_INITIAL_STATE_DIVERGED")
                models[arm], trained[arm] = model, proof
            if len({v["first_512_real_source_actions_trace_sha256"]
                    for v in trained.values()}) != 1:
                raise RuntimeError("P33_INITIAL_FACTUAL_ACTIONS_NOT_MATCHED")
            if (trained["actor_head_only_ppo"][
                  "components_bit_exact_unchanged_from_original"]["recurrent"] is not True
                or not all(trained["critic_only_frozen_policy"][
                    "components_bit_exact_unchanged_from_original"][n]
                    for n in ("recurrent", "actor"))):
                raise RuntimeError("P33_FROZEN_COMPONENTS_WERE_MODIFIED")
            evals = {}
            critic_equal = True
            for condition in CONDITIONS:
                eps = []
                for world in EVAL_SEEDS:
                    items = {}
                    for arm in EVAL_ARMS:
                        model = (initial if arm == "initial_no_update"
                                 else models.get(arm))
                        items[arm] = p32.evaluate_episode(
                            gym, model, world, condition, seed, arm)
                    if len({tuple(x["source_original_initial_observation"])
                            for x in items.values()}) != 1:
                        raise RuntimeError("P33_PAIRED_WORLD_INITIAL_DRIFT")
                    critic_equal = critic_equal and (
                        items["initial_no_update"][
                            "source_real_action_consequence_sha256"] ==
                        items["critic_only_frozen_policy"][
                            "source_real_action_consequence_sha256"])
                    eps.append(items)
                evals[condition] = eps
            if not critic_equal:
                raise RuntimeError("P33_CRITIC_ONLY_ACTIONS_CHANGED")
            means = {c: {
                arm: sum(e[arm]["actual_source_reward_return"]
                         for e in evals[c]) / len(EVAL_SEEDS)
                for arm in EVAL_ARMS} for c in CONDITIONS}
            candidate = publish_unaccepted_weights(
                candidate_root, seed, models["actor_head_only_ppo"],
                source, trained["actor_head_only_ppo"], means)
            records.append({
                "model_seed": seed,
                "original_initial_model_sha256": original_init,
                "training": trained,
                "real_source_reward_means_by_condition": means,
                "critic_only_source_actions_equal_initial": critic_equal,
                "immutable_actor_candidate": candidate,
                "no_heldout_training": True,
            })
    actions = sum(p["real_original_train_source_actions"]
                  for row in records for p in row["training"].values())
    updates = sum(p["original_ppo_minibatch_updates"]
                  for row in records for p in row["training"].values())
    if actions != 442368 or updates != 5184:
        raise RuntimeError("P33_SOURCE_BUDGET_MISMATCH")
    elapsed = time.perf_counter() - began
    if elapsed > contract["qualification"]["max_training_and_evaluation_wall_seconds"]:
        raise RuntimeError("P33_WALL_BUDGET_EXCEEDED")
    result = frozen_development_outcome(records)
    return {
        "schema": SCHEMA,
        "engineering": "PASS_PUBLIC_SOURCE_DEV_ONLY",
        "candidate_source_sha": source,
        "source_admission": source_admission,
        "actual_original_source_training_actions": actions,
        "actual_PPO_optimizer_minibatches": updates,
        "wall_seconds": elapsed,
        "frozen_development_result": result,
        "checkpoint_optimizer_state_saved": False,
        "accepted_model_pointer_updated": False,
        "production_training_activated": False,
        "scientific_first_use": "NOT_RUN",
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s29-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cloudpickle-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise RuntimeError("P33_RECEIPT_ALREADY_EXISTS")
    report = run(args.s29_root, args.source_root, args.cloudpickle_root,
                 args.candidate_root)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8") as f:
        json.dump(report, f, sort_keys=True, indent=2)
        f.write("\n")
    print(json.dumps({
        "engineering": report["engineering"],
        "scientific_first_use": report["scientific_first_use"],
        "fresh_seed_frozen_development":
            report["frozen_development_result"]["all_three_seed_actor_only_frozen_gate"],
        "checkpoint_count": len(report["records"]),
        "real_source_actions": report["actual_original_source_training_actions"],
        "source_sha": report["candidate_source_sha"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
