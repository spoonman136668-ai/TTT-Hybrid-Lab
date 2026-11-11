"""P99: prospective event-triggered checkpoint causal memory on fresh A-B-A worlds.

New protocol was frozen before source birth at 910ffd2d0fe68ec1f52d49b4e266e31db2d1736a.
Developer experiment only; no protected OOD, original-model edits, or science acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F

from experiments import p86_portable_causal_preflight as p86
from experiments import p97_recurrent_context_dev as p97

FREEZE = "910ffd2d0fe68ec1f52d49b4e266e31db2d1736a"
FREEZE_BLOB = "0bcf6c248f0f468f998aa18112d984f13283bb83"
D = 16
SEEDS = {"A_FIT": 990111, "A_CAL": 990113,
         "B_FIT": 990211, "B_CAL": 990213}
RESERVED = {"OOD": 990117, "NEGATIVE": 990119}
N = {"FIT": 128, "CAL": 64}
SHOCK = (4., 6.)
PERSIST = (2, 3)
BLOBS = {
    "model.py": "5d9ea57b827aa1796548447868fa5ee14c251b49",
    "experiments/p86_portable_causal_preflight.py":
        "008492665ee9090db5791963f2e2336ea283ee62",
    "experiments/p97_recurrent_context_dev.py":
        "74030c3d7d2f5f51781c740493be21d3c769792d",
    "experiments/p98_blind_switch_diagnostics.py":
        "7cd9521c2ceaa6f84f2128ba16d37ceaae7ff006",
}


def source(key: str) -> dict[str, torch.Tensor]:
    if key not in SEEDS:
        raise ValueError("P99_PROTECTED_SOURCE_DENIED")
    family, split = key.split("_")
    n, seed = N[split], SEEDS[key]
    changed = set(sorted(range(n), key=lambda w: p86.digest_int(
        seed, w, "p99-regime", 0, 0, 2**32))[:n // 2])
    history, probes, targets, flags = [], [], [], []
    for world in range(n):
        u0, v0 = p86.basis(seed, world, "p99-u0"), p86.basis(seed, world, "p99-v0")
        u1, v1 = p86.basis(seed, world, "p99-u1"), p86.basis(seed, world, "p99-v1")
        facts, queries, effects = [], [], []
        for t in range(72):
            a = [p86.digest_int(seed, world, "p99-action", t, j, 193) - 96
                 for j in range(D)]
            if family == "B":
                a = [z // 2 ** (j % 4) for j, z in enumerate(a)]
            u, v = (u1, v1) if (world in changed and 24 <= t < 48) else (u0, v0)
            facts.append([z / 128 for z in a] + p86.causal_effect(a, u, v))
        for phase in range(3):
            a = [p86.digest_int(seed, world, "p99-probe", phase, j, 193) - 96
                 for j in range(D)]
            if family == "B":
                a = [z // 2 ** (j % 4) for j, z in enumerate(a)]
            u, v = (u1, v1) if (world in changed and phase == 1) else (u0, v0)
            queries.append([z / 128 for z in a])
            effects.append(p86.causal_effect(a, u, v))
        history.append(facts)
        probes.append(queries)
        targets.append(effects)
        flags.append(world in changed)
    world = {
        "history": torch.tensor(history, dtype=torch.float32),
        "probes": torch.tensor(probes, dtype=torch.float32),
        "targets": torch.tensor(targets, dtype=torch.float32),
        "changed_world": torch.tensor(flags, dtype=torch.bool),
    }
    if not p97.audit(world):
        raise ValueError("P99_NEW_SOURCE_AUDIT")
    return world


@dataclass(frozen=True)
class CheckpointState:
    active: torch.Tensor  # [b,16,16] float, current causal W
    checkpoint: torch.Tensor  # [b,16,16] float; dormant model, lazily written
    energy: torch.Tensor  # [b] float, active innovation scale
    step: torch.Tensor  # [b] int64 confirmed observation count
    shock_count: torch.Tensor  # [b] int64
    restore_count: torch.Tensor  # [b] int64
    mode: torch.Tensor  # [b] int64, 0: original, 1: alternate, 2: restored
    switches: torch.Tensor  # [b] int64 total checkpoint/return events
    has_checkpoint: torch.Tensor  # [b] bool

    def snapshot(self) -> "CheckpointState":
        return CheckpointState(*(getattr(self, k).detach().clone()
                                 for k in self.__dataclass_fields__))


class EventCheckpointMemory:
    """One W256 active write per confirmed fact; dormant W256 compared only after shock."""

    def __init__(self, shock: float, persist: int):
        if shock not in SHOCK or persist not in PERSIST:
            raise ValueError("P99_UNFROZEN_HYPERPARAMETERS")
        self.shock, self.persist = shock, persist

    def initial(self, action: torch.Tensor) -> CheckpointState:
        if action.ndim != 2 or action.shape[-1] != D or action.dtype != torch.float32:
            raise ValueError("P99_INVALID_QUERY")
        b = len(action)
        empty = action.new_zeros((b, D, D))
        counts = torch.zeros(b, dtype=torch.int64, device=action.device)
        return CheckpointState(empty, empty.clone(), action.new_zeros(b),
                               counts.clone(), counts.clone(), counts.clone(),
                               counts.clone(), counts.clone(),
                               torch.zeros(b, dtype=torch.bool, device=action.device))

    def validate(self, state: CheckpointState, action: torch.Tensor) -> None:
        b = len(action)
        if (not isinstance(state, CheckpointState)
            or action.dtype != torch.float32 or action.shape[-1] != D
            or not bool(torch.isfinite(action).all())
            or state.active.shape != (b, D, D)
            or state.checkpoint.shape != (b, D, D)
            or state.energy.shape != (b,)
            or any(getattr(state, k).shape != (b,) for k in
                   ("step", "shock_count", "restore_count", "mode", "switches", "has_checkpoint"))
            or any(getattr(state, k).dtype != torch.int64 for k in
                   ("step", "shock_count", "restore_count", "mode", "switches"))
            or state.has_checkpoint.dtype != torch.bool
            or any(getattr(state, k).dtype != torch.float32
                   for k in ("active", "checkpoint", "energy"))
            or any(getattr(state, k).device != action.device
                   for k in state.__dataclass_fields__)
            or any(not bool(torch.isfinite(getattr(state, k)).all())
                   for k in ("active", "checkpoint", "energy"))
            or not bool((state.energy >= 0).all())
            or not bool(((state.mode >= 0) & (state.mode <= 2)).all())
            or not bool((state.step >= 0).all())):
            raise ValueError("P99_INVALID_STATE")

    def read_only(self, query: torch.Tensor, state: CheckpointState) -> torch.Tensor:
        if query.ndim != 2:
            raise ValueError("P99_BAD_READ")
        self.validate(state, query)
        return torch.bmm(state.active, F.normalize(query, dim=-1).unsqueeze(-1)).squeeze(-1)

    def ingest(self, actions: torch.Tensor, effects: torch.Tensor,
               confirmed: torch.Tensor, state: CheckpointState,
               trace: bool = False):
        if (not isinstance(actions, torch.Tensor) or actions.ndim != 3
            or actions.shape[-1] != D or actions.dtype != torch.float32
            or not isinstance(effects, torch.Tensor) or effects.shape != actions.shape
            or effects.dtype != torch.float32 or effects.device != actions.device
            or not isinstance(confirmed, torch.Tensor)
            or confirmed.shape != actions.shape[:2]
            or confirmed.dtype != torch.bool or confirmed.device != actions.device):
            raise ValueError("P99_INVALID_FACTS")
        self.validate(state, actions)
        safe = torch.where(confirmed.unsqueeze(-1), effects, torch.zeros_like(effects))
        if not bool(torch.isfinite(safe).all()):
            raise ValueError("P99_NONFINITE_CONFIRMED_EFFECT")
        W, saved, ema = state.active, state.checkpoint, state.energy
        step, shock, restore = state.step, state.shock_count, state.restore_count
        mode, switches, has = state.mode, state.switches, state.has_checkpoint
        modes = []
        all_confirmed = bool(confirmed.all())
        for t in range(actions.shape[1]):
            ok = confirmed[:, t]
            k = F.normalize(actions[:, t], dim=-1)
            prediction = torch.bmm(W, k.unsqueeze(-1)).squeeze(-1)
            error = safe[:, t] - prediction
            current_sq = error.square().mean(dim=-1)
            # Compare against PRE-event innovation: no target future leakage.
            unexpected = ((step >= 12) & (mode == 0) &
                          (current_sq > self.shock * ema.clamp_min(1e-6)))
            next_shock = torch.where(unexpected, shock + 1, torch.zeros_like(shock))
            spawn = (next_shock >= self.persist) & ok & (mode == 0)
            # Only use the dormant model when a checkpoint actually exists.
            if bool(has.any()):
                prior_prediction = torch.bmm(saved, k.unsqueeze(-1)).squeeze(-1)
                prior_sq = (safe[:, t] - prior_prediction).square().mean(-1)
                advantage = (mode == 1) & has & (prior_sq < .8 * current_sq)
                next_restore = torch.where(advantage, restore + 1, torch.zeros_like(restore))
            else:
                next_restore = torch.zeros_like(restore)
            return_old = (next_restore >= 2) & ok & has & (mode == 1)
            # Preserve W at the trigger event, BEFORE the alternative update.
            saved = torch.where(spawn[:, None, None], W, saved)
            start = torch.where(spawn[:, None, None], torch.zeros_like(W),
                                torch.where(return_old[:, None, None], saved, W))
            updated_error = safe[:, t] - torch.bmm(start, k.unsqueeze(-1)).squeeze(-1)
            proposal = .99 * start + updated_error.unsqueeze(-1) * k.unsqueeze(1)
            if all_confirmed:
                W = proposal
            else:
                W = torch.where(ok[:, None, None], proposal, W)
            next_energy = .90 * ema + .10 * updated_error.square().mean(-1)
            ema = torch.where(ok, next_energy, ema)
            step = step + ok.to(torch.int64)
            mode = torch.where(return_old, torch.full_like(mode, 2),
                               torch.where(spawn, torch.ones_like(mode), mode))
            switches = switches + spawn.to(torch.int64) + return_old.to(torch.int64)
            has = has | spawn
            shock = torch.where(ok & ~spawn, next_shock,
                                torch.where(ok, torch.zeros_like(shock), shock))
            restore = torch.where(ok & ~return_old, next_restore,
                                  torch.where(ok, torch.zeros_like(restore), restore))
            if trace:
                modes.append(mode.clone())
        result = CheckpointState(W, saved, ema, step, shock, restore,
                                 mode, switches, has)
        if trace:
            return result, torch.stack(modes, dim=1)
        return result

    def episode(self, history: torch.Tensor, probes: torch.Tensor,
                confirmed: torch.Tensor | None = None,
                initial: CheckpointState | None = None,
                trace: bool = False):
        if (history.ndim != 3 or history.shape[1:] != (72, 2*D)
            or probes.shape != (len(history), 3, D)
            or history.dtype != torch.float32 or probes.dtype != torch.float32):
            raise ValueError("P99_INVALID_EPISODE")
        if confirmed is None:
            confirmed = torch.ones(history.shape[:2], dtype=torch.bool, device=history.device)
        state = self.initial(probes[:, 0]) if initial is None else initial
        predictions, traces = [], []
        for i in range(3):
            facts = history[:, i*24:(i+1)*24]
            if trace:
                state, modes = self.ingest(facts[:, :, :D], facts[:, :, D:],
                                           confirmed[:, i*24:(i+1)*24], state, trace=True)
                traces.append(modes)
            else:
                state = self.ingest(facts[:, :, :D], facts[:, :, D:],
                                    confirmed[:, i*24:(i+1)*24], state)
            predictions.append(self.read_only(probes[:, i], state))
        pred = torch.stack(predictions, 1)
        return (pred, state, torch.cat(traces, dim=1)) if trace else (pred, state)


def select_fit(world: dict):
    rows = []
    for shock in SHOCK:
        for persist in PERSIST:
            model = EventCheckpointMemory(shock, persist)
            with torch.no_grad():
                prediction, _ = model.episode(world["history"], world["probes"])
            rows.append({"shock": shock, "persist": persist,
                         "metrics": p97.mse(prediction, world)})
    best = min(rows, key=lambda x: x["metrics"]["all"])
    return (best["shock"], best["persist"]), rows


def diagnostic(mode: torch.Tensor, world: dict):
    changed = world["changed_world"]
    stable = ~changed
    spawned = mode == 1
    return_old = mode == 2
    fp = spawned[stable].any(-1) | return_old[stable].any(-1)
    mid = spawned[changed, 24:36].any(-1)
    back = return_old[changed, 48:60].any(-1)
    return {
        "false_stable_switch_count": int(fp.sum()),
        "false_stable_switch_rate": float(fp.float().mean()),
        "mid_changed_12_count": int(mid.sum()),
        "mid_changed_12_rate": float(mid.float().mean()),
        "returned_12_count": int(back.sum()),
        "returned_12_rate": float(back.float().mean()),
        "state_active_alternate_final_count": int((mode[:, -1] == 1).sum()),
    }


def negative(model: EventCheckpointMemory, world: dict):
    tiny = {k: v[:8].clone() for k, v in world.items()}
    h, q = tiny["history"], tiny["probes"]
    with torch.no_grad():
        entire, whole, mode = model.episode(h, q, trace=True)
        state = model.initial(q[:, 0])
        parts = []
        for chunk in range(3):
            facts = h[:, chunk*24:(chunk+1)*24]
            state = model.ingest(facts[:, :, :D], facts[:, :, D:],
                                 torch.ones((8, 24), dtype=torch.bool),
                                 state.snapshot())
            parts.append(model.read_only(q[:, chunk], state))
        cold = (all(torch.equal(getattr(state, k), getattr(whole, k))
                    for k in CheckpointState.__dataclass_fields__)
                and torch.equal(torch.stack(parts, 1), entire))
        mask = torch.ones((8, 72), dtype=torch.bool)
        mask[:, 1::3] = False
        altered = h.clone()
        altered[:, 1::3, D:] = float("nan")
        real, state_real = model.episode(h, q, mask)
        fake, state_fake = model.episode(altered, q, mask)
        masked = (torch.equal(real, fake) and
                  all(torch.equal(getattr(state_real, k), getattr(state_fake, k))
                      for k in CheckpointState.__dataclass_fields__))
        shifted = q[:, 2].roll(1, dims=0)
        _ = model.read_only(shifted, whole)
        no_future = torch.equal(whole.active, whole.snapshot().active)
    return {"cold_exact":bool(cold), "masked_nan_inert":bool(masked),
            "query_cannot_write":bool(no_future)}


def timed(fn):
    with torch.no_grad():
        for _ in range(2):
            fn()
        values = []
        for _ in range(7):
            start = time.perf_counter()
            fn()
            values.append((time.perf_counter()-start)*1000)
    return statistics.median(values)


def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(__file__).resolve().parents[1]
    freeze_path = root/"experiments/p99_event_checkpoint_frozen.json"
    if p86.blob(freeze_path) != FREEZE_BLOB:
        raise ValueError("P99_FROZEN_CONTRACT_BLOB_DRIFT")
    freeze = json.loads(freeze_path.read_text())
    if freeze["schema"] != "ttt-hybrid.p99-event-driven-checkpoint-dev-freeze.v1":
        raise ValueError("P99_FREEZE_SCHEMA_DRIFT")
    for path, digest in BLOBS.items():
        if p86.blob(root/path) != digest:
            raise ValueError("P99_PARENT_BLOB_CUSTODY")
    if freeze["new_source"]["seeds"] != {**SEEDS, **{"OOD_RESERVED":990117,"NEGATIVE_RESERVED":990119}}:
        raise ValueError("P99_PROTOCOL_SOURCE_DRIFT")
    if (freeze["frozen_method"]["shock_threshold_grid"] != list(SHOCK)
        or freeze["frozen_method"]["shock_persistence_grid"] != list(PERSIST)
        or freeze["frozen_method"]["restore_relative_threshold"] != .8
        or freeze["frozen_method"]["restore_persistence"] != 2
        or freeze["frozen_acceptance"]["CPU_full_episode_ratio_to_fit_W256_max"] != 3.):
        raise ValueError("P99_PROTOCOL_GRID_DRIFT")
    out = {"schema":"ttt-hybrid.p99-event-checkpoint-dev-result.v1",
           "freeze_commit":FREEZE, "status":"NOT_RUN", "scientific_pass":False,
           "external_source_used":False, "ood_generated":False,
           "negative_reserved_generated":False, "original_model_edited":False,
           "trained_neural_weights":False, "predictive_float_state":513}
    worlds = {key:source(key) for key in SEEDS}
    out["source_sha256"] = {key:p97.fingerprint(w) for key,w in worlds.items()}
    if len(set(out["source_sha256"].values())) != 4:
        raise ValueError("P99_SOURCE_HASH_COLLISION")
    out["families"] = {}
    for family in ("A","B"):
        fit,cal = worlds[family+"_FIT"],worlds[family+"_CAL"]
        selected_base, base_grid = p97.select_control(fit)
        selected, candidate_grid = select_fit(fit)
        prior_setting, prior_grid = p97.select_candidate(fit)
        model=EventCheckpointMemory(*selected)
        prior_model=p97.RecurrentContextMemory(*prior_setting)
        with torch.no_grad():
            candidate,state,modes=model.episode(cal["history"],cal["probes"],trace=True)
            prior_prediction,_=prior_model.episode(cal["history"],cal["probes"])
            baseline=p97.baseline(cal["history"],cal["probes"],*selected_base)
            fixed=p97.baseline(cal["history"],cal["probes"])
            zero=torch.zeros_like(candidate)
            wrong=cal["history"].clone()
            wrong[:,:,D:]=wrong[:,:,D:].roll(13,dims=1)
            wrong_pred,_=model.episode(wrong,cal["probes"])
            scores={"candidate":p97.mse(candidate,cal),
                    "prior_P97_two_context":p97.mse(prior_prediction,cal),
                    "FIT_selected_W256":p97.mse(baseline,cal),
                    "fixed_W256":p97.mse(fixed,cal),
                    "zero":p97.mse(zero,cal),
                    "wrong_credit":p97.mse(wrong_pred,cal)}
            routing=diagnostic(modes,cal)
            neg=negative(model,cal)
        neg["wrong_credit_degrades"]=(
            scores["wrong_credit"]["all"]>=1.05*scores["candidate"]["all"])
        if not all(neg.values()):
            raise ValueError("P99_REQUIRED_NEGATIVE_FAILED")
        inert=cal["history"].new_zeros((len(cal["history"]),257))
        if (state.active.numel()!=len(cal["history"])*256
            or state.checkpoint.numel()!=len(cal["history"])*256
            or state.energy.numel()!=len(cal["history"])
            or inert.numel()!=len(cal["history"])*257
            or not torch.equal(baseline,p97.baseline(cal["history"],cal["probes"],*selected_base))):
            raise ValueError("P99_CAPACITY_PARITY")
        slice8={k:v[:8] for k,v in cal.items()}
        cost={"W256_ms":timed(lambda:p97.baseline(slice8["history"],slice8["probes"],*selected_base)),
              "checkpoint_ms":timed(lambda:model.episode(slice8["history"],slice8["probes"]))}
        cost["ratio"]=cost["checkpoint_ms"]/cost["W256_ms"]
        c,b=scores["candidate"],scores["FIT_selected_W256"]
        quality={
            "middle_changed":c["mid_changed"]<=.95*b["mid_changed"],
            "return_changed":c["return_changed"]<=.85*b["return_changed"],
            "stable_final":c["stable_final"]<=1.05*b["stable_final"],
            "overall":c["all"]<=.95*b["all"],
            "stable_false_switch":routing["false_stable_switch_rate"]<=.10,
            "changed_detect":routing["mid_changed_12_rate"]>=.75,
            "return_detect":routing["returned_12_rate"]>=.75,
            "episode_cost":cost["ratio"]<=3.,
        }
        out["families"][family]={
            "FIT_control":{"eta":selected_base[0],"decay":selected_base[1]},
            "FIT_grid":base_grid,
            "FIT_candidate":{"shock":selected[0],"persist":selected[1]},
            "FIT_candidate_grid":candidate_grid,
            "FIT_prior_P97":{"surprise":prior_setting[0],"alternative":prior_setting[1]},
            "FIT_prior_P97_grid":prior_grid,
            "CAL":scores,"routing":routing,"negative_controls":neg,
            "quality":quality,"cpu":cost,"family_pass":all(quality.values()),
            "changed_world_count":int(cal["changed_world"].sum()),
            "matched_inert_513_baseline_parity":True}
    out["status"]=("DEV_CAL_ONLY_PASS_PROTECTED_FIRST_USE_REQUIRED"
                   if all(x["family_pass"] for x in out["families"].values())
                   else "DEV_FAIL_QUALITY_OR_COST_OR_ROUTING")
    return out


def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--receipt",type=Path,required=True)
    args=ap.parse_args()
    x=run()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":x["status"],"family_pass":{
        k:v["family_pass"] for k,v in x["families"].items()},
        "scientific_pass":x["scientific_pass"]}))


if __name__=="__main__":
    main()
