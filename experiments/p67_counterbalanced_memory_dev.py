"""P67 isolated causal-memory development experiment (no original hybrid modification).

Prospective source and evaluation rules: experiments/p67_counterbalanced_interference_frozen.json
Synthetic facts are *executed simulator action/outcome pairs*. Future query outcomes
are readout targets, never historical write inputs. This is DEV, not science.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import io
import json
import platform
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

D = 16
ETA = 0.5
SEEDS = {'fit': 970111, 'val': 970113, 'ood_dev': 970117, 'negative': 970119}
COUNTS = {'fit': 1024, 'val': 256, 'ood_dev': 256, 'negative': 128}
MODEL_SEEDS = (970201, 970203, 970207)
MAX_UPDATES = 2000
BATCH = 64
CHECK = 100
MIN_UPDATES = 500
PATIENCE = 6


def digest_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def make_world(split: str, count: int | None = None):
    if split not in SEEDS:
        raise ValueError('P67_UNKNOWN_SPLIT')
    n = COUNTS[split] if count is None else count
    if n <= 0 or n > COUNTS[split]:
        raise ValueError('P67_INVALID_WORLD_COUNT')
    g = torch.Generator().manual_seed(SEEDS[split])
    is_ood = split == 'ood_dev'
    nuisance = 64 if is_ood else 32
    reps = 4
    # Independent world-coordinate frames; OOD receives a separately sampled rotation.
    q = torch.linalg.qr(torch.randn(n, D, D, generator=g)).Q
    if is_ood:
        r = torch.linalg.qr(torch.randn(D, D, generator=g)).Q
        q = torch.matmul(q, r)
    target_key = q[:, :, :8].transpose(1, 2).contiguous()
    target_value = torch.tanh(torch.randn(n, 8, D, generator=g))
    if is_ood:
        target_value = torch.tanh(target_value * 1.15)
    distractor_key = F.normalize(torch.randn(n, nuisance, D, generator=g), dim=-1)
    if split == 'negative':
        # Orthogonal nuisance facts cannot affect target reads under the all-write update.
        orth = q[:, :, 8:]
        raw = torch.randn(n, nuisance, 8, generator=g)
        distractor_key = F.normalize(torch.bmm(raw, orth.transpose(1, 2)), dim=-1)
    distractor_value = torch.tanh(torch.randn(n, nuisance, D, generator=g))
    # Only the factual stream contains a/e. Roles never enter model training features.
    keys = torch.cat((target_key.repeat(1, reps, 1), distractor_key), dim=1)
    values = torch.cat((target_value.repeat(1, reps, 1), distractor_value), dim=1)
    roles = torch.cat((torch.ones(n, 8 * reps, dtype=torch.bool),
                       torch.zeros(n, nuisance, dtype=torch.bool)), dim=1)
    perm = torch.rand(n, keys.shape[1], generator=g).argsort(dim=1)
    keys = keys.gather(1, perm[:, :, None].expand_as(keys))
    values = values.gather(1, perm[:, :, None].expand_as(values))
    roles = roles.gather(1, perm)
    # Query action executed in simulator AFTER all historical events have finished.
    chosen = torch.randint(8, (n,), generator=g)
    query = target_key[torch.arange(n), chosen]
    observed_query_effect = target_value[torch.arange(n), chosen]
    # Wrong credit is a separate error-corrupted factual index, never a real action.
    wrong = F.normalize(torch.randn(n, keys.shape[1], D, generator=g), dim=-1)
    return {'action': keys, 'effect': values, 'query_action': query,
            'observed_query_effect': observed_query_effect, 'role_audit': roles,
            'wrong_credit_key': wrong}


def source_hash(world):
    return hashlib.sha256(b''.join(world[k].contiguous().numpy().tobytes() for k in
        ('action', 'effect', 'query_action', 'observed_query_effect', 'role_audit',
         'wrong_credit_key'))).hexdigest()


def role_position_balanced(roles):
    # A prevalence-adjusted, pooled four-bin audit: 0 means perfect independence.
    n, length = roles.shape
    del n
    if length % 4:
        raise ValueError('P67_NONQUARTER_LENGTH')
    fractions = roles.float().reshape(-1, 4, length // 4).mean((0, 2))
    total = roles.float().mean()
    return float((fractions - total).abs().max()), fractions.tolist()


def index_only_balanced_accuracy(roles):
    # An index-only deterministic Bayes classifier evaluated on the observed split.
    # No index features are supplied to any write gate.
    per_pos = roles.float().mean(0)
    prediction = per_pos >= 0.5
    tpr = (prediction[None, :] & roles).sum().float() / roles.sum().clamp_min(1)
    tnr = ((~prediction)[None, :] & (~roles)).sum().float() / (~roles).sum().clamp_min(1)
    return float((tpr + tnr) / 2)


def factual(model_world):
    # Explicit allowlist prevents accidental evaluator metadata or query leakage.
    return {k: model_world[k] for k in ('action', 'effect')}


def sampled(world, ids):
    return {k: v[ids] for k, v in world.items()}


class WriteGate(nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.theta = nn.Parameter((torch.rand(4, generator=gen) - .5) * .1)

    def forward(self, facts, *, carry=None, lo=0, hi=None,
                mode='learned', threshold=None, rate=None, random_seed=0,
                wrong_key=None, record_rate=False):
        a, y = facts['action'], facts['effect']
        if a.ndim != 3 or a.shape != y.shape or a.shape[-1] != D:
            raise ValueError('P67_BAD_FACTS')
        if not (torch.isfinite(a).all() and torch.isfinite(y).all()):
            raise ValueError('P67_NONFINITE_FACTS')
        n, steps, _ = a.shape
        hi = steps if hi is None else hi
        if not 0 <= lo <= hi <= steps:
            raise ValueError('P67_BAD_RANGE')
        w = a.new_zeros(n, D, D) if carry is None else carry
        if w.shape != (n, D, D):
            raise ValueError('P67_BAD_CARRY')
        gen = torch.Generator().manual_seed(random_seed)
        rates = []
        # Repeated-key similarity is computed from *previous factual actions only*.
        for t in range(lo, hi):
            key = wrong_key[:, t] if wrong_key is not None else a[:, t]
            v = y[:, t]
            pred = torch.bmm(w, key.unsqueeze(-1)).squeeze(-1)
            delta = v - pred
            error = delta.square().mean(-1)
            if mode == 'learned':
                phi = torch.stack((torch.ones_like(error),
                    torch.log(error + 1e-6), torch.log(pred.square().mean(-1) + 1e-6),
                    F.cosine_similarity(pred, v, dim=-1, eps=1e-6)), -1)
                gate = torch.sigmoid(phi @ self.theta)
            elif mode == 'all':
                gate = torch.ones_like(error)
            elif mode == 'surprise':
                gate = (error <= threshold).to(error.dtype)
            elif mode == 'novelty':
                if t == 0:
                    gate = torch.ones_like(error)
                else:
                    seen = F.cosine_similarity(key[:, None, :],
                        (wrong_key if wrong_key is not None else a)[:, :t, :],
                        dim=-1).amax(dim=1)
                    gate = (seen >= threshold).to(error.dtype)
            elif mode == 'random':
                gate = (torch.rand(n, generator=gen) < rate).to(error.dtype)
            else:
                raise ValueError('P67_BAD_MODE')
            w = w + ETA * gate[:, None, None] * delta[:, :, None] * key[:, None, :]
            if record_rate:
                rates.append(gate.detach().mean())
        fraction = float(torch.stack(rates).mean()) if rates else None
        return w, fraction


def predict(w, query_action):
    return torch.bmm(w, query_action.unsqueeze(-1)).squeeze(-1)


def per_world_error(pred, target):
    return (pred - target).square().mean(-1)


def loss_gate(model, batch, *, wrong_credit=False):
    w, _ = model(factual(batch), wrong_key=batch['wrong_credit_key'] if wrong_credit else None)
    return per_world_error(predict(w, batch['query_action']), batch['observed_query_effect']).mean()


class GRUBaseline(nn.Module):
    # Recurrent factual state is exactly 256 floats; weights/FLOPs are NOT matched.
    def __init__(self, seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.rec = nn.GRU(2 * D, 256, batch_first=True)
            self.readout = nn.Linear(256 + D, D)

    def forward(self, facts, query_action):
        a, e = facts['action'], facts['effect']
        _, h = self.rec(torch.cat((a, e), dim=-1))
        return self.readout(torch.cat((h[-1], query_action), -1))


def loss_gru(model, batch):
    out = model(factual(batch), batch['query_action'])
    return per_world_error(out, batch['observed_query_effect']).mean()


def restore_exact(gate, val):
    with torch.no_grad():
        facts = factual(val)
        full, _ = gate(facts)
        midpoint = facts['action'].shape[1] // 2
        partial, _ = gate(facts, hi=midpoint)
        buff = io.BytesIO()
        torch.save({'model': gate.state_dict(), 'state': partial}, buff)
        buff.seek(0)
        saved = torch.load(buff, weights_only=True)
        resumed_gate = WriteGate(970201)
        resumed_gate.load_state_dict(saved['model'])
        resumed, _ = resumed_gate(facts, carry=saved['state'], lo=midpoint)
        return torch.equal(resumed, full)


def train(model, kind, fit, val, batches, *, max_updates=MAX_UPDATES):
    opt = torch.optim.AdamW(model.parameters(), lr=.003)
    best = float('inf')
    best_weights = None
    stagnant = 0
    history = []
    start = time.perf_counter()
    for step in range(max_updates):
        train_batch = sampled(fit, batches[step])
        model.train()
        opt.zero_grad(set_to_none=True)
        train_loss = loss_gru(model, train_batch) if kind == 'gru' else loss_gate(model, train_batch, wrong_credit=(kind == 'wrong'))
        if not torch.isfinite(train_loss):
            raise ValueError('P67_NONFINITE_TRAINING')
        train_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        if (step + 1) % CHECK:
            continue
        model.eval()
        with torch.no_grad():
            metric = float(loss_gru(model, val) if kind == 'gru' else loss_gate(model, val, wrong_credit=(kind == 'wrong')))
        history.append({'step': step + 1, 'val_mse': metric})
        if metric < best - 1e-6:
            best = metric
            best_weights = copy.deepcopy(model.state_dict())
            stagnant = 0
        else:
            stagnant += 1
        if step + 1 >= MIN_UPDATES and stagnant >= PATIENCE:
            break
    if best_weights is not None:
        model.load_state_dict(best_weights)
    return {'steps': step + 1, 'early_stopped': stagnant >= PATIENCE and step + 1 >= MIN_UPDATES,
            'best_validation_mse': best, 'validation_trace': history,
            'duration_seconds': time.perf_counter() - start,
            'parameters': sum(p.numel() for p in model.parameters())}


def static_controls(world):
    imbalance, fractions = role_position_balanced(world['role_audit'])
    balanced = index_only_balanced_accuracy(world['role_audit'])
    return {'source_sha256': source_hash(world), 'max_bin_imbalance': imbalance,
            'relevant_fraction_quartiles': fractions, 'index_only_balanced_accuracy': balanced,
            'no_phase_shortcut': imbalance <= .05 and balanced <= .55}


def eval_gate(gate, world, **kw):
    with torch.no_grad():
        state, rate = gate(factual(world), record_rate=True, **kw)
        return per_world_error(predict(state, world['query_action']), world['observed_query_effect']).numpy(), rate


def bootstrap_lower_relative(a, b, *, seed=970401, replicates=5000):
    # Paired resample over held-out worlds, never pointwise over the event stream.
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(a), (replicates, len(a)), generator=gen)
    x = torch.as_tensor(a)[idx].mean(1)
    y = torch.as_tensor(b)[idx].mean(1).clamp_min(1e-12)
    return float(torch.quantile(1 - x / y, .025))


def smoke():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    w = make_world('fit', 8)
    gate = WriteGate(970201)
    check = static_controls(make_world('fit', 256))
    loss = loss_gate(gate, w)
    loss.backward()
    return {'mode': 'ENGINEERING_SMOKE_NOT_FROZEN_STUDY', 'finite_loss': bool(torch.isfinite(loss)),
            'finite_grad': bool(torch.isfinite(gate.theta.grad).all()),
            'cold_restore_bitwise': restore_exact(gate, w),
            'distinct_seed_initializations': len({digest_tensor(WriteGate(s).theta) for s in MODEL_SEEDS}) == 3,
            'index_only_balanced_accuracy': check['index_only_balanced_accuracy'],
            'source_hash': check['source_sha256'], 'scientific_pass': False}


def study():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    data = {k: make_world(k) for k in SEEDS}
    audits = {k: static_controls(v) for k, v in data.items()}
    if not all(a['no_phase_shortcut'] for a in audits.values()):
        return {'status': 'DEV_FAIL_SOURCE_SHORTCUT', 'audits': audits, 'scientific_pass': False}
    sampler = torch.Generator().manual_seed(970299)
    batches = torch.randint(COUNTS['fit'], (MAX_UPDATES, BATCH), generator=sampler)
    shared_sampler_sha256 = digest_tensor(batches)
    candidate_ref = WriteGate(970201)
    # Validation-selected controls only, never select on OOD.
    thresholds = (.05, .15, .4, .75, .95)
    def choose(mode):
        return min(((float(eval_gate(candidate_ref, data['val'], mode=mode, threshold=t)[0].mean()), t) for t in thresholds), key=lambda x: x[0])[1]
    surprise = choose('surprise')
    novelty = choose('novelty')
    controls = {}
    control_errors = {}
    for name, opts in [('always', {'mode': 'all'}), ('surprise', {'mode': 'surprise', 'threshold': surprise}),
                       ('novelty', {'mode': 'novelty', 'threshold': novelty})]:
        score, _ = eval_gate(candidate_ref, data['ood_dev'], **opts)
        control_errors[name] = score
        controls[name] = float(score.mean())
    control_errors['zero_state'] = per_world_error(torch.zeros_like(data['ood_dev']['observed_query_effect']), data['ood_dev']['observed_query_effect']).numpy()
    controls['zero_state'] = float(control_errors['zero_state'].mean())
    # FIFO 8: nearest factual key in final 8 steps, no privileged labels.
    q = data['ood_dev']['query_action']
    recent_a = data['ood_dev']['action'][:, -8:]
    recent_y = data['ood_dev']['effect'][:, -8:]
    sim = (recent_a * q[:, None]).sum(-1)
    ix = sim.argmax(1)
    retrieved = recent_y[torch.arange(len(ix)), ix] * (sim.max(1).values > .999).float()[:, None]
    control_errors['fifo8'] = per_world_error(retrieved, data['ood_dev']['observed_query_effect']).numpy()
    controls['fifo8'] = float(control_errors['fifo8'].mean())
    out = []
    gate_inits = []
    for seed in MODEL_SEEDS:
        gate = WriteGate(seed)
        wrong = WriteGate(seed + 800)
        gru = GRUBaseline(seed)
        gate_inits.append(digest_tensor(gate.theta))
        arms = {'gate': (gate, 'gate'), 'wrong': (wrong, 'wrong'), 'gru': (gru, 'gru')}
        trained = {name: train(model, kind, data['fit'], data['val'], batches) for name, (model, kind) in arms.items()}
        candidate_errors, rate = eval_gate(gate, data['ood_dev'])
        wrong_errors, _ = eval_gate(wrong, data['ood_dev'], wrong_key=data['ood_dev']['wrong_credit_key'])
        fit_rate = eval_gate(gate, sampled(data['fit'], torch.arange(64)))[1]
        random_errors, _ = eval_gate(gate, data['ood_dev'], mode='random', rate=fit_rate, random_seed=seed + 20000)
        with torch.no_grad():
            gru_errors = per_world_error(gru(factual(data['ood_dev']), data['ood_dev']['query_action']),
                                         data['ood_dev']['observed_query_effect']).numpy()
        zero_val = float(data['val']['observed_query_effect'].square().mean())
        competence = trained['gru']['early_stopped'] and trained['gru']['best_validation_mse'] <= .85 * zero_val
        competitors = {**control_errors, 'gru': gru_errors}
        best_name = min(competitors, key=lambda name: float(competitors[name].mean()))
        best_control_errors = competitors[best_name]
        best_control = float(best_control_errors.mean())
        passed = (competence and trained['gate']['early_stopped'] and trained['wrong']['early_stopped']
                 and float(candidate_errors.mean()) <= .9 * best_control
                 and bootstrap_lower_relative(candidate_errors, best_control_errors) > 0
                 and float(wrong_errors.mean()) >= 1.2 * float(candidate_errors.mean())
                 and float(random_errors.mean()) >= 1.15 * float(candidate_errors.mean())
                 and restore_exact(gate, sampled(data['val'], torch.arange(8))))
        out.append({'seed': seed, 'initial_theta_sha256': gate_inits[-1], 'trained': trained,
                    'scores': {'learned': float(candidate_errors.mean()), 'wrong_credit': float(wrong_errors.mean()),
                               'random_rate_matched': float(random_errors.mean()), 'gru': float(gru_errors.mean())},
                    'gate_write_fraction_OOD': rate, 'gate_write_fraction_FIT': fit_rate,
                    'competent_gru': competence, 'best_comparator': best_name,
                    'bootstrap_lower_relative_gain': bootstrap_lower_relative(candidate_errors, best_control_errors),
                    'strict_dev_pass': passed})
    # No claim of scientific advancement absent independent source-owner admission.
    return {'schema': 'ttt-hybrid.p67-counterbalanced-dev-result.v1', 'status': 'DEV_PASS' if all(x['strict_dev_pass'] for x in out) else 'DEV_FAIL',
            'model_seeds': MODEL_SEEDS, 'distinct_initializations': len(set(gate_inits)) == 3,
            'source_audits': audits, 'sampler_sha256': shared_sampler_sha256, 'classical_controls': controls,
            'rows': out, 'elapsed_seconds': time.perf_counter() - start, 'python': platform.python_version(),
            'torch': torch.__version__, 'whole_process_compute_matched': False,
            'flops_exact': 'UNMEASURED', 'native_gpu': 'NOT_RUN', 'full_cross_platform': 'NOT_RUN',
            'scientific_pass': False, 'forge': 'UNCHANGED'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'study'], required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    result = smoke() if args.mode == 'smoke' else study()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result.get('status', result['mode'] if 'mode' in result else None),
                      'scientific_pass': False}, sort_keys=True))


if __name__ == '__main__':
    main()