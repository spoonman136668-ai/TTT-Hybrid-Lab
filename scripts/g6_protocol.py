"""Frozen development-only episodic regression protocol (no external data)."""
from __future__ import annotations
import hashlib
import json
import torch

SEEDS = (17, 29, 43)
TRAIN_STEPS = 96
TRAIN_BATCH = 16
VAL_BATCHES = 8
VAL_BATCH = 16
SEQ_LEN = 16
FEATURES = 3
WIDTH = 64
UPDATE_LR = 1.0
OPTIMIZER_LR = 0.002
ARMS = ('ttt_live', 'ttt_off', 'causal_attention', 'gru')
PROTOCOL = {
    'schema': 'ttt-hybrid.g6-development-protocol.v1',
    'task': 'episodic_signed_slope_inference',
    'seeds': list(SEEDS), 'train_steps': TRAIN_STEPS,
    'train_batch': TRAIN_BATCH, 'validation_batches': VAL_BATCHES,
    'validation_batch': VAL_BATCH, 'sequence_length': SEQ_LEN,
    'train_seed_multiplier': 100000, 'train_seed_base': 1000000,
    'validation_seed_base': 20000000, 'validation_seed_multiplier': 100000,
    'arms': list(ARMS), 'optimizer': 'AdamW', 'optimizer_lr': OPTIMIZER_LR,
    'ttt_update_lr': UPDATE_LR, 'ttt_disabled_lr': 0.0,
    'primary_metric': 'held-out-development-set-MSE',
    'positive_thresholds': {
        'ttt_vs_no_update_mse_ratio_max': 0.85,
        'ttt_vs_best_conventional_mse_ratio_max': 0.90,
        'ttt_vs_best_conventional_train_time_ratio_max': 1.25,
        'ttt_vs_best_conventional_params_ratio_max': 1.25,
        'ttt_support_scramble_mse_ratio_min': 1.10,
        'individual_seed_tt_vs_no_update_mse_ratio_max': 0.95
    },
    'science_class': 'development_screen_no_sealed_test',
    'provenance': 'generated-on-run, deterministic sources, no future labels in inputs',
}

def batch_seed(seed: int, index: int, *, phase: str) -> int:
    if seed not in SEEDS or index < 0:
        raise ValueError('UNAUTHORIZED_SEED_OR_INDEX')
    if phase == 'train':
        if index >= TRAIN_STEPS: raise ValueError('INDEX_OUTSIDE_TRAIN_BUDGET')
        return 1000000 + seed * 100000 + index
    if phase == 'validation':
        if index >= VAL_BATCHES: raise ValueError('INDEX_OUTSIDE_VALIDATION_BUDGET')
        return 20000000 + seed * 100000 + index
    raise ValueError('UNKNOWN_DATA_PHASE')

def make_batch(seed: int, index: int, *, phase: str, scramble: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    seed_value = batch_seed(seed,index,phase=phase)
    generator = torch.Generator(device='cpu').manual_seed(seed_value)
    b = TRAIN_BATCH if phase=='train' else VAL_BATCH
    slopes = (0.5+1.5*torch.rand(b, generator=generator))
    signs = torch.where(torch.rand(b,generator=generator)>0.5,1.0,-1.0)
    slopes = slopes * signs
    stimuli = 2.0 * torch.rand(b,SEQ_LEN,generator=generator) - 1.0
    features = torch.zeros(b,SEQ_LEN,FEATURES)
    features[:,:,0] = stimuli
    features[:,:-1,1] = slopes[:,None]*stimuli[:,:-1]
    features[:,-1,2] = 1.0
    target = slopes*stimuli[:,-1]
    if scramble:
        # Support values from a different episode: same queries/targets.
        # This never changes validation labels or exposes future observations.
        features[:,:-1,1] = features[:,:-1,1].roll(1,dims=0)
    return features,target

def protocol_sha256() -> str:
    return hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True,separators=(',',':')).encode()).hexdigest()
