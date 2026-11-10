"""Source-free engineering checks for P60's preregistered necessity ablations."""
import ast
import hashlib
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments import p60_history_necessity_ablation_dev as p60


def test_original_exact_model_physics_pinned():
    p60.check_source()


def test_frozen_seeds_and_budgets():
    assert len(p60.TRAIN_SEEDS)==128
    assert p60.VAL_SEEDS==(tuple(range(987200,987232)))
    assert p60.OOD_SEEDS==(tuple(range(987400,987496)))
    assert p60.MODEL_SEEDS==(9611,9623,9631)
    assert p60.UPDATES==400 and p60.BATCH==96


def test_current_only_teacher_is_exactly_current_sensor_without_history():
    assert p60.teacher_action([0,-.001])==-1
    assert p60.teacher_action([0,.001])==1
    assert p60.teacher_action([.5,0.00005])==1


def test_exact_p59_original_fit_factual_trace_reproduced():
    # Only previously consumed supervised FIT source, not P60's fresh dev OOD.
    x,y,episodes,digest=p60.teacher_episodes(p60.TRAIN_SEEDS)
    assert x.shape==(10195,5,5)
    assert len(episodes)==128
    assert digest==p60.P59_FIT_SOURCE_HASH
    assert all(v['goal'] for v in episodes)


def test_neural_parameter_budget_nearly_matched():
    assert p60.model_params(p60.HybridPolicy())==1282
    assert p60.model_params(p60.CurrentOnlyPolicy())==1255
    assert abs(1255/1282-1)<.05


def test_history_and_action_credit_are_distinct_from_current_sensor():
    x=torch.ones((3,5,5))
    x[:,-1,:2]=torch.tensor([[-.5,-3.],[-.4,0.],[-.2,5.]])
    torch.manual_seed(24)
    m=p60.HybridPolicy().eval()
    raw=x.clone()
    nohistory=x.clone();nohistory[:,:4,:]=0.
    wrong=x.clone();wrong[:,:4,4]=torch.roll(wrong[:,:4,4],shifts=1,dims=1)
    assert torch.equal(raw[:,-1,:],nohistory[:,-1,:])
    assert torch.equal(raw[:,-1,:],wrong[:,-1,:])
    assert torch.equal(raw[:,:4,:4],wrong[:,:4,:4])
    assert torch.isfinite(p60.ablated_logits(m,x,'ttt_off')).all()
    assert torch.isfinite(p60.ablated_logits(m,x,'no_history')).all()
    with pytest.raises(ValueError,match='P60_UNKNOWN_ABLATION'):
        p60.ablated_logits(m,x,'oracle')


def test_cold_replay_same_weights_and_predictions():
    m=p60.HybridPolicy().eval()
    x=torch.zeros((5,5,5));x[:,-1,0]=-.5
    okay,sha=p60.cold_restore(m,x)
    assert okay and len(sha)==64


def test_no_tokenization_in_new_learning_path():
    doc=Path(p60.__file__).read_text()
    tree=ast.parse(doc)
    names={p.attr for p in ast.walk(tree) if isinstance(p,ast.Attribute)}
    assert not names.intersection({'Embedding','CrossEntropyLoss','tokenizer','vocabulary','token_ids'})
