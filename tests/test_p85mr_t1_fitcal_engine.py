"""P85-MR-T1 independent engineering fixtures, zero CartPoleWind first-use."""
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

EXPERIMENTS=Path(__file__).resolve().parents[1]/"experiments"
sys.path.insert(0,str(EXPERIMENTS))

from p80mr_causalgym_trace_adapter import _action_key, _observed_effect
from p85mr_dynamics_baselines import KINDS, SEEDS, new_baseline, prequential, component_errors
from p85mr_t1_fitcal_engine import (
    BATCH, MAX_UPDATES, SAMPLE_SEED, _best_checkpoint, _check_fit_cal,
    _episodes, _padded, _physical_loss, calibration_mse,
    frozen_indices, full_external_fitcal, toy_diagnostic,
)

torch.set_num_threads(1)


def toy_episodes(count=6, *, offset=0):
    items=[]
    for n in range(count):
        ix=n+offset
        length=2+(ix%5)
        state=[.001*(ix+1), .02, .002, .01]
        actions=[]
        effects=[]
        for t in range(length):
            direction=(ix+t)%2
            post=[
                state[0] + (.04 if direction else -.02),
                state[1] + (.02 if direction else -.01),
                state[2] + (.003 if direction else -.001),
                state[3] + (.008 if direction else -.004),
            ]
            actions.append(_action_key(state,direction))
            effects.append(_observed_effect(state,post,1.0+.01*t,False,False))
            state=post
        items.append({"action":torch.tensor(actions,dtype=torch.float32),
                      "effect":torch.tensor(effects,dtype=torch.float32)})
    return items


def test_protocol_frozen_before_source_core_implementation():
    src=(EXPERIMENTS/"p85mr_t1_fitcal_engine_freeze.json").read_bytes()
    expected="7745855deaafdb80117e68ba437af27059005524"
    got=hashlib.sha1(f"blob {len(src)}\0".encode()+src).hexdigest()
    assert got==expected
    c=json.loads(src)
    assert c["parent"]["commit"]=="2656f85cda414c2617a1dd5724c126f3ba860b26"
    assert c["classification"]=="ENGINEERING_TOY_ONLY_NO_EXTERNAL_SOURCE"
    assert c["safety"]["first_use_authorized"] is False
    assert c["training"]["max_updates"]==800


def test_exact_global_sampler_shared_among_all_12_model_seeds():
    a=frozen_indices(128)
    b=frozen_indices(128)
    assert torch.equal(a,b)
    assert a.shape==(MAX_UPDATES,BATCH)
    assert a.min()>=0 and a.max()<128
    assert SAMPLE_SEED==985299
    for kind in KINDS:
        for seed in SEEDS:
            assert torch.equal(a,frozen_indices(128))


def test_padded_variable_length_worlds_keep_only_real_t_after_first():
    items=_episodes(toy_episodes(5))
    batch,mask=_padded(items)
    assert batch["action"].shape==(5,6,16)
    assert [int(x) for x in mask.sum(1)]==[1,2,3,4,5]
    assert not bool(mask[:,0].any())
    for i,world in enumerate(items):
        t=len(world["action"])
        assert torch.equal(batch["action"][i,:t],world["action"])
        assert torch.equal(batch["effect"][i,:t],world["effect"])
        assert torch.equal(batch["action"][i,t:,0],torch.ones(6-t))
        assert torch.equal(batch["effect"][i,t:],torch.zeros(6-t,16))


@pytest.mark.parametrize("kind", KINDS)
def test_valid_pred_and_loss_ignore_right_padding(kind):
    data=toy_episodes(4)
    m=new_baseline(kind,SEEDS[0])
    a,mask=_padded(data)
    first_loss,_=_physical_loss(m,a,mask)
    changed={"action":a["action"].clone(),"effect":a["effect"].clone()}
    for i,world in enumerate(data):
        t=len(world["action"])
        if t<changed["action"].shape[1]:
            changed["action"][i,t:,0] = 0.8
            changed["action"][i,t:,1] = 0.6
            changed["effect"][i,t:, :7] = 11.0
    second_loss,_=_physical_loss(m,changed,mask)
    assert torch.allclose(first_loss,second_loss,rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize("kind", KINDS)
def test_padded_batch_valid_predictions_match_unpadded_prefix(kind):
    data=toy_episodes(4)
    m=new_baseline(kind,SEEDS[0]).eval()
    padded,_=_padded(data)
    with torch.no_grad():
        out=prequential(m,padded)
        for i,world in enumerate(data):
            short=prequential(m,{"action":world["action"][None],
                                "effect":world["effect"][None]})
            assert torch.allclose(out[i,:len(world["action"])],
                                  short[0],atol=2e-6,rtol=3e-5)


def test_cal_weights_worlds_equally_despite_different_lengths():
    worlds=toy_episodes(5,offset=100)
    m=new_baseline("linear7",SEEDS[0]).eval()
    got=calibration_mse(m,worlds)
    manual=[]
    zero=[]
    for ep in worlds:
        f={"action":ep["action"][None],"effect":ep["effect"][None]}
        with torch.no_grad():
            predictions=prequential(m,f)
            errors=component_errors(predictions,f["effect"])["deltas"][0,1:]
        manual.append(float(errors.mean()))
        zero.append(float(f["effect"][0,1:,:4].square().mean()))
    assert got["cal_delta_mse"] == pytest.approx(sum(manual)/len(manual),rel=1e-6)
    assert got["zero_cal_delta_mse"] == pytest.approx(sum(zero)/len(zero),rel=1e-6)


def test_finite_cal_does_not_backprop_or_mutate_parameters():
    worlds=toy_episodes(5,offset=20)
    m=new_baseline("mlp7",SEEDS[0])
    before=[x.detach().clone() for x in m.parameters()]
    assert all(p.grad is None for p in m.parameters())
    scores=calibration_mse(m,worlds)
    assert scores["cal_delta_mse"]>=0
    assert all(p.grad is None for p in m.parameters())
    assert all(torch.equal(x,y) for x,y in zip(before,m.parameters()))


@pytest.mark.parametrize("kind", KINDS)
def test_toy_short_training_per_kind_checkpoint_restore_and_no_science(kind):
    fit=toy_episodes(5)
    cal=toy_episodes(4,offset=15)
    model, receipt=toy_diagnostic(kind,SEEDS[0],fit,cal,updates=2,every=2)
    assert receipt["kind"]==kind
    assert receipt["selected_update"]==2
    assert receipt["checkpoint_count"]==1
    assert receipt["steps"]==2
    assert receipt["full_800_update_protocol_completed"] is False
    assert receipt["fit_cal_external_source_accessed"] is False
    assert receipt["ood_accessed"] is False
    assert receipt["science_pass"] is False
    assert receipt["best_cal_delta_mse"]==pytest.approx(
        calibration_mse(model,cal)["cal_delta_mse"],rel=1e-6)


def test_toy_fitcal_deterministic_receipt_and_sampling_same_across_arms():
    fit=toy_episodes(6)
    cal=toy_episodes(4,offset=17)
    _,a=toy_diagnostic("linear7",SEEDS[0],fit,cal,updates=2,every=2)
    _,b=toy_diagnostic("linear7",SEEDS[0],fit,cal,updates=2,every=2)
    _,c=toy_diagnostic("mlp7",SEEDS[1],fit,cal,updates=2,every=2)
    assert a==b
    assert a["sample_index_sha256"]==c["sample_index_sha256"]
    assert a["best_cal_delta_mse"]>=0


def test_best_checkpoint_keeps_earliest_equal_metric():
    assert _best_checkpoint(.01, .01, 50, 100)==(.01,50)
    assert _best_checkpoint(.009, .01, 50, 100)==(.009,100)
    assert _best_checkpoint(.01, float("inf"), None,50)==(.01,50)


def test_cross_fit_cal_exact_episode_reuse_fail_closed():
    fit=toy_episodes(4)
    cal=toy_episodes(4,offset=15)
    cal[0]={k:v.clone() for k,v in fit[0].items()}
    with pytest.raises(ValueError,match="P85T1_FIT_CAL_CONTENT_LEAKAGE"):
        _check_fit_cal(fit,cal)


def test_duplicate_episode_and_privileged_field_rejected():
    fit=toy_episodes(4)
    cal=toy_episodes(3,offset=20)
    fit[-1]=fit[0]
    with pytest.raises(ValueError,match="P85T1_DUPLICATE_EPISODE"):
        _check_fit_cal(fit,cal)
    bad=toy_episodes(3)
    bad[0]["hidden_wind"]=torch.ones(2)
    with pytest.raises(ValueError,match="P85T1_NONFACTUAL_EPISODE"):
        _episodes(bad)


@pytest.mark.parametrize("invalid",[
    lambda a: a["action"].mul_(.1),
    lambda a: a["effect"].__setitem__((0,8),.2),
    lambda a: a.update(query_action=torch.ones(16)),
    lambda a: a["action"].__setitem__((0,0),float("nan")),
])
def test_bad_factual_input_fail_closed(invalid):
    episodes=toy_episodes(3)
    invalid(episodes[0])
    with pytest.raises(ValueError,match="P85MR_|P69_|P85T1_"):
        _episodes(episodes)


def test_padded_first_event_and_wrong_mask_fail_closed():
    batch,valid=_padded(toy_episodes(4))
    valid[:,0]=True
    with pytest.raises(ValueError,match="P85T1_MASK_LEAKAGE"):
        _physical_loss(new_baseline("linear7",SEEDS[0]),batch,valid)


def test_full_external_first_use_unconditionally_rejected():
    class Poison:
        def __getattribute__(self,key):
            raise AssertionError("SOURCE_TOUCHED")
    with pytest.raises(PermissionError,match="P85T1_OWNER_EXTERNAL_FIRST_USE_REQUIRED"):
        full_external_fitcal(Poison(), source_kind="toy_engineering", owner_approved=True)


def test_dev_bounds_cannot_be_mistaken_for_full_protocol():
    fit=toy_episodes(4)
    cal=toy_episodes(4,offset=20)
    for updates,every in [(800,50),(1,1),(7,1),(3,2),(4,0)]:
        with pytest.raises(ValueError,match="P85T1_ENGINEERING_BOUNDS"):
            toy_diagnostic("linear7",SEEDS[0],fit,cal,updates=updates,every=every)


def test_no_simulator_first_use_or_files_or_ood_in_training_core():
    txt=(EXPERIMENTS/"p85mr_t1_fitcal_engine.py").read_text()
    for forbidden in ("import gymnasium","import causal_gym",
                      "env.reset(","env.do(","requests.","subprocess",
                      "torch.save(","torch.load(","open(","if __name__",
                      "science_ood_0","science_ood_1","science_ood_2",
                      "runner.dispatch","schedule.create"):
        assert forbidden not in txt,forbidden
