"""R41 frozen history-conditioned Native incremental causal challenge."""
import inspect
import math

import pytest
import torch

from experiments import p85mr_f8r41_history_conditional as x

torch.set_num_threads(1)


def test_prospective_frozen_design_authority_and_source_lineage():
    f=x.locked()
    assert f["stage"]=="FROZEN_BEFORE_R41_IMPLEMENTATION_TRAINING_AND_RESULTS"
    assert f["parent"]["head"]=="c96316f33bcc0f89b6f032d87a9a390f9db6922a"
    assert f["models"]["instances"]==6
    assert f["preregistered_score"]["native_seed_both_panels_pass_min"]==2
    assert not any(f["authority"].values())
    with pytest.raises(PermissionError,match="EXTERNAL_FIRSTUSE"):
        x.refuse_external_firstuse()


def test_hidden_history_gate_not_observable_before_selected_B_marker():
    for panel in ("fit","dev","ood"):
        a=x.world(panel,0,0,1,-1)
        b=x.world(panel,0,0,1,1)
        aa,ae=x._factual_history(a)
        ba,be=x._factual_history(b)
        assert torch.equal(aa,ba)
        assert torch.equal(ae[:5],be[:5])
        assert torch.equal(ae[5:5+x.B_MARK[panel]],
                           be[5:5+x.B_MARK[panel]])
        assert not torch.equal(ae[5+x.B_MARK[panel]],
                               be[5+x.B_MARK[panel]])
        for i in range(5,len(ae)):
            if i!=5+x.B_MARK[panel]:
                assert torch.equal(ae[i],be[i])
        assert torch.count_nonzero(x.candidates(a)[1][:,-1,:])==0
        if panel=="ood":
            assert aa.shape==(21,16)
        else:
            assert aa.shape==(13,16)


def test_factual_training_pairs_selected_outcomes_and_balanced_gate():
    fit,labels=x.fit_cohort()
    assert len(fit)==768 and len(labels)==384
    assert len(labels[:256])==256 and len(labels[256:])==128
    assert [labels[:256].count(i) for i in (0,1)]==[128,128]
    assert [labels[256:].count(i) for i in (0,1)]==[64,64]
    a,e,y=x._training_batch(fit,labels,list(range(16)))
    assert a.shape==e.shape==(32,14,16)
    assert y.shape==(16,)
    for i in range(0,32,2):
        assert torch.equal(a[i,:-1],a[i+1,:-1])
        assert torch.equal(e[i,:-1],e[i+1,:-1])
        assert not torch.equal(a[i,-1],a[i+1,-1])
    with pytest.raises(ValueError,match="INVALID_TRAIN_PAIRS"):
        x._training_batch(fit,labels,[256]*16)


def test_future_return_effect_mask_and_recurrent_baseline_exact_seed_alignment():
    binder=x.r38.TrainableBinding(x.SEEDS[0])
    native=x.HistoryAdapter("native_history",x.SEEDS[0],binder)
    control=x.HistoryAdapter("history_only",x.SEEDS[0],binder)
    assert x._weight_hash(native.recurrent.memory)==x._weight_hash(
        control.recurrent.memory)
    assert x._weight_hash(native.recurrent.readout)==x._weight_hash(
        control.recurrent.readout)
    assert control.recurrent.backbone is None
    w=x.world("dev",0,0,1,1)
    aa,ee=x.candidates(w)
    with torch.no_grad():
        native.eval()
        normal=native(aa,ee)
        poison=ee.clone()
        poison[:,-1,:]=17.
        assert torch.allclose(normal,native(aa,poison),atol=0,rtol=0)
        frozen=native(aa,ee,zero_history=True)
        assert torch.allclose(frozen,
                              x.r38.predict(binder,"binder",aa,ee),
                              atol=1e-7,rtol=1e-7)


def test_analytic_factual_positive_control_and_B_erasure_negative():
    fit,_=x.fit_cohort()
    for panel in x.PANELS:
        ww=x.worlds(panel)
        assert len(ww)==128
        assert all(ww[i]["q_evaluator_only"]==ww[i+1]["q_evaluator_only"]
                   and ww[i]["gate_evaluator_only"]==-ww[i+1]["gate_evaluator_only"]
                   for i in range(0,128,2))
        reference=x._fitted_conventional_baseline(fit,ww,panel)
        assert reference["history_gate_pair_flips"]==64
        assert reference["mean_regret"]==0.
        for w in ww[:2]:
            _,e=x.candidates(w,erase_b=True)
            assert torch.count_nonzero(e[:,5+x.B_MARK[panel]])==0
            _,swapped=x.candidates(w,opposite_gate_measurement=True)
            _,original=x.candidates(w)
            assert not torch.equal(swapped[:,5+x.B_MARK[panel]],
                                   original[:,5+x.B_MARK[panel]])


def test_full_real_neural_training_and_frozen_nonpromotional_receipt():
    data=x.run()
    assert data["schema"]==x.SCHEMA
    assert data["factual_fit_episodes"]==768
    assert data["train_factual_pairs"]==256
    assert data["heldout_factual_pairs"]==128
    assert data["original_R38_pretrain_optimizer_updates"]==1024
    assert data["new_native_control_optimizer_updates"]==2304
    assert len(data["models"])==6
    assert data["judgment"]["factual_conventional_positive_pass"]
    assert isinstance(data["judgment"]["frozen_native_incremental_development_pass"],bool)
    assert not data["independent_scientific_pass"]
    assert not data["spent_F7_replay"]
    assert data["original_external_effects"]==0
    assert data["deployment"]=="NONE"
    for m in data["models"]:
        assert m["optimizer_updates"]==384
        assert m["native_initial_sha256"]!=m["native_final_sha256"]
        assert m["cold_restore_action_identity"]
        assert [a["updates"] for a in m["validation_checkpoints"]]==[192,384]
        assert all(math.isfinite(a["heldout"]["choice_CE"])
                   for a in m["validation_checkpoints"])
        for p in x.PANELS:
            for name in ("active","erase_B","swap_B","erase_fifth",
                         "zero_history","zero_backbone"):
                score=m["panels"][p][name]
                assert len(score["choices"])==128
                assert len(score["pergroup_regret"])==16
                assert 0<=score["history_gate_pair_flips"]<=64
                assert math.isfinite(score["mean_regret"])
    x.retain_ci_receipt(data)


def test_training_does_not_know_q_gate_evaluator_or_unchosen_return_effect():
    for fn in (x.HistoryAdapter.forward,x._training_batch,
               x._validation,x._train):
        code=inspect.getsource(fn)
        for bad in ("q_evaluator_only","gate_evaluator_only",
                    "_regrets(","_law(","oracle_label"):
            assert bad not in code
    full=inspect.getsource(x)
    for banned in ("workflow_dispatch(","subprocess.run(","requests.get(",
                   "urlopen(","import causal_gym","from causal_gym"):
        assert banned not in full
