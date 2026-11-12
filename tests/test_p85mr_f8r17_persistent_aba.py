"""F8-R17: source-neutral actually trained native causal A→B→A regression."""
import inspect
import math

import pytest
import torch

from experiments import p85mr_f8r17_persistent_aba as x

torch.set_num_threads(1)


def test_frozen_world_source_and_strict_firstuse_denial():
    f=x.locked()
    assert f["stage"]=="FROZEN_BEFORE_IMPLEMENTATION_AND_NUMERICAL_RESULTS"
    assert f["models"]["total_model_fits"]==12
    assert f["outcome"]["score_worlds"]==12
    assert f["authority"]["external_source_firstuse"] is False
    assert f["authority"]["spent_f7_replay"] is False
    assert f["authority"]["independent_scientific_pass"] is False
    with pytest.raises(PermissionError,match="FIRSTUSE_NOT_AUTHORIZED"):
        x.refuse_original_source_firstuse()


def test_hidden_aba_schedule_and_no_regime_labels_in_learners():
    groups=[x.cohort(role) for role in ("fit","diagnostic","test")]
    assert [len(g) for g in groups]==[24,12,12]
    assert len({p["seed"] for group in groups for p in group})==48
    for split,worlds in zip(("fit","diagnostic","test"),groups):
        for ep in worlds:
            a,b=ep["switches"]
            assert ep["regimes"][:a] == (ep["initial_sign"],)*a
            assert ep["regimes"][a:b] == (-ep["initial_sign"],)*(b-a)
            assert ep["regimes"][b:] == (ep["initial_sign"],)*(12-b)
            assert ep["action"].shape==(12,16)
            assert ep["effect"].shape==(12,16)
            assert set(x.factual(ep))=={"action","effect"}
            assert bool(torch.isfinite(ep["action"]).all())
        assert worlds[0]["switches"]==((3,7) if split=="test" else (4,8))
    with pytest.raises(ValueError,match="UNREGISTERED_WORLD"):
        x.episode("test",12)
    with pytest.raises(ValueError,match="UNREGISTERED_WORLD"):
        x.episode("fit",True)
    with pytest.raises(ValueError,match="INVALID_HIDDEN_EVALUATOR_REGIME"):
        x.sign_at(1,12,(4,8))


def test_native_and_conventional_current_effect_strict_causal_mask():
    ep=x.episode("test",0)
    for arm in x.ARMS:
        model=x.r5.model_for(arm,985201).cpu().eval()
        pred=x.predictions(arm,model,ep)
        tampered={**ep,"effect":ep["effect"].clone()}
        tampered["effect"][6:]=2.5
        again=x.predictions(arm,model,tampered)
        assert torch.allclose(pred[:6],again[:6],atol=1e-6,rtol=0)
        assert torch.isfinite(again).all()
        with pytest.raises(ValueError,match="INVALID_HYPOTHETICAL_ACTION"):
            x.predict_action(arm,model,ep,2)


def test_actual_native_training_complete_falsifiable_and_cold_restore():
    r=x.run()
    assert r["classification"]=="SOURCE_NEUTRAL_NATIVE_TRAINED_DEV_ONLY"
    assert r["model_count"]==12 and r["total_optimizer_updates"]==1152
    assert r["synthetic_fit_selected_effects"]==288
    assert r["synthetic_test_factual_effects"]==144
    assert r["score_model_selected_synthetic_actions"]==144
    assert r["original_source_effects"]==0
    assert r["score_unfamiliar_switches"]==[3,7]
    assert not r["spent_F7_replayed"] and not r["scientific_pass"]
    assert r["trained_controls_convergence_independently_qualified"] is False
    assert len(r["candidate_wins"])==3
    assert r["candidate_native_dev_gate_pass"] in (True,False)
    assert {z["arm"] for z in r["models"]}==set(x.ARMS)
    assert {z["seed"] for z in r["models"]}==set(x.SEEDS)
    for row in r["models"]:
        sc=row["score"]
        assert row["optimizer_updates"]==96
        assert row["initial_weight_sha256"]!=row["trained_weight_sha256"]
        assert row["cold_restore_parity"] is True
        assert row["parameter_count"]>0
        assert math.isfinite(row["train_seconds"])
        assert all(math.isfinite(sc[k]) for k in
                   ("mean_physical_mse","mean_action_regret",
                    "delayed_history_negative_mse"))
        assert len(sc["perworld_mse"])==12
        assert len(sc["perworld_action_regret"])==12
        assert sc["model_selected_toy_actions"]==12
        assert sc["hypothetical_model_queries"]==24
        assert all(math.isfinite(v) for v in sc["phases"].values())
    x.retain_ci_receipt(r)


def test_source_is_not_external_simulator_or_runner():
    src=inspect.getsource(x)
    for forbidden in ("import causal_gym", "from causal_gym",
                      "env.reset(", "env.do(", "ctf_do(", "workflow_dispatch(",
                      "subprocess.run(", "requests.get(", "urlopen("):
        assert forbidden not in src
