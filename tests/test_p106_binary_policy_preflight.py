"""P106 binary authentic-action policy pure engineering — no simulator F6 execution."""
from dataclasses import replace
import hashlib
import math
from pathlib import Path

import pytest
import torch

from experiments import p105_f3_action_support_preflight as p105
from experiments import p106_binary_policy_preflight as p106


def test_p106_pinned_frozen_design_no_owner_authority_or_source_birth():
    f=p106.frozen()
    assert p106.FROZEN_COMMIT=="383298f9c5913b77d80ed77fd7c364158657498d"
    assert f["identity"]["base_pr"]==232
    assert f["identity"]["base_sha"]=="955d032ac21b71b9d5e02061ae9727600205d569"
    assert f["factual_task"]["actual_action_domain"]==[0,1]
    assert f["factual_task"]["source_birth_permitted"] is False
    assert f["prospective_F6_design_only"]["authorization"]=="NOT_GRANTED_BY_P105_OR_THIS_FREEZE"
    assert f["prospective_F6_design_only"]["replicate_groups"]==3
    assert f["implementation_contract"]["no_trial_workflow"] is True
    assert f["implementation_contract"]["no_experimental_source_birth"] is True
    assert f["implementation_contract"]["no_merge_or_deployment"] is True
    assert f["result_classification"].startswith("ENGINEERING_PREFLIGHT_ONLY")
    root=Path(__file__).resolve().parents[1]
    assert p105._git_blob((root/"experiments/p106_binary_policy_freeze.json").read_bytes())==p106.FROZEN_BLOB
    for rel,blob in p106.PINNED.items():
        assert p105._git_blob((root/rel).read_bytes())==blob


def test_p106_two_action_public_state_keys_exact_P80_order():
    for state in ((0.,0.,0.,0.),(.5,-1.25,.13,-.77)):
        keys=p106.action_pair(state)
        assert keys.shape==(2,16)
        assert keys.dtype==torch.float64
        assert torch.allclose(keys.norm(dim=-1),torch.ones(2,dtype=torch.float64),
                              atol=1e-15,rtol=0.)
        for a in (0,1):
            # P80 original actual state-conditioned key exact scalar operation order.
            raw=[1.]+[math.tanh(x/scale) for x,scale in zip(state,p106.SCALES)]
            raw += [float(a==0),float(a==1)] + [0.]*9
            factor=math.sqrt(math.fsum(v*v for v in raw))
            expected=torch.tensor([x/factor for x in raw],dtype=torch.float64)
            assert torch.equal(keys[a],expected)
            assert bool((keys[a,7:]==0).all())
    with pytest.raises(ValueError,match="P106_ONLY_TWO_CAUSAL_ACTIONS"):
        p106.public_action_key([0.,0.,0.,0.],2)
    for bad in (["bad",0,0,0], [float("nan"),0,0,0], [0,0,0],
                [0,0,float("inf"),0]):
        with pytest.raises(ValueError,match="P106_BAD_PUBLIC_PHYSICAL_STATE"):
            p106.action_pair(bad)


def test_p106_prior_reproduces_P101_16x16_RLS_and_policy_is_deterministic():
    s=p106.initial()
    assert s.weights.shape==s.precision.shape==(16,16)
    assert torch.equal(s.weights,torch.zeros((16,16),dtype=torch.float64))
    assert torch.equal(s.precision,torch.eye(16,dtype=torch.float64)*2)
    assert s.confirmed_count==0
    state=(.07,-.31,.05,.19)
    a=p106.action_pair(state)
    v=((a@s.precision)*a).sum(-1)
    assert p106.choose(s,state,"max_variance",trial_public_key="fixture-1",step=0)==int(torch.argmax(v))
    for policy in ("max_variance","hash_passive","cyclic"):
        assert p106.choose(s,state,policy,trial_public_key="fixture-1",step=4)==(
            p106.choose(s,state,policy,trial_public_key="fixture-1",step=4))
    with pytest.raises(ValueError,match="P106_POLICY_SCOPE"):
        p106.choose(s,state,"oracle",trial_public_key="fixture-1",step=0)
    with pytest.raises(TypeError):
        p106.choose(s,state,"max_variance",trial_public_key="fixture-1",step=0,
                    true_effect_of_unchosen_action=torch.ones(16))


def test_p106_Aoptimal_matches_classical_closed_form_with_PUBLIC_covariance():
    state=(.01,-.04,.03,.02)
    s=p106.initial()
    k=p106.action_pair(state)
    sig=torch.eye(16,dtype=torch.float64)/16
    got=p106.choose(s,state,"a_opt",sigma=sig,trial_public_key="fit-public",step=0)
    g=k@s.precision
    info=((g@sig)*g).sum(-1)/(1.+(g*k).sum(-1))
    assert got==int(torch.argmax(info))
    for index in (5,6):
        sigma=torch.eye(16,dtype=torch.float64)*0.001
        sigma[index,index]=100.
        expected=index-5
        assert p106.choose(s,state,"a_opt",sigma=sigma,
                           trial_public_key="fit-public",step=0)==expected
    for invalid in (
        torch.eye(16,dtype=torch.float32),
        torch.full((16,16),float("nan"),dtype=torch.float64),
        torch.diag(torch.tensor([-1.]+[1.]*15,dtype=torch.float64)),
        torch.zeros((16,16),dtype=torch.float64),
        torch.ones((16,16),dtype=torch.float64)  # symmetric PSD is valid; keep out
    )[:-1]:
        with pytest.raises(ValueError,match="P106_UNQUALIFIED_PUBLIC_FIT_COVARIANCE"):
            p106.choose(s,state,"a_opt",sigma=invalid,
                        trial_public_key="fit-public",step=0)


def _observed_effect_for_fixture(state,action):
    # This is a deterministic TEST-ONLY manufactured vector, never an external
    # simulator response. Pure enough to compare original RLS math and replay.
    k=p106.public_action_key(state,action)
    return torch.arange(1,17,dtype=torch.float64)*.01 + .025*k


def test_p106_commit_then_confirm_only_selected_factual_action_exact_P101_math():
    state=(.11,-.05,.002,.28)
    model=p106.initial()
    proposal=p106.commit(model,state,"max_variance",
                         trial_public_key="fixture:world-A",step=0)
    assert proposal.action in (0,1)
    assert proposal.event_number==1
    assert proposal.policy=="max_variance"
    assert proposal.model_state_sha256==p106.model_sha(model)
    effect=_observed_effect_for_fixture(state,proposal.action)
    k=p106.public_action_key(state,proposal.action)
    g=model.precision@k
    den=1.+k@g
    expected_W=model.weights+torch.outer(effect-model.weights@k,g)/den
    expected_P=model.precision-torch.outer(g,g)/den
    result=p106.confirmed_write(model,proposal,executed_action=proposal.action,
                               state_before=state,observed_effect=effect,
                               confirmed=True,receipt_ref="fixture:one")
    assert torch.equal(result.weights,expected_W)
    assert torch.equal(result.precision,expected_P)
    assert result.confirmed_count==1
    assert torch.equal(model.weights,torch.zeros_like(model.weights))
    assert result.journal_sha256!=model.journal_sha256
    assert torch.equal(p106.read(result,state,proposal.action),result.weights@k)
    with pytest.raises(ValueError,match="P106_UNKNOWN_OR_REPLAYED_PROPOSAL"):
        p106.confirmed_write(result,proposal,executed_action=proposal.action,
                            state_before=state,observed_effect=effect,
                            confirmed=True,receipt_ref="fixture:one-again")


def test_p106_negative_credit_missing_ack_replay_and_tampering_fail_closed():
    state=(0.,0.,.02,-.01)
    base=p106.initial()
    prop=p106.commit(base,state,"cyclic",trial_public_key="fixture:source",step=0)
    effect=_observed_effect_for_fixture(state,prop.action)
    cases=(
        dict(executed_action=1-prop.action,state_before=state,
             observed_effect=effect,confirmed=True,receipt_ref="fixture:bad"),
        dict(executed_action=prop.action,state_before=(.5,0.,0.,0.),
             observed_effect=effect,confirmed=True,receipt_ref="fixture:bad"),
        dict(executed_action=prop.action,state_before=state,
             observed_effect=torch.full((16,),float("nan"),dtype=torch.float64),
             confirmed=True,receipt_ref="fixture:bad"),
    )
    messages=("P106_WRONG_ACTION_EFFECT_CREDIT","P106_UNKNOWN_OR_REPLAYED_PROPOSAL",
              "P106_BAD_OBSERVED_EFFECT")
    for kwargs,code in zip(cases,messages):
        with pytest.raises(ValueError,match=code):
            p106.confirmed_write(base,prop,**kwargs)
    with pytest.raises(PermissionError,match="P106_UNCONFIRMED_EFFECT_DENIED"):
        p106.confirmed_write(base,prop,executed_action=prop.action,state_before=state,
                            observed_effect=effect,confirmed=False,receipt_ref="fixture:no")
    with pytest.raises(PermissionError,match="P106_ONLY_STATIC_TEST_FIXTURE_RECEIPTS"):
        p106.confirmed_write(base,prop,executed_action=prop.action,state_before=state,
                            observed_effect=effect,confirmed=True,
                            receipt_ref="ghrun:unapproved-source")
    for altered in (
        replace(prop,commitment_sha256="f"*64),
        replace(prop,public_step=7),
        replace(prop,action=1-prop.action),
        replace(prop,policy="oracle"),
    ):
        with pytest.raises(ValueError,match="P106_(TAMPERED_PROPOSAL_COMMITMENT|PROPOSAL_SCOPE_DRIFT)"):
            p106.confirmed_write(base,altered,executed_action=altered.action,
                                state_before=state,observed_effect=effect,
                                confirmed=True,receipt_ref="fixture:tampered")
    assert base.confirmed_count==0
    assert torch.equal(base.weights,p106.initial().weights)


def test_p106_cold_retention_and_consequence_update_only_after_receipt():
    start=p106.initial()
    state=(.1,.2,-.03,.4)
    proposal=p106.commit(start,state,"a_opt",sigma=torch.eye(16,dtype=torch.float64),
                         trial_public_key="fixture:trial",step=0)
    assert torch.equal(start.weights,p106.initial().weights)
    expected=p106.read(start,state,proposal.action).clone()
    snap=p106.snapshot(start)
    changed=p106.confirmed_write(start,proposal,executed_action=proposal.action,
                        state_before=state,
                        observed_effect=_observed_effect_for_fixture(state,proposal.action),
                        confirmed=True,receipt_ref="fixture:observed")
    assert not torch.equal(p106.read(changed,state,proposal.action),expected)
    assert torch.equal(p106.read(snap,state,proposal.action),expected)
    assert torch.equal(snap.weights,start.weights)
    assert torch.equal(snap.precision,start.precision)
    assert p106.model_sha(snap)==p106.model_sha(start)
    assert p106.model_sha(changed)!=p106.model_sha(start)


def test_p106_is_pure_never_imports_upstream_or_dispatches_external_effects():
    s=Path(p106.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "import causal_gym", "from causal_gym", "import requests",
        "import socket","subprocess.", "urllib.", "http.client",
        "env.do(", "env.reset(", "github_actions.dispatch",
    ):
        assert forbidden not in s
    assert "class Proposal" in s and "def confirmed_write" in s
    assert "ghrun:unapproved-source" not in s
