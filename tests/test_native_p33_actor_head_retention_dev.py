"""P33 prospective guard tests: no real source world or seed first use."""
import hashlib
import io
import json
from pathlib import Path

import pytest
import torch

from scripts import native_p33_actor_head_retention_dev as p33
from scripts import native_p32_component_damage_dev as p32
from scripts import native_p30_reward_motion_ppo_dev as p30
from scripts import native_p20_reward_only_ppo_dev as p20


def test_frozen_protocol_and_parent_source_identity(monkeypatch):
    protocol=p33.freeze_contract()
    assert protocol["parent"]["exact_source_sha"]==p33.PARENT
    assert tuple(protocol["training"]["model_initialization_seeds"])==p33.SEEDS
    assert protocol["source"]["protected_scientific_first_use"] is False
    monkeypatch.delenv("NATIVE_P33_DEV_ONLY",raising=False)
    with pytest.raises(RuntimeError,match="OPT_IN_REQUIRED"):
        p33.qualify_local_source()


def test_scoped_new_seeds_leave_p32_and_p30_originals_bit_exact():
    before=(p32.SEEDS,p32.TRAIN_BASE,p32.EVAL_SEEDS,p32.TRAIN_ARMS,
            p32.EVAL_ARMS,p30.SEEDS)
    with pytest.raises(RuntimeError,match="artificial-negative-control"):
        with p33.scoped_fresh_source():
            assert p32.SEEDS==p33.SEEDS
            assert p30.SEEDS==p33.SEEDS
            assert p32.TRAIN_BASE==p33.TRAIN_BASE
            assert p32.EVAL_SEEDS==p33.EVAL_SEEDS
            assert p32.EVAL_ARMS==p33.EVAL_ARMS
            assert p32.TRAIN_ARMS==p33.TRAIN_ARMS
            assert all(s not in before[0] for s in p32.SEEDS)
            raise RuntimeError("artificial-negative-control")
    assert (p32.SEEDS,p32.TRAIN_BASE,p32.EVAL_SEEDS,p32.TRAIN_ARMS,
            p32.EVAL_ARMS,p30.SEEDS)==before
    assert p32.frozen()["phase"]=="FROZEN_BEFORE_IMPLEMENTATION_AND_PUBLIC_SOURCE_USE"


def _train_record(seed):
    return {
        "arm":"actor_head_only_ppo",
        "model_seed":seed,
        "cold_exact_logits_and_values":True,
        "real_original_train_source_actions":49152,
        "original_ppo_minibatch_updates":576,
        "original_actual_action_consequence_sha256":"b"*64,
    }


def _means(actor,full):
    return {condition:{
        "actor_head_only_ppo":actor,
        "full_actor_recurrent_ppo":full,
        "finite_difference_public_two_frame":200.0,
        "critic_only_frozen_policy":25.0,
        "initial_no_update":25.0,
        "seeded_random":20.0,
    } for condition in p33.CONDITIONS}


def test_unaccepted_actor_checkpoint_cold_load_and_no_overwrite(tmp_path):
    seed=p33.SEEDS[0]
    torch.manual_seed(seed)
    model=p30.RewardGRU10()
    root=tmp_path/"candidate"
    root.mkdir()
    with p33.scoped_fresh_source():
        m=p33.publish_unaccepted_weights(
            root,seed,model,"a"*40,_train_record(seed),_means(200.,100.))
        assert m["model_seed"]==seed
        assert m["checkpoint_optimizer_included"] is False
        assert m["model_promoted"] is False
        assert m["cold_weights_and_logits_bit_exact"] is True
        actual=(root/m["weights_filename"]).read_bytes()
        assert hashlib.sha256(actual).hexdigest()==m["weights_sha256"]
        cold=p30.RewardGRU10()
        cold.load_state_dict(torch.load(
            io.BytesIO(actual),weights_only=True,map_location="cpu"),strict=True)
        with torch.no_grad():
            a=model(p33._policy_sentinel(seed))
            b=cold(p33._policy_sentinel(seed))
        assert all(torch.equal(x,y) for x,y in zip(a,b))
        with pytest.raises(FileExistsError):
            p33.publish_unaccepted_weights(
                root,seed,model,"a"*40,_train_record(seed),_means(200.,100.))
    assert not (root/"accepted").exists()
    assert not (root/"latest").exists()


def test_candidate_refuses_accepted_pointer_symlink_and_wrong_training(tmp_path):
    model=p30.RewardGRU10()
    seed=p33.SEEDS[0]
    with p33.scoped_fresh_source():
        root=tmp_path/"empty"
        root.mkdir()
        (root/"latest").write_text("pointer-not-allowed")
        with pytest.raises(RuntimeError,match="ACCEPTED_POINTER_FORBIDDEN"):
            p33.publish_unaccepted_weights(
                root,seed,model,"a"*40,_train_record(seed),_means(160.,100.))
        (root/"latest").unlink()
        bad=_train_record(seed);bad["real_original_train_source_actions"]=0
        with pytest.raises(RuntimeError,match="TRAINING_DRIFT"):
            p33.publish_unaccepted_weights(
                root,seed,model,"a"*40,bad,_means(160.,100.))
        alias=tmp_path/"alias"
        try:
            alias.symlink_to(root,target_is_directory=True)
        except (OSError,NotImplementedError):
            pass
        else:
            with pytest.raises(RuntimeError,match="UNTRUSTED"):
                p33.publish_unaccepted_weights(
                    alias,seed,model,"a"*40,_train_record(seed),_means(160.,100.))


def test_frozen_results_reject_single_seed_positive_and_report_conventional():
    records=[{
        "model_seed":s,
        "real_source_reward_means_by_condition":
            _means(200. if i!=2 else 120.,100.),
        "critic_only_source_actions_equal_initial":True,
    } for i,s in enumerate(p33.SEEDS)]
    out=p33.frozen_development_outcome(records)
    assert out["per_fresh_seed_frozen_gates"]==[True,True,False]
    assert out["all_three_seed_actor_only_frozen_gate"] is False
    assert out["scientific_architecture_superiority"] is False
    assert out["strong_conventional_control_reported"] is True
    assert out["protected_scientific_first_use"]=="NOT_RUN"
    records[-1]["real_source_reward_means_by_condition"]=_means(200.,100.)
    assert p33.frozen_development_outcome(records)[
        "all_three_seed_actor_only_frozen_gate"] is True
    with pytest.raises(RuntimeError,match="DUPLICATE_MODEL_SEED"):
        p33.frozen_development_outcome(records[:1]+records[:1]+records[2:])


def test_no_source_firstuse_from_running_untrusted_checkpoint(tmp_path):
    source=tmp_path/"source";source.mkdir()
    with pytest.raises(RuntimeError,match="ALREADY_EXISTS_NO_REPLAY"):
        p33.run(tmp_path,tmp_path,tmp_path,source)
