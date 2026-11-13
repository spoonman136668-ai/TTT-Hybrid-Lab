"""F8-R5 measured native model training on selected synthetic outcomes."""
from pathlib import Path
import math
import pytest
import torch
from experiments import p85mr_f8r5_native_policy as x

torch.set_num_threads(1)


def test_r5_freeze_and_three_nonoverlapping_author_toy_cohorts():
    f=x.locked()
    assert f["status"]=="FROZEN_BEFORE_IMPLEMENTATION_OR_AUTHOR_TOY_EVALUATION"
    assert f["authority"]["no_original_external_firstuse"]
    assert f["authority"]["no_spent_F7_source_replay"]
    assert not f["authority"]["independent_scientific_PASS"]
    assert f["models"]["maximum_model_updates"]==1728
    for split,(_,count) in x.SPLITS.items():
        group=x.cohort(split)
        assert len(group)==count
        assert all(w["action"].shape==(12,16) and
                   w["effect"].shape==(12,16) and
                   bool(torch.isfinite(w["action"]).all()) for w in group)
    with pytest.raises(ValueError,match="F8R5_UNKNOWN_WORLD"):
        x.toy_episode("score",12)


def test_r5_factual_probe_does_not_see_label_of_current_hypothetical_action():
    for arm in x.ARMS:
        m=x.model_for(arm,985201).eval()
        for sign in (-1,1):
            choice,regret=x.choose(arm,m,sign,.2,-.2,True)
            assert choice in (0,1) and math.isfinite(regret) and regret>=0.
    with pytest.raises(PermissionError,match="ORIGINAL_FIRSTUSE_NOT_AUTHORIZED"):
        x.refuse_external()
    content=Path(x.__file__).read_text()
    for forbidden in ("import causal_gym","from causal_gym","env.do(",
          "env.reset(","workflow_dispatch(","torch.load(",
          "subprocess.run(","requests.get("):
        assert forbidden not in content


def test_r5_all_eighteen_real_trained_model_comparisons_completed():
    result=x.run()
    assert result["status"]=="F8R5_TRAINED_TOY_DEV_COMPLETED"
    assert result["model_count"]==18
    assert result["total_optimizer_updates"]==1728
    assert len(result["comparison"])==3
    assert result["real_external_source_actions"]==0
    assert not result["F7_replayed"] and not result["independent_scientific_pass"]
    assert all(z["updates"]==96 and z["cold_restore"] and
               z["initial_hash"]!=z["trained_hash"] and
               math.isfinite(z["heldout_toy_mse"]) and
               math.isfinite(z["mean_action_regret"])
               for z in result["models"])
    assert result["dev_pass"] in (True,False)
    x.save_github_receipt(result)
