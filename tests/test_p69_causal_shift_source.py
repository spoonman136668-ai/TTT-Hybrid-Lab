"""P69 source-contract tests. FIT/VAL engineering only; never score OOD."""
import importlib.util
from pathlib import Path
import pytest
import torch

src=Path(__file__).resolve().parents[1]/"experiments"/"p69_causal_shift_source.py"
spec=importlib.util.spec_from_file_location("p69_causal_shift_source",src)
p69=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p69)


def test_fit_repeatability_and_independent_worlds():
    w=p69.make_world("fit",0)
    assert p69.digest_world(w)==p69.digest_world(p69.make_world("fit",0))
    assert p69.digest_world(w)!=p69.digest_world(p69.make_world("fit",1))
    assert p69.digest_world(w)!=p69.digest_world(p69.make_world("val",0))
    assert w["action"].shape==(96,16) and w["effect"].shape==(96,16)


def test_frozen_pre_post_interventions_and_hidden_change():
    for split in ("fit","val"):
        w=p69.make_world(split,0)
        at=w["boundary_audit"]
        assert 40<=at<=56
        assert len(w["changed_roles_audit"])==4
        for role in range(8):
            assert int((w["role_audit"][:at]==role).sum())==3
            assert int((w["role_audit"][at:]==role).sum())==3
        assert int(w["target_audit"].sum())==48
        for r in range(8):
            before=w["role_changed_audit"][w["role_audit"]==r]
            assert before.unique().numel()==1
        assert bool(w["query_changed_audit"])==(w["query_role_audit"] in w["changed_roles_audit"])


def test_source_signs_balanced_per_pre_post_pair():
    w=p69.make_world("fit",12)
    basis_actions=w["action"]
    assert torch.isfinite(basis_actions).all()
    assert bool((basis_actions.norm(dim=-1)-1).abs().max()<1e-5)
    assert torch.isfinite(w["effect"]).all()


def test_factual_history_excludes_future_and_oracle_labels():
    batch=p69.make_batch("fit",8)
    facts=p69.factual_history(batch)
    assert set(facts)=={"action","effect"}
    assert "query_action" not in facts and "boundary_audit" not in facts
    assert batch["query_action"].shape==(8,16)
    assert not torch.equal(batch["query_action"],batch["action"][:,-1,:])


def test_split_selection_fails_closed():
    for split,index in (("train",0),("fit",-1),("fit",512),("ood_99",0)):
        with pytest.raises(ValueError,match="P69_BAD_SOURCE_SELECTOR"):
            p69.make_world(split,index)
    with pytest.raises(ValueError,match="P69_BAD_BATCH"):
        p69.make_batch("val",193)


def test_cold_source_prefix_is_independent_of_requested_batch_size():
    a=p69.make_batch("fit",8)
    b=p69.make_batch("fit",9)
    assert torch.equal(a["action"],b["action"][:8])
    assert torch.equal(a["effect"],b["effect"][:8])


def test_frozen_fit_val_no_index_shortcut_or_duplicate_query():
    for split in ("fit","val"):
        batch=p69.make_batch(split,64)
        audit=p69.source_integrity(batch)
        assert audit["target_quartile_pass"],audit
        assert audit["index_target_pass"],audit
        assert audit["changed_role_index_pass"],audit
        assert audit["query_duplicate_pass"],audit


def test_sampled_world_input_permutation_does_not_expose_metadata():
    w=p69.make_world("fit",17)
    batch={"action":w["action"][None],"effect":w["effect"][None],"query_action":w["query_action"][None]}
    assert set(p69.factual_history(batch))=={"action","effect"}


def test_no_model_or_science_execution_entrypoint():
    source=src.read_text(encoding="utf8")
    assert "def study(" not in source
    assert "optimizer" not in source
    assert "torch.save" not in source
    assert "if __name__" not in source
