"""P21 distinct-process checkpoint custody, deterministic cold weights and negatives."""
from pathlib import Path
import io
import json
import subprocess
import sys
import pytest
import torch
from scripts import native_p18_real_cartpole_factual_dev as p18
from scripts import native_p21_process_cold_retention_dev as p21


ROOT=Path(__file__).resolve().parents[1]


class FakeEnv:
    def reset(self,seed=None):
        self.t=0
        self.pos=0.015 if seed%2 else -0.015
        return [0.,0.04,self.pos,0.09],{}
    def step(self,action):
        self.t+=1
        self.pos+=0.002*(2*action-1)
        return [0.,0.04,self.pos,0.09],1.,self.t==3,False,{}
    def close(self):pass


class FakeGym:
    def make(self,name,render_mode=None):
        assert name=="CartPole-v1" and render_mode is None
        return FakeEnv()


def seeded_entry(checkpoint_dir,seed,kind):
    torch.manual_seed(seed)
    net=p21.p19.new_model("history_gru")
    file=checkpoint_dir/p21.checkpoint_name(kind,seed)
    with file.open("xb") as w:
        torch.save(net.state_dict(),w)
    return {
        "architecture":kind,"model_seed":seed,"source_sha":p21.P20_PARENT,
        "filename":file.name,
        "checkpoint_archive_sha256":p21.sha256(file.read_bytes()),
        "checkpoint_state_sha256":p18.model_sha(net),
        "sentinel_logits_sha256":p21.probe_sha(net,kind),
    }


def test_frozen_sources_and_prospective_evaluation_are_distinct():
    d=p21.frozen()
    assert d["schema"]=="ttt-hybrid.native-p21.process-death-cold-retention-frozen.v1"
    assert d["parent"]["source_sha"]==p21.P20_PARENT
    assert d["science"]["first_use_authorized"] is False
    assert d["science"]["no_optimizer_rng_state_continuation_claim"] is True
    assert len(p21.EVAL_SEEDS)==8
    assert set(p21.EVAL_SEEDS).isdisjoint(p21.p20.EVAL_SEEDS)
    assert set(p21.EVAL_SEEDS).isdisjoint(p21.p19.EVAL_SEEDS)
    assert len(p21.MODEL_SEEDS)==3


def test_scientific_owner_fence_requires_explicit_development_flag(monkeypatch):
    monkeypatch.delenv("NATIVE_P21_DEV_ONLY",raising=False)
    with pytest.raises(RuntimeError,match="P21_SCIENCE_FIRST_USE_DENIED"):
        p21.require_dev()
    monkeypatch.setenv("NATIVE_P21_DEV_ONLY","1")
    p21.require_dev()


def test_checkpoint_archive_hash_wrong_source_and_wrong_seed_denied(tmp_path):
    entry=seeded_entry(tmp_path,p21.MODEL_SEEDS[0],"full_history_gru")
    restored=p21.model_from_checkpoint(tmp_path,entry,"full_history_gru",p21.MODEL_SEEDS[0])
    assert p18.model_sha(restored)==entry["checkpoint_state_sha256"]
    assert p21.probe_sha(restored,"full_history_gru")==entry["sentinel_logits_sha256"]
    with pytest.raises(RuntimeError,match="P21_CHECKPOINT_CUSTODY_MISMATCH"):
        p21.model_from_checkpoint(tmp_path,{**entry,"source_sha":"bad"},"full_history_gru",p21.MODEL_SEEDS[0])
    with pytest.raises(RuntimeError,match="P21_CHECKPOINT_CUSTODY_MISMATCH"):
        p21.model_from_checkpoint(tmp_path,entry,"full_history_gru",p21.MODEL_SEEDS[1])
    raw=bytearray((tmp_path/entry["filename"]).read_bytes())
    raw[len(raw)//2]^=1
    (tmp_path/entry["filename"]).write_bytes(bytes(raw))
    with pytest.raises(RuntimeError,match="P21_CHECKPOINT_ARCHIVE_DIGEST_MISMATCH"):
        p21.model_from_checkpoint(tmp_path,entry,"full_history_gru",p21.MODEL_SEEDS[0])


def test_trained_state_only_not_optimizer_or_source_model_promotion():
    assert p21.KINDS==("full_history_gru","current_only_gru")
    assert p21.P20_PARENT=="ad69bcd06e973f0c05546c3e5782b770a0ce0e6e"
    for seed in p21.MODEL_SEEDS:
        assert p21.checkpoint_name("full_history_gru",seed).endswith(".pt")
        with pytest.raises(ValueError,match="P21_UNKNOWN_MODEL_IDENTITY"):
            p21.checkpoint_name("unexpected",seed)


def test_new_cold_inference_receipt_deterministic_with_all_models(tmp_path):
    checkpoint=tmp_path/"trained"
    checkpoint.mkdir()
    entries=[seeded_entry(checkpoint,seed,kind) for seed in p21.MODEL_SEEDS for kind in p21.KINDS]
    train_receipt=tmp_path/"cohort.json"
    train_receipt.write_text(json.dumps({
        "schema":p21.SCHEMA,
        "mode":"TRAIN_NEW_P20_EQUIVALENT_COHORT_ON_PUBLIC_DEV",
        "parent_source_sha":p21.P20_PARENT,
        "checkpoint_count":len(entries),
        "checkpoints":entries,
        "science_first_use":"NOT_RUN"
    }))
    a=p21.evaluate(FakeGym(),checkpoint,train_receipt)
    b=p21.evaluate(FakeGym(),checkpoint,train_receipt)
    assert a==b
    assert a["cold_loaded_models"]==6
    assert a["corrupt_archive_wrong_source_wrong_seed_denied"] is True
    assert a["all_training_absent_in_cold_process"] is True
    assert a["scientific_first_use"]=="NOT_RUN"
    assert all(e["full_minus_current_only"]==0.0 or isinstance(e["full_minus_current_only"],float)
               for e in a["per_seed"])
    assert all(len(e["all_new_world_factual_actions"])==8 for e in a["per_seed"])


def test_immutable_output_refuses_replay_write(tmp_path):
    out=tmp_path/"p21.json"
    p21.receipt_write(out,{"x":1})
    with pytest.raises(FileExistsError):
        p21.receipt_write(out,{"x":2})
    assert json.loads(out.read_text())=={"x":1}


def test_cli_help_no_runtime_training_and_no_model_install():
    src=ROOT/"scripts/native_p21_process_cold_retention_dev.py"
    result=subprocess.run([sys.executable,"-I","-B",str(src),"--help"],
                          cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert "--checkpoint-dir" in result.stdout
    assert "--mode" in result.stdout
    assert "pip install" not in src.read_text()
    assert "no_optimizer_rng_state_cold_training_continuation" in src.read_text()
