"""F5 source-free exact raw-event causal negative tests, no F3/F4 artifact in CI."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
import p85mr_f5_native_vs_controls as f5
from p80mr_causalgym_trace_adapter import _action_key, _observed_effect
from p85mr_t1_fitcal_engine import _padded


def toy():
    rows=[]
    for w in range(3):
        pos=[.01*w,.02,.001,.03]
        a=[];y=[]
        for t in range(3+w):
            action=(t+w)%2
            nxt=[pos[0]+(.03 if action else -.01),pos[1]+.02,
                 pos[2]+.004,pos[3]-.003]
            a.append(_action_key(pos,action))
            y.append(_observed_effect(pos,nxt,1.,False,False))
            pos=nxt
        rows.append({"action":torch.tensor(a,dtype=torch.float32),
                     "effect":torch.tensor(y,dtype=torch.float32)})
    return rows


def test_f5_frozen_exact_model_source_and_training_comparator():
    f=f5.locked()
    assert f["parent"]["source_sha"]=="00196c634509c7d04962e78988f4da97227f38f6"
    assert f["parent"]["F4_controls_result_sha256"]==f5.CONTROLS_SHA
    assert f["parent"]["F4_controls_artifact_id"]==13883054583
    assert f["training"]["max_updates"]==800
    assert f["training"]["sampler_seed"]==985299
    assert f["evaluation"]["quality_max_ratio_vs_strongest"]==.9
    assert f["execution"]["runner"]=="TTT-HYBRID-GPU-WIN"
    assert f["data"]["never_ood_or_dev"] is True
    source=(ROOT/"model.py").read_bytes()
    blob=hashlib.sha1(f"blob {len(source)}\0".encode()+source).hexdigest()
    assert blob=="5d9ea57b827aa1796548447868fa5ee14c251b49"


def test_f5_padded_factual_inputs_have_same_step_target_withheld():
    facts,mask=_padded(toy())
    first=f5.factual_inputs(facts)
    assert first.shape==(3,5,32)
    assert torch.equal(first[:,:, :16],facts["action"])
    assert not first[:,0,16:].any()
    assert torch.equal(first[:,1:,16:],facts["effect"][:,:-1])
    altered={k:v.clone() for k,v in facts.items()}
    altered["effect"][:,2,:7]+=3.
    new=f5.factual_inputs(altered)
    assert torch.equal(first[:,:3],new[:,:3])
    assert not torch.equal(first[:,3:],new[:,3:])


def test_f5_original_unmodified_four_blocks_finite_masked_train_loss():
    model=f5.original_native(985201)
    assert model.ssm is not None and model.attn is not None
    assert model.ttt is not None and model.depth is not None
    facts,mask=_padded(toy())
    y=f5.native_predict(model,facts)
    assert y.shape==(3,5,16)
    assert bool(torch.isfinite(y).all())
    loss=f5.training_loss(model,facts,mask)
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_f5_mutated_current_label_cannot_change_its_prediction():
    model=f5.original_native(985201).eval()
    facts,mask=_padded(toy())
    mutated={k:v.clone() for k,v in facts.items()}
    mutated["effect"][:,2,:7]+=10.
    with torch.no_grad():
        a=f5.native_predict(model,facts)
        b=f5.native_predict(model,mutated)
    assert torch.equal(a[:,:3],b[:,:3])


def test_f5_owner_and_unpinned_artifact_fail_before_training(monkeypatch,tmp_path):
    for key in ("GITHUB_REPOSITORY","GITHUB_ACTOR","GITHUB_REF",
                "GITHUB_RUN_NUMBER","RUNNER_NAME","P85MR_F5_NATIVE_FITCAL_ONLY"):
        monkeypatch.delenv(key,raising=False)
    with pytest.raises(PermissionError,match="F5_OWNER_ROUTE_DENIED"):
        f5.owner_admission()
    p=tmp_path/"fake.zip"
    with zipfile.ZipFile(p,"w") as z:z.writestr("p85mr_f4_controls_result.json","{}")
    with pytest.raises(ValueError,match="F5_CONTROL_ARCHIVE_ZIP_SHA_DRIFT"):
        f5.source_controls(p)


def test_f5_exact_isolated_interpreter_imports_on_both_os():
    cmd=[sys.executable,"-I","-B",
         str(ROOT/"experiments/p85mr_f5_native_vs_controls.py"),"--help"]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=35)
    assert r.returncode==0,r.stderr[-1800:]
    assert "--f3" in r.stdout and "--f4" in r.stdout and "--receipt" in r.stdout


def test_f5_science_promotion_and_source_effects_not_contained():
    txt=(ROOT/"experiments/p85mr_f5_native_vs_controls.py").read_text()
    for phrase in ("import causal_gym","env.reset(","env.do(","workflow_dispatch",
                   "torch.load(","torch.save(","schedule.create","pull_request_target"):
        assert phrase not in txt,phrase
    assert '"scientific_pass":False' in txt
    assert '"no_ood":True' in txt
    assert 'allow_pickle=False' in txt
