"""Reproducible, non-scientific CPU architectural smoke benchmark."""
import hashlib, json, platform, time
import torch
from model import Configuration, HybridPilot

CONFIGS={
 "selective_ssm":Configuration(attention=False,ttt=False,depth=False),
 "attention":Configuration(ssm=False,ttt=False,depth=False),
 "ssm_attention":Configuration(ttt=False,depth=False),
 "ssm_attention_ttt":Configuration(depth=False),
 "ssm_attention_ttt_routed":Configuration(),
}

def main():
    torch.set_num_threads(1)
    cases=[]
    for name,cfg in CONFIGS.items():
        torch.manual_seed(101)
        m=HybridPilot(12,7,cfg).eval()
        params=sum(p.numel() for p in m.parameters())
        for seq in (16,64):
            torch.manual_seed(202)
            x=torch.randn(2,seq,12)
            with torch.inference_mode():
                for _ in range(3): _=m(x)
                samples=[]
                for _ in range(7):
                    t=time.perf_counter();y=m(x);samples.append((time.perf_counter()-t)*1e3)
            assert torch.isfinite(y).all()
            cases.append({"arm":name,"sequence_length":seq,"batch":2,"params":params,
                          "median_ms":round(sorted(samples)[len(samples)//2],3),
                          "routing_active_fraction":None if m.depth is None else round(m.depth.last_utilization,3)})
    result={"label":"engineering_smoke_only_not_quality_or_speed_acceptance",
            "torch":torch.__version__,"python":platform.python_version(),"platform":platform.platform(),
            "threads":torch.get_num_threads(),"seed":101,"cases":cases}
    print(json.dumps(result,indent=2))
    with open("smoke_receipt.json","w",encoding="utf-8") as f:json.dump(result,f,indent=2)

if __name__=="__main__":main()
