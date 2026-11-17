"""R115 genuine chosen interventions and exact query-targeted causal information value.

Private four-factor three-stage SCM truth tables are only accessible inside
the generator. Public policy receives intended query treatment, two offered
actions, actual previous chosen-action signed outcomes and phase. No policy
gets hidden operator identity, query sign or unchosen consequences.
"""
from __future__ import annotations

import numpy as np

from experiments import r115_source_custody as custody

SEEDS=custody.SEEDS
OPS=("AND","OR","XOR","NAND","NOR","XNOR")
UNSEEN=(("AND","OR","XOR"),("OR","NAND","NOR"),
        ("XOR","XNOR","AND"),("NAND","XOR","OR"),
        ("NOR","AND","NAND"),("XNOR","NOR","XNOR"),
        ("AND","XNOR","NOR"),("XOR","NAND","XNOR"))
PANEL={"UNIT_FIT":256,"UNIT_VALID":96,"UNIT_UNSEEN":128}
SALTS={"UNIT_FIT":27501101,"UNIT_VALID":27501107,
       "UNIT_UNSEEN":27501113,"UNIT_BATCH":27501119,
       "UNIT_CHOICES":27501123,"UNIT_OUTCOMES":27501129,
       "UNIT_QUERIES":27501131,"UNIT_COLD":27501137}
STEPS=5
NOISE=.10
_SCORER_ONLY=object()


def operator(op,a,b):
    if op=="AND":return a&b
    if op=="OR":return a|b
    if op=="XOR":return a^b
    if op=="NAND":return 1-(a&b)
    if op=="NOR":return 1-(a|b)
    if op=="XNOR":return 1-(a^b)
    raise ValueError("R115_UNKNOWN_CAUSAL_OPERATOR")


def bit(triple,pattern):
    if type(pattern) is not int or not 0<=pattern<16:
        raise ValueError("R115_UNREGISTERED_INTERVENTION_BITS")
    a=(pattern>>3)&1;b=(pattern>>2)&1
    c=(pattern>>1)&1;d=pattern&1
    return operator(triple[2],operator(triple[0],a,b),
                    operator(triple[1],c,d))


def mask(triple):
    return sum(bit(triple,p)<<p for p in range(16))


ALL=tuple((a,b,c) for a in OPS for b in OPS for c in OPS)
RESERVED=frozenset(mask(t) for t in UNSEEN)
FIT=tuple(t for t in ALL if mask(t) not in RESERVED)
FIT_INDICES=np.asarray([i for i,t in enumerate(ALL) if t in FIT],np.int64)
FULL_INDICES=np.arange(len(ALL),dtype=np.int64)
TRUTH=np.asarray([[bit(t,p) for p in range(16)] for t in ALL],np.int8)


def audit_novelty():
    f=custody.audit_freeze()
    src=f["source"]
    if (src["salts"]!=SALTS or src["worlds_per_seed"]!=PANEL
        or tuple(src["operators"])!=OPS
        or tuple(tuple(x) for x in src["unseen_triples"])!=UNSEEN
        or [f"{mask(t):04x}" for t in UNSEEN]!=src["unseen_truth_masks_hex"]
        or len(ALL)!=216 or len(FIT)!=194
        or len({mask(t) for t in FIT})!=82 or len(RESERVED)!=8
        or any(mask(t) in RESERVED for t in FIT)
        or any(not any(t[k]==o for t in FIT)
               for k in range(3) for o in OPS)
        or STEPS!=src["chosen_probes_per_phase"] or NOISE!=src["outcome_noise_probability"]):
        raise PermissionError("R115_STRUCTURAL_CAUSAL_COMPOSITION_NOVELTY_DRIFT")
    return f


def assigned(panel,index):
    if panel not in PANEL or type(index) is not int or not 0<=index<PANEL[panel]:
        raise PermissionError("R115_UNAUTHORIZED_SOURCE_PANEL_OR_INDEX")
    catalog=UNSEEN if panel=="UNIT_UNSEEN" else FIT
    old=catalog[index%len(catalog)]
    changed=bool(index%2)
    new=(catalog[(index+7)%len(catalog)] if changed else old)
    return old,new,changed


class PhysicalCausalWorld:
    """Per-action counterfactual protected: only chosen interventions return."""
    def __init__(self,panel,seed,index,phase,*,_token=None):
        if (panel not in PANEL or seed not in SEEDS or
            type(index) is not int or not 0<=index<PANEL[panel] or
            phase not in (0,1)):
            raise PermissionError("R115_SOURCE_PROTECTED_OR_INVALID")
        if panel=="UNIT_UNSEEN":
            if _token is not _SCORER_ONLY:
                raise PermissionError("R115_FRESH_UNSEEN_BEFORE_SCRATCH_TRAINING")
        else:
            custody.guard_public_panel(panel)
        pair=assigned(panel,index)
        self._truth=pair[phase]
        self._panel=panel;self._seed=seed;self._index=index
        self._phase=phase;self._step=0
        self.history=[]
        rng=np.random.default_rng(np.random.SeedSequence([
            SALTS[panel],SALTS["UNIT_QUERIES"],seed,index,phase]))
        # Query treatment is public BEFORE choosing interventions. Query sign
        # is private until five physically observed chosen probe outcomes.
        self.public_query=int(rng.integers(0,16))

    def offers(self):
        if self._step>=STEPS:
            raise PermissionError("R115_ACTION_BUDGET_EXHAUSTED")
        rng=np.random.default_rng(np.random.SeedSequence([
            SALTS[self._panel],SALTS["UNIT_CHOICES"],
            self._seed,self._index,self._phase,self._step]))
        a=rng.choice(16,2,replace=False)
        return tuple(sorted(int(v) for v in a))

    def intervene(self,choice):
        offered=self.offers()
        if type(choice) is not int or choice not in offered:
            raise PermissionError("R115_UNCHOSEN_OR_UNOFFERED_EFFECT_REQUEST")
        sign=1 if bit(self._truth,choice) else -1
        rng=np.random.default_rng(np.random.SeedSequence([
            SALTS[self._panel],SALTS["UNIT_OUTCOMES"],self._seed,
            self._index,self._phase,self._step,choice]))
        if rng.random()<NOISE:
            sign=-sign
        self.history.append((choice,int(sign)))
        self._step+=1
        return int(sign)

    def sealed_query(self):
        if self._step!=STEPS:
            raise PermissionError("R115_TARGET_SIGN_BUDGET_FENCE")
        return self.public_query,(1 if bit(self._truth,self.public_query) else -1)


def _posterior(history,indices):
    if len(history)>STEPS:
        raise ValueError("R115_TOO_MANY_OBSERVATIONS")
    selected=np.asarray(indices,np.int64)
    if (selected.ndim!=1 or not len(selected)
        or np.any(selected<0) or np.any(selected>=len(ALL))):
        raise ValueError("R115_UNREGISTERED_PRIOR_TEMPLATE")
    w=np.ones(len(selected),np.int64)
    for p,s in history:
        if type(p) is not int or not 0<=p<16 or s not in (-1,1):
            raise ValueError("R115_NONPUBLIC_FACTUAL_FEEDBACK")
        w*=np.where(TRUTH[selected,p]==int(s==1),9,1)
    return selected,w


def posterior_prediction(query,history,*,full=True):
    if type(query) is not int or not 0<=query<16:
        raise ValueError("R115_NONPUBLIC_TARGET_QUERY")
    selected,w=_posterior(history,FULL_INDICES if full else FIT_INDICES)
    positive=int(w[TRUTH[selected,query]>0].sum())
    return (1 if 2*positive>=int(w.sum()) else -1),int(2*positive==int(w.sum()))


def target_query_information_value(candidate,query,history,*,full=True):
    """Exact integer expected decrease in query Bayes 0-1 error.

    New observation y is either +1 or -1. Each structural hypothesis weighs
    9 for observed sign agreement or 1 for disagreement, total denominator 10
    shared by hypotheses. The improvement numerator is pure integer and
    never accesses private ground truth, outcomes or future test query sign.
    """
    if (type(candidate) is not int or not 0<=candidate<16
        or type(query) is not int or not 0<=query<16):
        raise ValueError("R115_UNREGISTERED_PUBLIC_INTERVENTION_OR_QUERY")
    ids,w=_posterior(history,FULL_INDICES if full else FIT_INDICES)
    qpositive=(TRUTH[ids,query]>0)
    xpositive=(TRUTH[ids,candidate]>0)
    total=int(w.sum());positive=int(w[qpositive].sum())
    loss=min(positive,total-positive)
    plus=np.where(xpositive,9,1);minus=np.where(xpositive,1,9)
    wp=w*plus;wm=w*minus
    tp=int(wp.sum());tm=int(wm.sum())
    qp=int(wp[qpositive].sum());qm=int(wm[qpositive].sum())
    after=(min(qp,tp-qp)+min(qm,tm-qm))
    gain=10*loss-after
    if not 0<=gain<=10*total:
        raise RuntimeError("R115_NONPHYSICAL_EXPECTED_ERROR_GAIN")
    return int(gain),int(10*total)


def public_query_value_for_fit(candidate,query,history):
    gain,denom=target_query_information_value(candidate,query,history,full=False)
    return 20.*gain/denom
