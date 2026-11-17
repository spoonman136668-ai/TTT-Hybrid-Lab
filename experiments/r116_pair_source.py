"""R116 new paired do-interventions: identical counts, opposite genuine do(7) signs.

The first three chosen probes are physically performed, NEVER imputed.
Only after them can a public policy choose do(12) or do(14). Query do(15)
is public, while its true consequence is private until four real probes.
"""
from __future__ import annotations

from experiments import r115_structured_query_source as structural
from experiments import r116_source_custody as custody

SEEDS=custody.SEEDS
UNSEEN=(("XOR","NOR","AND"),("OR","OR","XOR"),
        ("AND","OR","XNOR"),("AND","NOR","OR"),
        ("AND","XOR","NOR"),("XOR","XOR","XOR"),
        ("OR","XNOR","AND"),("AND","XNOR","OR"))
RESERVED_R116=frozenset(structural.mask(t) for t in UNSEEN)
PREVIOUS_R115=frozenset((0x1eee,0x0008,0x0990,0x6fff,
                         0xfff7,0x1ee1,0x0666,0x8778))
FIT=tuple(t for t in structural.ALL
          if structural.mask(t) not in RESERVED_R116|PREVIOUS_R115
          and structural.bit(t,8)==1 and structural.bit(t,9)==0)
ACTIONS=(8,9,7)
FINAL=(12,14)
QUERY=15
COUNT={"UNIT_FIT":256,"UNIT_VALID":96,"UNIT_UNSEEN":128}
_SCORER_ONLY=object()


def admit():
    f=custody.audit_freeze()
    src=f["source"]
    if (len(structural.ALL)!=216 or len(UNSEEN)!=8 or len(FIT)!=24 or
        len({structural.mask(t) for t in FIT})!=9 or
        len(RESERVED_R116)!=8 or
        not RESERVED_R116.isdisjoint(PREVIOUS_R115) or
        any(structural.mask(t) in RESERVED_R116|PREVIOUS_R115 for t in FIT) or
        tuple(tuple(x) for x in src["unseen_composition_representatives"])!=UNSEEN or
        [f"{structural.mask(t):04x}" for t in UNSEEN]!=
            src["unseen_masks_branch_negative_hex"]+
            src["unseen_masks_branch_positive_hex"] or
        sum(structural.bit(t,7)==0 for t in FIT)!=12 or
        sum(structural.bit(t,7)==1 for t in FIT)!=12 or
        any([structural.bit(t,p) for p in ACTIONS[:2]]!=[1,0]
            for t in FIT+UNSEEN) or
        [structural.bit(t,7) for t in UNSEEN]!=[0]*4+[1]*4 or
        [structural.bit(t,QUERY) for t in UNSEEN[:4]]!=[0,0,1,1] or
        [structural.bit(t,QUERY) for t in UNSEEN[4:]]!=[0,0,1,1]):
        raise PermissionError("R116_CAUSAL_PAIR_FIT_HELDOUT_NOVELTY_DRIFT")
    return f


def composition(panel,seed,index):
    if panel not in COUNT or seed not in SEEDS or type(index) is not int or not 0<=index<COUNT.get(panel,0):
        raise PermissionError("R116_SOURCE_PAIR_NOT_REGISTERED")
    if panel=="UNIT_UNSEEN":
        return UNSEEN[index%8]
    return FIT[(index+seed%len(FIT))%len(FIT)]


class PhysicalPairWorld:
    """Strict chosen-action causal simulator; hidden SCM never passed to actor."""
    def __init__(self,panel,seed,index,*,_token=None):
        if panel=="UNIT_UNSEEN":
            if _token is not _SCORER_ONLY:
                raise PermissionError("R116_ORIGINAL_SEALED_UNSEEN_BEFORE_ALL_FIT_UPDATES")
        else:
            custody.permit_public_panel(panel)
        self._private_triple=composition(panel,seed,index)
        self._step=0
        self.history=[]
        self.public_query=QUERY

    def offers(self):
        if self._step>=4:
            raise PermissionError("R116_FOUR_REAL_PROBE_BUDGET_EXHAUSTED")
        return (ACTIONS[self._step],) if self._step<3 else FINAL

    def intervene(self,choice):
        if type(choice) is not int or choice not in self.offers():
            raise PermissionError("R116_UNCHOSEN_COUNTERFACTUAL_OR_UNOFFERED_PROBE_DENIED")
        real_sign=1 if structural.bit(self._private_triple,choice) else -1
        self.history.append((choice,int(real_sign)))
        self._step+=1
        return int(real_sign)

    def sealed_query(self):
        if self._step!=4:
            raise PermissionError("R116_QUERY_TRUTH_BEFORE_FOUR_REAL_PROBES")
        return QUERY,(1 if structural.bit(self._private_triple,QUERY) else -1)


def public_exact_prediction(history):
    return structural.posterior_prediction(QUERY,history,full=True)


def public_exact_voi_for_next(history):
    if (len(history)!=3 or [p for p,_ in history]!=list(ACTIONS)
        or [s for _,s in history[:2]]!=[1,-1]):
        raise PermissionError("R116_PUBLIC_CALIBRATION_COUNTS_NOT_MATCHED")
    return [structural.target_query_information_value(p,QUERY,history,full=True)
            for p in FINAL]


def public_observation_target(history):
    scores=public_exact_voi_for_next(history)
    if scores[0][1]!=scores[1][1]:
        raise RuntimeError("R116_VOI_SHARED_PUBLIC_PRIOR_DENOMINATOR_DRIFT")
    # Exact non-privileged information-gain target, same for every scratch arm.
    return [20*s[0]/s[1] for s in scores]
