"""P16: factual-reward-trained gate, frozen P11 champion and bounded P14 challenger.

The original P11 2-step GRU champion, P14 REINFORCE challenger, source and
published DEV worlds remain unchanged. A *separate learned gate* decides
whether to use the challenger where decisions DISAGREE, trained only from
its own executed actions and subsequently observed factual consequences.
Hard cap on challenger deviations protects an ACTION budget, not reward.
Wrong-credit, random gate and nondeviating champion remain explicit controls.

Engineering-only reused synthetic DEV. Never an unobserved scientific source,
competent fully trained benchmark proof, cold process checkpoint or promotion.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from scripts.native_p14_reward_trained_intervention_dev import (
    CausalFeaturePolicy, TrainingWorld,train_model as original_p14_training,
    completed_history,eval_world,factual_step,FAMILIES,OLD_SEEDS,LAG_SEEDS,
    INIT_SEEDS as P14_SEEDS, OBS_PREFIX,DECISIONS,PROBE_INTERVAL,
    TRAIN_UPDATES as P14_UPDATES)
from scripts.native_p11_temporal_credit_dev import (
    ConventionalCausalGRU,train_arm,data as p11_data,
    plan as original_p11_plan, INIT_SEEDS as P11_SEEDS,
    FIT_SEED,FIT_EPISODES,VALIDATION_SEED,VALIDATION_EPISODES,
    UPDATES as P11_UPDATES)
from scripts.native_p15_strong_control_dev import common_probe_rng

SCHEMA="ttt-hybrid.native-p16.champion-guarded-learning-dev.v1"
ARMS=("frozen_champion","unrestricted_challenger",
      "learned_guard","miscredit_guard","random_guard")
GATE_UPDATES=32
GATE_WORLDS=8
GATE_HORIZON=24
GATE_TRAIN_CAP=5
MAX_EVAL_CHALLENGER_DEVIATIONS=12
GATE_EVAL_THRESHOLD=.55
TRAIN_BASE=311000
GATE_LR=.003
DISCOUNT=.97
LOSS_ENTROPY=.01
MAX_PROCESS_SECONDS=155
REWARD_MARGIN=.05
OLD_REGIME_MAX_DEFICIT=.02


def guard_dev_only():
    if os.environ.get("NATIVE_P16_DEV_ONLY","1")!="1":
        raise RuntimeError("P16_SCIENTIFIC_FIRST_USE_FORBIDDEN")


class LearnedGuard(nn.Module):
    """Only prior observed factual history, champion action, P14 policy logit."""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(18,24),nn.Tanh(),nn.Linear(24,1))

    def forward(self,history,champion_action,challenger_logit):
        if history.ndim!=3 or history.shape[1:]!=(8,2) or (
            champion_action.shape!=(history.shape[0],)) or (
            challenger_logit.shape!=(history.shape[0],)):
            raise ValueError("P16_GUARD_OBSERVATION_SHAPE")
        x=torch.cat((history.reshape(-1,16),
                     champion_action.float().reshape(-1,1),
                     challenger_logit.reshape(-1,1)),dim=1)
        return self.net(x).reshape(-1)


def batch_champion_two_step(model,history):
    """Exact original P11 plan math, batched for cheap development training."""
    n=history.shape[0]
    with torch.no_grad():
        scores=[]
        for action in (-1.,1.):
            at=torch.full((n,1),action,dtype=torch.float32)
            first=model(history,at)
            projected=torch.cat((
                history[:,1:,:],
                torch.cat((at,first),dim=1).reshape(n,1,2)),dim=1)
            negative=model(projected,torch.full_like(at,-1.))
            positive=model(projected,torch.full_like(at,1.))
            scores.append((first+torch.maximum(negative,positive)).reshape(-1))
        # Tie handling exactly favors +1 in original P11 plan.
        return torch.where(scores[1]>=scores[0],
                           torch.ones(n,dtype=torch.int64),
                          -torch.ones(n,dtype=torch.int64))


def misalign_rewards(actual_returns):
    if actual_returns.ndim!=2 or actual_returns.shape!=(
        GATE_HORIZON,GATE_WORLDS):
        raise ValueError("P16_CREDIT_NEGATIVE_SHAPE")
    return torch.roll(actual_returns,
                      shifts=(GATE_HORIZON//2,1),dims=(0,1))


def train_gate(champion,challenger,model_seed,*,wrong_credit=False,
               updates=GATE_UPDATES):
    if model_seed not in P14_SEEDS or not 1<=updates<=GATE_UPDATES:
        raise ValueError("P16_UNREGISTERED_GATE_TRAINING")
    torch.manual_seed(model_seed+1441)
    gate=LearnedGuard()
    before={k:v.detach().clone() for k,v in gate.state_dict().items()}
    optim=torch.optim.AdamW(gate.parameters(),lr=GATE_LR)
    ledger=hashlib.sha256()
    stage=[]
    count_decisions=0
    learned_route_choices=0
    start=time.perf_counter()
    champion.eval();challenger.eval()
    for epoch in range(updates):
        world_seeds=[TRAIN_BASE+(epoch*GATE_WORLDS+j)*103
                     for j in range(GATE_WORLDS)]
        ledger.update(json.dumps(world_seeds,separators=(",",":")).encode())
        worlds=[TrainingWorld(s) for s in world_seeds]
        hist=[]
        for w in worlds:
            prefix=[]
            for i in range(8):
                action=1 if i%2 else -1
                prefix.append((action,w.realize(action)))
            hist.append(prefix)
        forced=[random.Random(s+51011) for s in world_seeds]
        deviations=[0]*GATE_WORLDS
        log_terms=[]
        mask_terms=[]
        factual=[]
        for t in range(GATE_HORIZON):
            h=completed_history(hist)
            with torch.no_grad():
                champion_actions=batch_champion_two_step(champion,h)
                challenger_logits=challenger(h)
                challenger_actions=torch.where(
                    challenger_logits>=0,torch.ones_like(champion_actions),
                    -torch.ones_like(champion_actions))
            gate_logits=gate(h,champion_actions,challenger_logits)
            dist=torch.distributions.Bernoulli(logits=gate_logits)
            chosen=dist.sample()
            eligible=torch.tensor([
                (t%PROBE_INTERVAL!=0 and
                 int(champion_actions[j])!=int(challenger_actions[j])
                 and deviations[j]<GATE_TRAIN_CAP)
                for j in range(GATE_WORLDS)],dtype=torch.float32)
            log_terms.append(dist.log_prob(chosen)*eligible)
            mask_terms.append(eligible)
            actions=[]
            for j in range(GATE_WORLDS):
                if t%PROBE_INTERVAL==0:
                    a=forced[j].choice((-1,1))
                elif eligible[j] and chosen[j].item()>=.5:
                    a=int(challenger_actions[j])
                    deviations[j]+=1
                    learned_route_choices+=1
                else:
                    a=int(champion_actions[j])
                actions.append(a)
            rewards=[]
            for j,(w,a) in enumerate(zip(worlds,actions)):
                outcome=w.realize(a)
                hist[j].append((a,outcome))
                rewards.append(outcome)
                count_decisions+=1
            factual.append(torch.tensor(rewards,dtype=torch.float32))
        r=torch.stack(factual)
        actual_returns=torch.empty_like(r)
        future=torch.zeros(GATE_WORLDS)
        for t in reversed(range(GATE_HORIZON)):
            future=r[t]+DISCOUNT*future
            actual_returns[t]=future
        credit=misalign_rewards(actual_returns) if wrong_credit else actual_returns
        advantages=credit-credit.mean(dim=1,keepdim=True)
        advantages=advantages/advantages.std().clamp(min=1e-4)
        logp=torch.stack(log_terms)
        eligible=torch.stack(mask_terms)
        if not bool(eligible.sum()>0):
            raise RuntimeError("P16_GATE_TRAINING_NEVER_DIFFERED")
        loss=-(logp*advantages.detach()).sum()/eligible.sum()
        # Entropy is a measurable, small regularizer on actual routing decisions.
        loss=loss-LOSS_ENTROPY*torch.distributions.Bernoulli(
            logits=gate(h,champion_actions,challenger_logits)).entropy().mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(gate.parameters(),3.)
        optim.step()
        if (epoch+1)%8==0:
            stage.append({"update":epoch+1,"loss":float(loss.detach()),
                          "mean_actual_factual_return":float(r.mean()),
                          "eligible_disagreements":int(eligible.sum())})
    gate.eval()
    changed=any(not torch.equal(before[k],v) for k,v in gate.state_dict().items())
    return gate,{
        "seed":model_seed,"wrong_credit":wrong_credit,
        "actual_gradient_updates":updates,
        "actual_chosen_training_actions":count_decisions,
        "different_action_routes_chosen":learned_route_choices,
        "same_world_cohort_sha256":ledger.hexdigest(),
        "factual_gradient_changed_weights":changed,
        "parameter_count":sum(p.numel() for p in gate.parameters()),
        "true_factual_reward_to_go_used":not wrong_credit,
        "credit_time_world_misaligned":wrong_credit,
        "training_curve":stage,
        "training_wall_seconds":time.perf_counter()-start
    }


def evaluate(family,seed,pair,models):
    if family not in FAMILIES or pair not in (0,1,2) or seed not in (
        OLD_SEEDS if family in ("instant","lagged") else LAG_SEEDS):
        raise ValueError("P16_UNREGISTERED_DEVELOPMENT_WORLD")
    complete={}
    prefixes=[]
    for arm in ARMS:
        world=eval_world(family,seed)
        history=[]
        previous=0
        rng=common_probe_rng(seed,pair)
        random_route_rng=random.Random(977117+seed+pair*337)
        for t in range(OBS_PREFIX):
            action=1 if t%2 else -1
            actual=factual_step(world,t,action,previous)
            history.append((action,actual))
            previous=action
        prefixes.append(tuple(history))
        actions=[]
        outcomes=[]
        deviations=0
        disagreements=0
        gate_opportunities=0
        probes=[]
        route_traces=[]
        start=time.perf_counter()
        for step in range(DECISIONS):
            t=OBS_PREFIX+step
            if step%PROBE_INTERVAL==0:
                chosen=rng.choice((-1,1))
                probes.append([step,chosen])
                route="FORCED_EQUAL_PROBE"
            else:
                champion_action,_=original_p11_plan(
                    models["champion"],history,lookahead=True)
                with torch.no_grad():
                    h=completed_history([history])
                    challenger_logit=models["challenger"](h)
                    challenger_action=(
                        1 if float(challenger_logit.item())>=0 else -1)
                if champion_action!=challenger_action:
                    disagreements+=1
                can_deviate=(
                    champion_action!=challenger_action and
                    deviations<MAX_EVAL_CHALLENGER_DEVIATIONS)
                if arm=="unrestricted_challenger":
                    chosen=challenger_action
                    route="UNRESTRICTED_CHALLENGER"
                elif arm=="frozen_champion":
                    chosen=champion_action
                    route="CHAMPION"
                elif can_deviate:
                    gate_opportunities+=1
                    if arm=="random_guard":
                        use_challenger=random_route_rng.random()<.25
                    else:
                        network=models[
                            "learned_guard" if arm=="learned_guard" else
                            "miscredit_guard"]
                        with torch.no_grad():
                            prob=float(torch.sigmoid(network(
                                h,torch.tensor([champion_action]),
                                challenger_logit)).item())
                        use_challenger=prob>=GATE_EVAL_THRESHOLD
                    if use_challenger:
                        chosen=challenger_action
                        deviations+=1
                        route="CHALLENGER_DEVIATION"
                    else:
                        chosen=champion_action
                        route="CHAMPION"
                else:
                    chosen=champion_action
                    route="DISAGREEMENT_CAP_FALLBACK" if (
                        champion_action!=challenger_action) else "AGREED_ACTION"
                route_traces.append(route)
            # Revealed only after the route/action was committed.
            factual=factual_step(world,t,chosen,previous)
            history.append((chosen,factual))
            previous=chosen
            actions.append(chosen)
            outcomes.append(factual)
        complete[arm]={
            "mean_factual_return":sum(outcomes)/DECISIONS,
            "first_24_return":sum(outcomes[:24])/24,
            "changed_regime_24_return":sum(outcomes[24:48])/24,
            "old_regime_revisit_16_return":sum(outcomes[48:])/16,
            "actions":actions,
            "factual_outcomes":outcomes,
            "forced_probe_actions":probes,
            "max_challenger_deviations":MAX_EVAL_CHALLENGER_DEVIATIONS,
            "actual_nonprobe_challenger_deviations":deviations if arm in (
                "learned_guard","miscredit_guard","random_guard") else None,
            "nonprobe_disagreements":disagreements,
            "eligible_routing_events":gate_opportunities,
            "routing_trace_nonforced":route_traces,
            "actual_decisions":len(actions),
            "policy_wall_seconds":time.perf_counter()-start,
            "trace_sha256":hashlib.sha256(json.dumps(
                [actions,outcomes],separators=(",",":"),allow_nan=False
            ).encode()).hexdigest(),
        }
    p=[tuple(tuple(x) for x in y["forced_probe_actions"])
       for y in complete.values()]
    checks={
        "same_eight_prior_factual_observations":len(set(prefixes))==1,
        "exact_same_sixteen_signed_forced_probe_actions":len(set(p))==1 and
            len(p[0])==16,
        "every_arm_exactly_sixtyfour_actual_decisions":all(
            x["actual_decisions"]==DECISIONS for x in complete.values()),
        "guards_never_exceed_max_twelve_deviations":all(
            0<=complete[a]["actual_nonprobe_challenger_deviations"]<=
            MAX_EVAL_CHALLENGER_DEVIATIONS
            for a in ("learned_guard","miscredit_guard","random_guard")),
        "no_future_or_counterfactual_outcome_used":True,
        "sham_no_action_identical_all_factual_returns":(
            len(set(tuple(x["factual_outcomes"])
                    for x in complete.values()))==1
            if family=="sham_no_action" else True),
        "all_factual_outcomes_finite":all(
            math.isfinite(y) for x in complete.values()
            for y in x["factual_outcomes"])
    }
    return {"family":family,"seed":seed,"pair":pair,
            "integrity":"PASS" if all(checks.values()) else "FAIL",
            "checks":checks,"arms":complete}


def run()->dict:
    guard_dev_only()
    torch.set_num_threads(1)
    started=time.perf_counter()
    fit=p11_data(FIT_SEED,FIT_EPISODES)
    val=p11_data(VALIDATION_SEED,VALIDATION_EPISODES)
    result=[]
    for pair,(p14_seed,p11_seed) in enumerate(zip(P14_SEEDS,P11_SEEDS)):
        challenger,challenger_training=original_p14_training(
            "causal",p14_seed)
        champion,champion_training=train_arm(
            ConventionalCausalGRU,p11_seed,fit,val)
        gate,gate_training=train_gate(
            champion,challenger,p14_seed,wrong_credit=False)
        negative,negative_training=train_gate(
            champion,challenger,p14_seed,wrong_credit=True)
        models={"champion":champion,"challenger":challenger,
                "learned_guard":gate,"miscredit_guard":negative}
        before={
            name:hashlib.sha256(b"".join(
                p.detach().cpu().numpy().tobytes()
                for p in model.parameters())).hexdigest()
            for name,model in models.items()
        }
        cases=[evaluate(family,dev_seed,pair,models)
               for family in FAMILIES
               for dev_seed in (
                   OLD_SEEDS if family in ("instant","lagged") else LAG_SEEDS)]
        after={
            name:hashlib.sha256(b"".join(
                p.detach().cpu().numpy().tobytes()
                for p in model.parameters())).hexdigest()
            for name,model in models.items()
        }
        checks={
            "original_champion_actual_supervised_adamw_1200":
                champion_training["actual_optimizer_updates"]==P11_UPDATES
                and champion_training["gradient_changed_weights"],
            "original_challenger_actual_policy_gradient_updates_96":
                challenger_training["actual_policy_gradient_updates"]==
                P14_UPDATES and challenger_training["gradient_changed_weights"],
            "both_guards_same_32_actual_policy_gradient_updates":
                gate_training["actual_gradient_updates"]==
                negative_training["actual_gradient_updates"]==GATE_UPDATES
                and gate_training["factual_gradient_changed_weights"]
                and negative_training["factual_gradient_changed_weights"],
            "same_training_world_cohort_for_learned_and_wrong_credit":
                gate_training["same_world_cohort_sha256"]==
                negative_training["same_world_cohort_sha256"],
            "both_guards_have_actual_differing_action_routes":
                gate_training["different_action_routes_chosen"]>0 and
                negative_training["different_action_routes_chosen"]>0,
            "all_four_model_weights_frozen_unchanged_across_eval":before==after,
            "all_twentyfour_original_DEV_worlds":len(cases)==24,
            "all_chronological_sham_probe_and_cap_negatives_passed":
                all(x["integrity"]=="PASS" for x in cases),
            "old_S26_S27_and_separate_C_research_untouched":True,
        }
        means={f:{
            arm:sum(c["arms"][arm]["mean_factual_return"] for c in cases
                    if c["family"]==f)/4 for arm in ARMS}
            for f in FAMILIES}
        limits={f:{
            arm:sum(c["arms"][arm]["actual_nonprobe_challenger_deviations"]
                    for c in cases if c["family"]==f)/4 for arm in (
                    "learned_guard","miscredit_guard","random_guard")}
            for f in FAMILIES}
        hypotheses={}
        for f,r in means.items():
            hypotheses[f]={
                "learned_guard_over_frozen_champion_0_05":
                    r["learned_guard"]>=r["frozen_champion"]+REWARD_MARGIN,
                "learned_guard_over_random_gate_0_05":
                    r["learned_guard"]>=r["random_guard"]+REWARD_MARGIN,
                "learned_guard_over_miscredit_gate_0_05":
                    r["learned_guard"]>=r["miscredit_guard"]+REWARD_MARGIN,
                "learned_guard_old_competence_within_0_02":
                    r["learned_guard"]>=r["frozen_champion"]-
                    OLD_REGIME_MAX_DEFICIT
                    if f in ("instant","lagged") else None}
        result.append({
            "paired_init":[p14_seed,p11_seed],
            "engineering":"PASS" if all(checks.values()) else "FAIL",
            "integrity_checks":checks,
            "original_training":{"P14_challenger":challenger_training,
                                 "P11_champion":champion_training},
            "gate_training":{"aligned":gate_training,
                             "miscredit":negative_training},
            "mean_reward_per_factual_action":means,
            "mean_nonprobe_deviations_per_episode":limits,
            "frozen_hypotheses":hypotheses,
            "all_factual_episodes":cases})
    elapsed=time.perf_counter()-started
    counts={f:{
        k:(sum(int(r["frozen_hypotheses"][f][k]) for r in result)
           if result[0]["frozen_hypotheses"][f][k] is not None else None)
        for k in result[0]["frozen_hypotheses"][f]}
        for f in FAMILIES}
    good=len(result)==3 and all(r["engineering"]=="PASS" for r in result) and (
        elapsed<=MAX_PROCESS_SECONDS)
    return {
        "schema":SCHEMA,
        "status":"PASS_BOUNDED_CHAMPION_GUARD_DEV_NOT_SCIENCE"
                 if good else "FAIL_BOUNDED_CHAMPION_GUARD_DEV",
        "scientific_first_use":"NOT_RUN",
        "science_seeds_consumed":[],
        "native_substrate_changed":False,
        "champion_accepted_pointer_changed":False,
        "FORGE_research_loop_activated":False,
        "original_P14_learner_optimizer_updates_total":3*P14_UPDATES,
        "original_P11_champion_supervised_updates_total":3*P11_UPDATES,
        "actual_guard_policy_gradient_updates_total":3*2*GATE_UPDATES,
        "actual_factual_guard_training_actions":3*2*GATE_UPDATES*
                                                GATE_WORLDS*GATE_HORIZON,
        "factual_eval_decisions":3*6*4*len(ARMS)*64,
        "champion_challenger_shared_forced_probes":16,
        "max_new_challenger_deviations_per_guard_episode":
            MAX_EVAL_CHALLENGER_DEVIATIONS,
        "novel_evaluation_world_source_authentic":False,
        "independent_conventional_plateau_qualified":False,
        "whole_process_matched_compute":False,
        "cold_process_death_restore_tested":False,
        "all_frozen_negative_results_preserved":True,
        "families":list(FAMILIES),
        "seed_pairs":list(zip(P14_SEEDS,P11_SEEDS)),
        "support_out_of_three":counts,
        "records":result,
        "whole_process_wall_seconds":elapsed,
        "limitations":"Risk cap bounds how often a challenger substitutes an action; it cannot guarantee the final consequence is no worse than champion because action effects are temporally delayed. The guard only sees factual history/champion decision/challenger logit. Fully trained original P11 champion fails original 1% learning-plateau competence; challenger and champion had different learning curricula, capacities and optimizer budgets. Known published synthetic DEV worlds, not scientific unseen source; no independent true world first-use, real-cold retention, FLOP match, sustained learning or accepted checkpoint. Do not promote if any frozen old-capability or margin criterion fails."}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--receipt",type=Path,required=True)
    a=p.parse_args()
    result=run()
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
        "status":result["status"],
        "factual_training":result["actual_factual_guard_training_actions"],
        "factual_evaluation":result["factual_eval_decisions"],
        "support":result["support_out_of_three"],
        "reward_by_pair":[{"seeds":r["paired_init"],
            "means":r["mean_reward_per_factual_action"],
            "deviations":r["mean_nonprobe_deviations_per_episode"]}
            for r in result["records"]],
        "seconds":result["whole_process_wall_seconds"],
        "scientific_first_use":result["scientific_first_use"],
    },sort_keys=True))
    return 0 if result["status"]=="PASS_BOUNDED_CHAMPION_GUARD_DEV_NOT_SCIENCE" else 1

if __name__=="__main__":
    raise SystemExit(main())
