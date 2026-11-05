# G5 — Native official TTT-Linear evaluation on CPU

## Why this gate exists

G4's full CPU suite qualified only our wrapper/test doubles. G5 executes
the **real pinned** `TTTLinear` from the official
`test-time-training/ttt-lm-pytorch` source repository and checks whether
its test-time update path is actually exercised. It does not attempt Mamba2
CUDA or a scientific comparison yet.

## Exact identities

- Original project baseline: `3c3d58f0f2e670aa7e640a656224e2e59caf5ca4`.
- G4 parent candidate: `7548074711a3eee5d7b92f54d45392dd1c620564`.
- Official upstream TTT-LM source:
  `cd831db10c8c9a0f6340f02da5613316a8a92b67`.
- Source file blob: `da267f4a88196fd4709fa1e5e62235a3fa746ec6`.
- Python 3.11, pinned CPU torch, Transformers, NumPy.
- GitHub Actions and checkout are exact-SHA pinned.
- No new scheduler, controller, model service, queue, or local Windows runner.

## Engineering acceptance criteria

Run one native CPU probe per OS at the exact PR-head SHA. Fail unless
all of the following are observed:

1. Exact clean upstream checkout and source blob matching the frozen pin.
2. Python `ttt` module, `TTTLinear` and `TTTConfig` resolve to that checkout.
3. Real native forward produces finite [2,16,64] outputs.
4. Repeated episode forward with `cache_params=None` is deterministic.
5. Suffix perturbation does not affect outputs for the earlier prefix.
6. Running the prefix alone reproduces its outputs from the longer episode.
7. Replacing `ttt_base_lr=1` with `ttt_base_lr=0` *on the same weights*
   produces a measurable difference. Restoring the rate restores output.
8. Original TTT layer backward computes finite, nonzero parameter gradients.
9. Every receipt records the candidate SHA and pins, platform, versions,
   test outcome, elapsed time, control deltas and failure provenance.
10. G4 Linux+Windows full regression must still pass at the exact G5 SHA.

Native source integration and these controls do **not** demonstrate useful
adaptation, training convergence, superior memory or retention, transfer,
speedup, MoD routing quality, or broad intelligence. `SCIENTIFIC=UNVERIFIED`.

## Execution

GitHub Actions starts automatically on PR. The workflow downloads only the
pinned public upstream source into its temporary GitHub-hosted job; it does
not touch `C:\Research\TTT-Hybrid` or your other projects.

G5 is stacked against the G4 development branch because G4 is still open,
so neither G4 main-branch acceptance nor any competing work is presumed.

## Follow-on

Once both systems pass authentic TTT CPU and mandatory G4 regression,
the next independently authorized experiment is source- and ABI-qualified
Mamba2/TTT CUDA integration on isolated GPU infrastructure. No automatic
installation on the owner's workstation is implied by this PR.
