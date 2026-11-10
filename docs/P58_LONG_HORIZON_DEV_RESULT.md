# P58 — Long-horizon actual motor-action value, published physics DEV

**Source:** P57 PR #154 `37c4d6753f9a5e8c50d233375d271eaff2383ad6`; frozen before source at `05b16d3320d88e6c56e4cfeb56718ab1f285ddbc`.

Ninety-six shared reset states per controller, 256-step maximum: momentum bang-bang **96/96 successful**, mean reward **+92.0708**, mean real steps **79.29**; neutral, random 3-action and constant-right each **0/96**, mean rewards **0**, **-17.0010**, **-25.6000**. All preregistered action-value discrimination conditions passed. Mean absolute immediate action effect per motor unit was **0.0015**. This explains why one-step prediction is a weak action-selection task and identifies a more meaningful, delayed-goal-control target.

**No learning was tested:** The successful controller is handcoded; it must remain a strong conventional baseline in future learned-control comparisons. This uses P54's independently transcribed published dynamics, *not* official installed Gymnasium, external physical experiment, independently qualified science, or any promotion. No full Windows/Linux CI acceptance.

Focused CPU tests: **44 passed, 1 official-installed-runtime test skipped**. Commands from extracted folder:

```bash
PYTHONPATH=. python -m experiments.p58_long_horizon_action_value_dev --output experiments/p58_replay.json
PYTHONPATH=. python -m pytest -q tests
```

Check per-policy source trace SHA-256 and deterministic outcomes for replay, not wall-clock time. The first frozen result before adding preregistered reward-audit reporting is preserved as a separate file.

**Next serious science gate:** qualify official installed Gymnasium runtime on existing authorized host; then separately freeze a token-free Hybrid learning/decision-value experiment comparing fully converged GRU/MLP and strong momentum controllers, matched actual actions and information, cold retention, intervention value and whole-process cost. No runner, scheduler, controller, FORGE integration, merge or live action occurred.
