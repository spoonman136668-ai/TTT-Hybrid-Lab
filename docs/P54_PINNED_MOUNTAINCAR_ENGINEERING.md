# P54 — externally specified MountainCarContinuous-v0 factual engineering seam

**Scope: Engineering development only.** This code copies the published physical transition equations and fixed parameters from pinned upstream Gymnasium source `Farama-Foundation/Gymnasium@e6be96abfb0a9df36da5e7884522c4f1f2432e53` (`gymnasium/envs/classic_control/continuous_mountain_car.py`, Git blob `96dbb0d31a17d144468562f4bc68fce902e2a69f`). It **does not** copy or run installed official Gymnasium source in this environment, because `gymnasium` is not installed; direct original-runtime parity is explicitly **SKIPPED, UNVERIFIED**. This is a published **simulator**, not a real-world laboratory experiment or externally qualified scientific source.

The unmodified P50 native factual history (`model.py`, `native_factual_hybrid_stream.py`) accepts real completed `position, velocity, executed action, realized next sensor` physical transitions from a separate bounded source. An exploratory candidate preview never executes the physical step, writes a future observed outcome, or advances factual memory. P50 requires nonzero actions, so this engineering bridge only admits executed bounded actions in `[-1,1]` except `0` and is **not a complete implementation of the full environment action space**. Factual rank 3 is necessary but insufficient for genuine causal identification.

Local commands (Python 3.13, torch 2.10 CPU, NumPy 2.3, pytest 9):

```bash
PYTHONPATH=. python -m pytest -q tests
PYTHONPATH=. python run_p54_engineering.py
```

**Local result:** 23 tests passed, one official-runtime parity test skipped; 64 episodes × 8 actual published-equation reference transitions = **512 completed actions**; all 64 bounded streams passed a linear-algebraic rank certificate, no hypothetical state commits; source-free reference untrained P50 two-action next-state MSE **0.072759**. A known exact simulator dynamics baseline trivially has **0** MSE against the same deterministic reference, and the P50 model here is **untrained**. These numbers **do not** establish model learning, transfer, physics fidelity to a running upstream package, or superiority.

**Next requirements before science:** source-qualified native upstream Gymnasium runtime with exact Git blob, independent original physical task/hardware or independently adjudicated simulator dynamics, frozen model training and full qualified GRU/MLP/analytical dynamics/last-effect controls, balanced real action interventions, unfamiliar dynamics, retention and full-process cost; Windows/Linux mandatory final regression; separate scientific first-use and acceptance authority. No FORGE, runner, workflow, controller, scheduler, paid API or merge was modified.
