# P59 — exact installed Gymnasium runtime qualification harness

**Result:** local harness engineering PASS (50 PASS, 2 SKIPPED). **Actual official runtime UNAVAILABLE**, not PASS. The current isolated container lacks the installed `gymnasium` package, and the package registry DNS lookup failed. The runner-backed Windows/Linux full regression is also unverified.

P59 checks **the actual installed** `gymnasium.make("MountainCarContinuous-v0")`, original unwrapped class/module, **whole source-file Git blob** `96dbb0d31a17d144468562f4bc68fce902e2a69f` (pinned upstream `Farama-Foundation/Gymnasium@e6be96abfb0a9df36da5e7884522c4f1f2432e53`), registered 999-step TimeLimit, 16 previously consumed engineering seeds × two controls (neutral and momentum), actual source reset/step outcomes against the P54 transcription, rewards/terminal flags and a 999-step neutral wrapper truncation. It requires absolute state agreement <= 5e-7 and reward agreement <= 1e-9; historical engineering tolerance, not scientific significance threshold. All calls execute only on an actual installed source once its exact file bytes pass the pin. No drift fallback or substituted P57 extracted step implementation.

From repository root on a qualified host that already has the **exact pinned official** package installed (avoid installing an unverified version):

```bash
PYTHONPATH=. python -m scripts.p59_official_gymnasium_runtime_parity --receipt experiments/p59_official_host_receipt.json
PYTHONPATH=. python -m pytest -q tests
```

Exit 0 means the **official installed runtime engineering parity** finished successfully with a source-verified package; exit 2 means `UNAVAILABLE` and exit 3 means `FAIL`. The emitted JSON differentiates all statuses. A wrong source blob, class, registration, time-limit or physical observation is never promoted to PASS.

**This does not prove Hybrid learning.** The original P56 authored development test failed all three strong control groups and P58 tested only hand-coded controllers. Original Gymnasium wrapper and full source runtime are not qualified here. Before any model integration or scientific acceptance, finish exact-source Windows/Linux regression plus independently authorized evaluation against fully converged controls, matched actions, cold retention and whole-process costs.

No new CI workflow, scheduled task, runner, controller, model promotion, protected science source or FORGE change.
