# C10 — Action-law transfer falsification, DEV only

C9 PR #80 established that trained P7 and conventional MLP tied across 18 observational DEV stress traces. C10 probes a distinct question: do either trained context selectors and the actual archived native memory snapshots function when the action–consequence law changes, including action magnitudes absent from all original P3 and P7 training?

## Preregistration and experiment

The fixed source-locked protocol is in experiments/native_c10_action_law_transfer_frozen.json, committed before C10 execution. It declares all trained source seeds, separate development stream seeds, constitutive laws, controls, loss and competence thresholds.

Original P7 factual-gradient training is unchanged: three independent seeds 303/307/311, 6,000 familiar factual examples, 1,920 separate-RNG validation examples, 600 AdamW updates times batch 256. LearnedRouter has 828 parameters; independently trained conventional MLP has 849. Exact source C8 fitting uses identical minibatch indices. Full convergence and FLOPs matching have not been established. Original native P3 seeds 46111/46117/46129 train 40 old then 40 recent actual updates, save root→40→80, and restore distinct source-validated C2 model/native-state checkpoints.

For each of 18 synthetic *development* traces, apply the same stochastic 96-step interventions to all six arms. The actions take values -1, -0.65, -0.25, 0.25, 0.65, or 1 (uniform); latent context changes with a 6% event hazard; action-result Gaussian noise has sigma 0.025. Use independently fixed development evaluation seeds 123041 and 123059, and three *new* action–consequence families:
- Linear magnitudes: bias 0.04 plus or minus 0.45 times action.
- Saturation: bias 0.04 plus or minus 0.45 times tanh(2×action)/tanh(2).
- Dead-zone friction: bias 0.04 plus or minus 0.45 times sign(action)×max(abs(action)−0.30,0)/0.70.

The six arms are genuinely trained P7, genuinely trained conventional MLP, a conventional past-only online ridge linear-slope estimator, always-recent memory, evaluator-only oracle true context, and P7 zero-history. Learned models receive the same last eight *completed factual action/consequence pairs*, and use the same two hypothetical-action prediction calls. Online ridge uses only those same eight most recent observations, with a fixed 0.05 slope ridge penalty and 0.04 minimum effect. Ridge can choose after two facts, whereas frozen P7/MLP default to old until eight; this difference is recorded and not a compute-matched scientific baseline.

Every nonoracle route is selected before the current outcome is calculated and revealed; no hidden context label, current outcome or future observation is supplied. All variants share the same frozen step80 model, old and recent saved native memories, actions and actual outcomes. There are zero fast-memory writes and zero AdamW updates during evaluation.

## Falsification and limits

The frozen diagnostics are P7 mature (at least eight elapsed events after hidden switch) route accuracy at least 75%; P7 mature MSE better than trained MLP by at least 0.01; better than online ridge by at least 0.01; better than always-recent by at least 0.05; better than zero-history by at least 0.05; and oracle mature MSE at most 0.20. This oracle competence requirement is essential: if even a correctly selected native memory produces poor predictions under shifted laws, routing accuracy is not a meaningful learning success. No thresholds are changed after seeing results, and nulls/negatives are retained.

This is a newly authored, known, transparent synthetic development law, NOT independently born or sealed scientific worlds, independent original first-use, novel real-world transfer, scientific qualification, permanent retention or recursive self-improvement. Source history and digests establish local engineering custody, not separate owner authentication. P7 and trained MLP share information, training steps and near-matched parameters, but their total compute/FLOPs and fully converged conventional competency remain unverified.

C10 runs only through the existing isolated Hybrid project runner, with one-use clean pinned exact-SHA source, mandatory final complete Windows/WSL regression, and original CUDA negative controls. Existing active P9/P10/P11/P12/P13 science and Wingless/Yggdrasil remain untouched. No real scientific first-use, new runner/controller/queue, paid API, accepted/latest pointer, FORGE integration, merge or deployment.
