# Hybrid P6: synthetic remote checkpoint artifact recovery (inactive)

**Purpose:** verify actual external GitHub Actions artifact transport of a P2-trained, synthetic model checkpoint, then recover it in a separate workflow job using P4's independent source-chain validation. This does not modify a training owner, model accepted pointer, checkpoint store, research controller, GPU or external/local drives.

## What this proves

The **producer job** starts from the pinned exact candidate commit on the project's existing isolated runner. A synthetic one-step AdamW training fixture saves both an immutable zero-step root and a trained child: model weights, optimizer state, native recurrent/fast memory/history, RNG and data-loader cursor. A short metadata file identifies the exact source SHA, root and tip; only five explicit nonsecret files are uploaded to GitHub Actions artifact storage, under a run-ID-qualified name with seven-day retention.

The **consumer job** runs only after upload completion, checks out the same exact commit into a fresh destination, and downloads that specific artifact from the GitHub Actions artifact service. It refuses missing, additional, oversized, symlinked or tampered files; validates all manifest and payload hashes, complete parent chain and source/protocol fingerprints; uses P4 to copy into new staging; cold-loads the state; and compares the recovered model, optimizer and native memory against independently recomputed deterministic fixture values. It also checks exact sampler progress and refuses any accepted/latest pointer.

Each phase is separately run-scoped to the preexisting `TTT-HYBRID-GPU-WIN` self-hosted project runner. The qualification includes **full Windows and Ubuntu WSL CPU regressions** at exact candidate SHA and the existing original-source negative controls. No original science-world data, real GPU training, model promotion, or other research-loop modification occurs.

## Deliberate security and custody limits

- This is a **small public/synthetic demonstration** of remote artifact upload/download and local cold recovery, not an approved off-host production backup system.
- GitHub Actions artifacts are retention-limited and can expire, be deleted, be inaccessible during service interruption, or be subject to storage/billing limits. A seven-day artifact is not a durable accepted-model repository.
- The job2 download demonstrates transit from a separately operated artifact service, but the two jobs may execute on the same physical Windows runner. It does not prove survival of full workstation loss, long-term retention, or a fresh independent GPU host restore.
- Content digests establish byte integrity; the test fixture/source lock provide scoped provenance; no separate cryptographic owner-issuer signing or independent secure-storage custodian is introduced.
- Neither actual power-loss atomicity nor directory-fsync durability is claimed. There is no live checkpoint pointer, garbage collection, automated backup, scheduler, backup encryption, or permission expansion.
- Real checkpoint data might contain private model state or proprietary training information; **do not upload real models via this workflow**. A separate project-owner storage policy/authorization is required before any authentic checkpoint remote transport.
- The existing P2/P3/P4/P5 stacks and separate science development remain unmodified, draft and gated by their own requirements. This candidate cannot be silently merged or promoted based on green engineering results.

## Future live activation requirements

Choose an owner-approved persistent encrypted destination with independently confirmed identity, failure domain, credentials/ACL and retention; freeze full state and source compatibility; test offline fresh-host cold restore and real loss of primary; test power-loss semantics; reconcile exact model and sampler context with P3 competence/retention before any model promotion. FORGE can receive independent read-only receipts only through its existing authorized research interface.
