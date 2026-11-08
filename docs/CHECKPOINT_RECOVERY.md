# Native Hybrid P4 — explicit checkpoint replication and cold recovery (inactive)

**Purpose:** Preserve the exact immutable P2 training lineage in an independently selected copy, so future project-owned training can recover from a damaged/unavailable primary without silently restarting learned state. This is a Python library, **not** an active trainer, off-host backup service, FORGE integration, controller, accepted-pointer writer, or scientific evaluation.

## Bounded, fail-closed transport

`mirror_verified_chain(source, destination, tip_sha256=..., identity=...)` requires an existing destination and an **explicit** immutable P2 checkpoint SHA-256. It validates the source chain using P2 `validate_chain` (including model, optimizer, RNG, native memory, sampler), then copies only its P2 `.pt` and `.json` objects in root-to-tip order. Each file goes to a temporary file, is streamed and SHA-256-checked, flushed, immutable-hardlink-published, and reread. Collisions fail closed: no overwrite, no implicit reset, no latest selection, and no accepted-pointer changes. A total-byte limit applies. The destination chain is independently validated before `COPY_VERIFIED_UNAUTHENTICATED` is returned.

A transfer may be interrupted after publishing a subset of complete objects, but that does not make the tip a valid completed copy. Retry must explicitly select and validate the tip. Corrupt primary storage does not trigger implicit fallback: an authorized owner must select a verified replica and restore to fresh staging. Source, protocol, architecture, or data-split changes need separately qualified migration.

## What this does **not** prove

Two local directories **do not** prove independent durable backups, disaster-recovery reliability, or actual GPU-model fidelity. Directory fsync/power-loss behavior on Windows and network volumes remains unverified. Caller must separately authenticate owner, source commit, checkpoint issuer, custody and evidence bytes: SHA-256 confirms integrity, not provenance. No remote storage, secrets, ACL changes, deletion/GC, host resource arbitration, multi-host installation, accepted-pointer CAS, trainer integration or science first-use is provided.

## Before activation

1. Project owner provisions a truly independent backup location and proves independent failure domains and access controls.
2. Verify P2 ancestry and P3 capability-policy identity against authenticated source/artifacts.
3. Qualify real termination during a write, power/storage failures, corrupt manifest/payload, missing ancestors, destination collision and fresh-host recovery without losing the original accepted checkpoint.
4. Cold restore model, optimizer, RNG, native recurrent/fast memory and sampler cursor on supported Windows/WSL/GPU environments; test retained capabilities, no replay, and full exact-source regression.
5. Use separately authorized frozen first-use, matched trained controls, and project-owned scientific qualification for any accepted-model promotion. FORGE may later read authenticated receipts via its existing isolated contract; this module grants no authority.

P2/P3 remain draft dependencies; research loops remain unchanged. Synthetic engineering checks alone never certify owner-authenticated off-host durability, cumulative scientific progress, or autonomous improvement.
