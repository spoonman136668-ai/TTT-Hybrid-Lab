# Hybrid P7 — offline checkpoint custody admission (inactive DEV)

**Scope:** This development library adds keyed source/byte identity checks to the existing P2–P6 checkpoint chain. It does **not** configure external storage, create owner keys, sign any authentic model, claim independent physical backup, activate training, or approve promotion.

## What P7 actually implements

`native_backup_custody.py` accepts a caller-selected immutable P2 checkpoint tip, source/runtime identity, isolated object directory, a textual claimed storage-domain tag, expiry window and a **caller-held** HMAC-SHA256 key (minimum 32 bytes). The key is never written into a checkpoint or receipt.

`seal_backup_inventory` checks the full P2 parent chain, enforces exact expected filenames and SHA-256 object bytes, rejects symlinks/extra mutable pointers, bounds size and 90-day maximum attestation lifetime, and creates a canonical, domain-separated HMAC envelope. This is only a *preparation* API; it neither writes nor uploads files.

`verify_backup_inventory` verifies the MAC using a separately supplied key, scope, checkpoint identity, explicit tip, domain claim, signed time window, every source object and full parent chain. It returns `HASH_AND_TEST_KEY_VERIFIED_ONLY` with explicit fields stating owner-key custody, physical backup separation, retention assurance, scientific acceptance, training and promotion are all **unverified/false**. It does not select a fallback checkpoint or mutate an accepted/latest pointer. Test keys in source are conspicuously synthetic and never suitable for actual deployment.

Ten focused negative/positive tests exercise wrong/missing key, tampered signed metadata, mismatched domain, expiry, missing parent, corrupt saved weights, extra mutable files, altered source identity, size ceilings and no-promotion, using tiny one-step synthetic CPU checkpoints.

## Why not activate a real off-host backup now?

A real encrypted backup requires an owner-selected destination and failure domain, key custody, credentials/ACL, authenticated issuer, retention and disaster-recovery policy. *No such concrete authorized destination, independently witnessed failure domain, or owner-managed key is established in the P6 record.* A filesystem path, opaque domain label, content checksum, or locally held HMAC key by itself cannot establish independent physical storage. The pending P7 code cannot substitute for those facts.

The existing seven-day GitHub Actions synthetic P6 artifact is for an engineering demonstration, **not durable accepted-model storage**. Do not upload actual learned/private model states or reusable signing secrets to GitHub Actions artifacts based on this test.

## Qualification and future integration

P7 builds on exact P6 R2 candidate `f1c2a0b069972c5b1caa3b0b84d02674e8978dab`. All prior research/first-use/runner/controller branches and frozen scientific protocols remain unchanged. Its project-owned one-use CPU workflow must pass a full exact-source Windows/WSL regression including original-source negative controls, plus the ten new synthetic tests. HMAC success still does **not** prove authentic owner issuer signing until separately provisioned and qualified.

Before real storage admission: create a project-owner-approved encrypted durable repository with independent outage characteristics, determine safe key management/rotation, authenticate issuer and verify actual artifacts on a **different physical recovery host** after primary loss, test power-loss fsync and recovery, protect checkpoint data/sampler progress, and separately qualify retention, retention thresholds, science and accepted-pointer authority. FORGE may eventually consume read-only authenticated receipts through its existing interface; this module grants no new scheduler/controller authority.

P7 is **development-only and draft**. It is intentionally not connected to P8 scientific training or other projects.
