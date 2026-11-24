# Changelog

## Unreleased

### Added
- Local single-plate inference adapter, pure decoder and Typer JSON CLI.
- Opt-in per-plate auto-crop mode, with empty-list behavior when no plate is found.
- Offline regression tests, dotenv configuration helper, pinned dependencies and local quality hooks.
- Setup, annotation audit, bounded evaluation and verification documentation.

### Fixed
- Final notebook prediction uses the current image rather than a previous prediction.
- Suppress near-identical character boxes by confidence, with synthetic regression tests.
- Correct reviewed Arabic aliases from original training annotations, including hamza and missing dad/zay mappings.

### Verified
- Original-checkpoint CPU inference without retraining.
- Frozen next-slice evaluation records 7/10 exact primary-plate reads, with failures and limitations, not general accuracy.

### Security
- Notebook credential moved to the environment and saved outputs cleared. This does not revoke the old credential or remove its history.
- No sample images, annotation cache or model weights published.


### Engineering follow-up
- Use the committed uv lock in CI and key dependency-installer caching by that lock.
- Complete copy-only profile README drafts and clarify historical versus current verification.

### Synthetic demonstration follow-up

Added standalone invented-input backend demo, checked JSON output and documented limits. Opt-in ML container added where a model backend exists; default lightweight image unchanged. Hosted container verification is required separately.

## Portable local notebook follow-up

- Replace platform-specific notebook execution with local opt-in CPU training/inference and a safe default decoder run.
- Add local training guards and notebook execution regressions; document actual one-epoch random-weight/rectangle smoke with zero quality claim.
- Correct stale credential/license/cache documentation, preserve inherited-rights caveats, and pin Node 24 checkout/cache releases.
