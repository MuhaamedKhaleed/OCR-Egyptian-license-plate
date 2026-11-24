# Changelog

## Unreleased

### Added
- Local single-plate inference adapter, pure decoder and Typer JSON CLI.
- Offline regression tests, dotenv configuration helper, pinned dependencies and local quality hooks.
- Setup, evaluation limits and verification documentation.

### Fixed
- Final notebook prediction uses the current image rather than a previous prediction.

### Security
- Notebook credential moved to the environment and saved outputs cleared. This does not revoke the old credential or remove its history.
