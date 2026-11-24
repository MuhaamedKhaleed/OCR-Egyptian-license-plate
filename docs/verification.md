# Verification record

Verified locally on Python 3.11.18. No training, real-weight inference, dataset download, paid API or GPU was used. The follow-up CI uses a free standard public-repository runner.

## Commands and observed results

- `uv sync --locked --extra dev`: passed.
- `python -m egypt_plate_ocr --help`: passed.
- `egypt-plate-ocr --help`: passed with required image and weights options.
- `python -m pytest --cov=egypt_plate_ocr --cov-report=term-missing`: 17 passed, 86% package statement coverage. This is software coverage, not OCR accuracy. Tests use temporary local files and fake models, not a network service.
- `python -m ruff check src tests`: passed.
- `python -m ruff format --check src tests`: passed.
- `python -m mypy src`: passed with typed definitions required and untyped bodies checked.
- `python -m pre_commit run --all-files`: all local hooks passed in an initialized git checkout.
- `command -v docker`: no executable found. Docker build/run was not possible.

Typer 0.15.2 initially failed against the automatically selected Click 8.5.1. Pinning Click 8.1.8 fixed both required-option validation and help rendering, and the full suite passed again. No failed check is presented as a pass.

## Requested phase status

1. Package and editable install: verified. Names are `cli.py`, `inference.py`, `core.py`, `config.py`, and `__main__.py`; no empty `utils.py`.
2. Cleanup: reusable decoder and local adapter extracted. Original experimental notebook still has hardcoded Kaggle paths and training experiments; not fully refactored.
3. Errors/logging: CLI configures logging once, JSON stays on stdout, input/model errors exit 2. Notebook's exploratory prints remain. No real model error behavior verified.
4. Configuration: dotenv helper and key failure tests added, `.env.example` documents key. Notebook uses its environment directly, not the helper. Old history still contains a credential and must be handled separately.
5. Dependencies: runtime/inference/dev extras separated, uv lock present. Fresh editable install and help verified. Optional ML extra and real inference unverified.
6. Tests: 17 pass, 86% statement coverage. Tests exercise decoding/order, finite coordinates, missing files/config, CLI required options, clean external failure and per-image prediction calls. Missing coverage includes the real Ultralytics import path and some entry-point/error branches.
7. Quality: Ruff, configured mypy and local pre-commit hooks pass. Old notebook is not included in the package lint scope.
8. CI: added on push and pull request using standard ubuntu-24.04 with a public-repository-only job guard. No cache, uploaded artifact or image publication. Live results are recorded below.
9. Docker: local executable unavailable; CI builds the multi-stage slim image, runs CLI help, confirms non-root UID 10001 and checks invalid input exits 2. The image excludes the optional ML runtime and weights. This is a CLI smoke image, not real inference validation.
10. Interface: thin Typer CLI, help and offline interface tests pass. No API or hosted deployment was added.
11. README: setup, architecture, real decoder output and limits are documented. Real inference commands are explicitly intended usage, not tested success. No fake screenshot/demo.
12. Git: feature branch and incremental commits, no main merge. The first README commit was not conventional; later commits use conventional prefixes. No v1 tag or release, because this is not validated as release-ready. CHANGELOG tracks the pilot, not a fabricated release.
13. Presentation: description/topics/profile text is in `docs/presentation.md` for review, not applied to the GitHub profile.

## Assumptions and original problems

One plate crop, original mapping/order preserved, unknown labels ignored. Mapping aliases require dataset review. The final notebook cell reused the previous image's results, an exposed key was embedded, outputs made the notebook large, and reusable logic had no tests. The prediction correction is committed separately from package extraction.

## Remaining manual/blocked work

Revoke the previously exposed key; agree coordinated history cleanup. Verify code/data/model rights before licensing. Supply trusted compatible weights, measure plate-level exact match on a held-out set, test real inference and deployment. Public standard-runner free eligibility was checked in GitHub billing documentation before execution. Review costs again before moving to a private repository or adding caches, artifacts, larger runners or image publication. Review PR before merge; add a genuine demo and decide description/topics/pinning/release only when ready. Existing repository and push are handled by this pilot, not left as invented manual steps.

## Hosted verification follow-up

Run https://github.com/MuhaamedKhaleed/OCR-Egyptian-license-plate/actions/runs/45386868698 uses the public standard Ubuntu runner. Observed final result: success. All quality checks, Docker build and container smoke checks passed. Local Docker was not installed or claimed. This run does not test trained weights, GPU behavior or model accuracy.
