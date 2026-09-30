# Newton Development Guidelines

Read and follow the canonical [source code and public API guidelines](CODING_GUIDELINES.rst) before changing or reviewing Newton code. This file contains agent-specific workflow instructions; the linked guide is authoritative for coding and API design.
For reviews, also read and apply the [review guidelines](REVIEW_GUIDELINES.rst) and follow `.claude/skills/code-review-newton/SKILL.md`.

- Create a feature branch on your fork before committing—never commit directly to `main`. Give the pull request a concise, descriptive title.
- Use imperative mood in commit messages ("Fix X", not "Fixed X"), with a roughly 50-character subject and a body wrapped at 72 characters that explains what and why.
- Verify regression tests fail without the fix before committing.

Run `uvx pre-commit run -a` to lint and format before committing. Use `uv` for all commands; fall back to `venv` or `conda` only if `uv` is unavailable.

```bash
# Examples
uv sync --extra examples
uv run -m newton.examples basic_pendulum
```

## Tests

### Monolithic test policy

For Monolithic work, apply the following scope rules to the general testing instructions above. The detailed component map and commands are in [project.md](project.md).

- Organize tests around currently supported functionality: joint, tet, contact, collision, linear, SDF, solver lifecycle, assets, controls and tooling. Use `newton/tests/test_monolithic_<component>_*.py`; do not introduce development-phase names such as `p2`, `p3` or `p4` for new tests.
- During development, run the affected component tests and relevant small q+x integration checks. Ordinary CI discovers these tests through the existing Newton test runner. CPU and CUDA results, unavailable assets and skips must be reported accurately. Do not substitute a complete robot trajectory for a small regression reproducer.
- Retire tests whose only purpose is guarding historical reports, obsolete diagnostic accounting, frozen source snapshots or retired implementations. Check imports, test registration and callers first. Move any still-valid mathematical, input-validation or safety assertions to the appropriate functional test before removing the historical file. Keep independent oracles, failure rollback and explicitly supported state/checkpoint compatibility coverage.
- The nine Franka plus one Sharpa FP64 trajectories are a **manually triggered overnight regression**, entered with `scripts.monolithic_reference.run_p4 --profile nightly`. They are not a default local, commit or ordinary CI gate. The `dev` profile still runs two complete trajectories and is also an explicit long run; it is not the small integration tier. Do not automatically launch either profile after a refactor, documentation change or component test failure. Do not add a schedule or provision runners as part of routine development.
- Run the overnight matrix serially on one GPU and inspect `summary.json`, `RESULTS.md` and case receipts after it finishes. A failed case must not prevent collection of the other cases. Preserve source/input/reference identities, physical gates and existing comparison policies; never weaken them to obtain PASS. Filtered runs remain PARTIAL; mixed source/input identities must not produce an aggregate PASS.
- Keep the ten reference trajectories immutable. By default, successful new recordings are deleted after their summaries and comparison results are saved; failed recordings remain for diagnosis. Use `--retain-recordings all` only when full new recordings are needed, and clean diagnostic output after review. Do not accumulate duplicate successful trajectories or commit large assets/logs.
- Report component checks and overnight results separately. Mark unexecuted checks NOT_RUN and bind results to the actual tested version; do not reuse another commit's PASS. Stopped is not convergence. FP32 numerical/performance experiments are separate from default FP64 regression, while current precision-safety component checks remain in the component suite.
- Respect explicit user instructions not to run tests. Static checks are not runtime acceptance. Commit each completed reviewable batch; push only when requested.

The commands below are general Newton examples; they do not authorize an automatic Monolithic long run.

```bash
uv run --extra dev -m newton.tests
uv run --extra dev -m newton.tests -k test_viewer_log_shapes           # specific test
uv run --extra dev -m newton.tests -k test_basic.example_basic_shapes  # example test
uv run --extra dev --extra torch-cu12 -m newton.tests                  # with PyTorch
```

```bash
# Benchmarks
uvx --with virtualenv asv run --launch-method spawn main^!
```

## PR Instructions

- If opening a pull request on GitHub, use the template in `.github/PULL_REQUEST_TEMPLATE.md`.
- Follow `changelog/README.md`: add a Towncrier fragment for user-facing changes instead of editing `CHANGELOG.md` directly. A `.skip` reason is optional for changes without user-facing impact.
- Preview fragments with `uvx --from towncrier==25.8.0 towncrier build --draft --version X.Y.Z --date YYYY-MM-DD`.

## Examples

- Follow the `Example` class format.
  - Implement `test_final()` or `test_post_step()`; an example may implement both.
  - In test mode, `test_post_step()` runs after each simulation step and `test_final()` runs after the example completes.
- Register the example in `README.md` with its `python -m newton.examples <name>` command and a 320x320 JPEG screenshot.
