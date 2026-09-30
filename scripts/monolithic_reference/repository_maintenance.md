# Monolithic maintenance receipt — 2026-09-30

This receipt covers structural cleanup after P4. It does not grant new numerical
qualification to the current source.

## Code and release notes

- `f370d64b`: extract BSR/scaling, diagnostic PCG, actor preconditioning, shared
  types and masked device kernels. Keep workspace methods, arithmetic, state
  identities and the public solver API. `linear.py`: 2023 → 539 lines;
  `pcg_production.py`: 1257 → 664 lines.
- Rename current articulation/contact/iteration modules and P3/P4/P5 reference
  tools by function. `e7b32e1c` also renames the remaining three P2 analysis
  modules. P1Q3 remains a quadrature name; serialized format/schema identifiers
  and the fixed-baseline adapter's old-checkout imports remain unchanged.
- `474914ab`: consolidate 47 Monolithic changelog fragments into three entries
  covering the solver/precision API, scenes/SDF/assets, and maintenance/testing.

The current scene commands are `scripts.monolithic_reference.run_regression`,
`run_precision` and `run_performance`. Only an explicitly requested overnight run
executes the ten full FP64 reference comparisons.

## Workspace documents and retained inputs

The sibling workspace directories are outside the Newton Git repository:

- `dev/monolithic/`: only `PRD.md`, `architecture.md`, `open_items.md` at the top;
  P0–P4 contain stage archives. `P4/HANDOFF.md` is a 35-line current handoff.
- The old handoff banners are now in [p4_stage_results.md](p4_stage_results.md).
- `agents/README.md` describes the P4 closeout, current development tree and
  remaining qualification limits. Six retired roles and integration reports
  moved under `agents/integration/artifacts/history/roles/`; old cleanup records
  moved under `history/storage-cleanup/`. Research references and retained asset
  symlinks were preserved.
- Move two distinct early control inputs into
  [asset/monolithic/trajectories](../../asset/monolithic/trajectories/README.md).
  The 10 mm root input is byte-identical to the existing retained Sharpa control:
  remove only its duplicate. All three old paths resolve through the asset map.
  No reference trajectory or frozen manifest was modified.

The workspace's `agents/integration/artifacts/history/repository-consolidation-v1/`
contains original document/changelog archives, code/module maps,
`relocations.csv.gz`, control hashes and current-document hashes. The 312 moved
files were checked against their before/after hashes; 41 frozen document files
kept their exact bytes. Ordinary historical Markdown links were rebased; the
original versions remain in the archive. Archived role reports retain their
original relative paths and can be located through the relocation table.
These local documents and history files are not included merely by committing or
pushing this worktree.

## Validation boundary

Pass static AST equivalence for 92 moved definitions/methods, parse 899 active
Python files, resolve project imports, import the public solver and extracted
modules, and expand the nightly case list to ten without executing a scene.
Current document links and relocated-file hashes pass. Full pre-commit passes.
Runtime tests, discovery, CPU/CUDA numerical comparisons and trajectories are
**NOT_RUN** under the user's no-rerun instruction. Graph preloading now targets
the moved kernel module; graph execution still requires subsequent runtime
validation. Historic P4/T5 results do not replace that validation.
