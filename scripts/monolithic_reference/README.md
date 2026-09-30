# Monolithic scripts

Executable tools and fixtures remain in this directory. Current usage documentation
is maintained in the shared workspace:

- [Script guide](../../../../dev/scripts/monolithic_reference.md)
- [PCG benchmark](../../../../dev/scripts/pcg_benchmark.md)
- [Stage history](../../../../dev/monolithic/P4/script_records/README.md)
- [Documentation relocation index](documentation-index.json)

The relative workspace links require the local `worktrees/<stage>` layout; the
external documents are not included in a standalone Git checkout. The index records
original paths and hashes so historical references remain traceable. Tests stay in
`newton/tests/`; this change moves documentation only.
