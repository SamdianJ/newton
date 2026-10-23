Fix run-to-run differences of experimental `SolverMonolithic` on CUDA: tet energies and assembled tet residuals are now reduced in a fixed order instead of by floating-point atomic accumulation.
