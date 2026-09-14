Prevent premature zero-direction PCG acceptance in the experimental monolithic solver by capping each linear residual target against the current outer Newton goal, while preserving CUDA graph reuse.
