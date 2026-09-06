# NodalS serial CUDA solver

Experimental single-GPU, device-resident CUDA implementation of the
P1+BF3/P0 NodalS SIMPLE solver.

This subtree is intentionally separate from the CPU implementation.

## Current checkpoint

Validated development checkpoint: H4.

## Precision

- FP32 device state and operators
- FP64 reductions and convergence-control quantities

## Momentum

- one shared scalar momentum operator for Ux, Uy and Uz
- persistent CSR topology
- persistent laminar diffusion numeric CSR
- convection numeric contribution refreshed every SIMPLE iteration
- static diffusion Dirichlet contribution retained
- FGS1: one forward multicolor GS pass per SIMPLE iteration
- ALTGS1: alternating forward and backward pass between SIMPLE iterations
- fixed-work timing path contains no momentum residual audits

For variable or turbulent viscosity, the diffusion numeric values can be
refreshed while preserving the CSR topology and precomputed geometry.

## Pressure

The exact pressure-correction operator is

    A_p = B diag(rAU) B^T

Current pressure path:

- exact fine FP32 CSR
- fine CSR numeric values refreshed every SIMPLE iteration
- warp-per-row fine CSR refresh
- current fine CSR used for both the physical PCG operator and AMG fine level
- FP32 PCG pressure solver
- smoothed-aggregation AMG preconditioner
- power-iteration spectral estimate
- Chebyshev/Jacobi AMG smoothing
- explicit Galerkin coarse matrices
- dense terminal coarse solve
- FP64 reductions

The coarse SA transfers, coarse Galerkin matrices and spectral estimates are
currently setup snapshots.

## Current RTX 3060 reference

Re=20 pipe, fixed 10 SIMPLE iterations, FGS1:

- 768,530 cells: approximately 91.43 ms/SIMPLE = 8.41 Mcell-SIMPLE/s
- 1,143,041 cells: approximately 135.21 ms/SIMPLE = 8.45 Mcell-SIMPLE/s

ALTGS1 gives substantially stronger early continuity reduction while retaining
one momentum GS pass per SIMPLE iteration.

## Scope

This is currently a serial single-GPU experimental solver.

The GPU execution path does not use PETSc, HYPRE or MPI.

The existing CPU NodalS implementation remains independent and unchanged.
