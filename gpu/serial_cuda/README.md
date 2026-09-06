# NodalS serial CUDA solver

Experimental single-GPU, device-resident CUDA implementation of the
P1+BF3/P0 NodalS SIMPLE solver.

This subtree is intentionally separate from the CPU implementation.

## Current checkpoint

Validated development checkpoint: H8.

The current implementation cumulatively includes the H6, H7 and H8
performance optimizations.

## Precision

Current optimized path:

- FP32 device state
- FP32 operators
- FP32 AMG data
- FP64 reductions and convergence-control quantities

A validated full-FP64 reference implementation is retained separately under
`fp64_reference/`.

## H6: coarse AMG CSR SpMV

Explicit non-terminal AMG levels benchmark scalar thread-per-row and
warp-per-row CSR SpMV at setup and independently select the faster backend.

On the 768k Re=20 benchmark:

- AMG L1 warp SpMV speedup: approximately 5.88x
- AMG L2 warp SpMV speedup: approximately 2.28x
- SIMPLE throughput increased from 8.406 to 9.581 MIUPS

The AMG hierarchy and numerical operator are unchanged.

## H7: momentum convection assembly

The changing momentum convection contribution uses a cooperative
warp-per-cell CUDA kernel.

On the 768k benchmark:

- convection assembly: approximately 14.46 -> 5.64 ms
- total momentum assembly: approximately 15.94 -> 6.24 ms
- SIMPLE throughput: 10.869 MIUPS

The scalar row-L1 relaxation finalizer remained faster and is retained.

## H8: persistent B geometry

The static P1+BF3 divergence/gradient coefficients are precomputed once:

- 8 local velocity basis functions
- 3 components
- 24 FP32 values per tetrahedron
- 96 bytes per cell

For 768,530 cells the persistent table consumes approximately 70.36 MiB.

Measured improvements:

- exact fine pressure CSR refresh: 11.03 -> 4.76 ms
- B^T p kernel: 4.31 -> 1.09 ms
- continuity B u: 2.53 -> 0.80 ms

Integrated 768k fixed-10 result:

- SIMPLE: 59.471 ms
- throughput: 12.923 MIUPS
- pressure: 33.584 ms
- momentum: 18.732 ms
- assembly: 6.112 ms
- average pressure PCG iterations: 2.9

The pressure iteration count and fixed-10 continuity history remain unchanged.

## Pressure solver

Current pressure path:

- exact fine FP32 CSR
- fine CSR numeric refresh every SIMPLE iteration
- persistent precomputed B geometry
- PCG
- smoothed-aggregation AMG
- power-iteration spectral estimates
- Chebyshev/Jacobi smoothing
- hybrid scalar/warp explicit coarse CSR SpMV
- terminal dense inverse

## Momentum

- one shared scalar momentum operator for Ux, Uy and Uz
- persistent CSR topology
- persistent laminar diffusion numeric CSR
- convection contribution refreshed each SIMPLE iteration
- cooperative warp-per-cell convection assembly
- FGS1 momentum solve in the current performance path
- no fixed-work momentum residual audits

## Dependencies

The serial CUDA path has no PETSc, HYPRE or MPI runtime dependency.

## Preserved FP64 reference

`fp64_reference/` contains the validated G5D full-FP64 CUDA implementation:

- StateReal = FP64
- OperatorReal = FP64
- AMGReal = FP64
- reductions = FP64

The FP64 implementation is retained as the numerical reference while the
optimized FP32 CUDA path continues to evolve.
