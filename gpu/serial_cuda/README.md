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

## Full-FP64 cumulative H8 mode

The cumulative H8 CUDA source is precision-generic and can now be built in
either full FP32 or full FP64 without maintaining a second copy of the H8
implementation.

Build targets:

- `make fp32` / `nodals_gpu_h8_fp32`: FP32 state, operator and AMG with FP64 reductions.
- `make fp64` / `nodals_gpu_h8_fp64`: FP64 state, operator, AMG and reductions.

Both precision modes retain the same cumulative H8 execution structure:

- H6 per-level scalar/warp coarse AMG SpMV selection
- H7 cooperative warp-per-cell momentum convection assembly
- H8 persistent precomputed B geometry
- exact fine pressure CSR refreshed every SIMPLE iteration
- PCG + smoothed-aggregation AMG
- FGS1 momentum path
- no PETSc, HYPRE or MPI dependency

### Validated FP64 checkpoints

111,183-cell Re=20 convergence gate (`simpleTol=1e-3`):

- 407 SIMPLE iterations
- final continuity residual: `3.6751e-6`
- final momentum initial residuals: `[9.851e-4, 9.885e-4, 8.333e-6]`
- pressure drop: `16.0643006` vs exact `16.0000012`
- pressure-drop relative error: `0.40187%`
- 13.006 ms/SIMPLE, 8.549 MIUPS
- explicit VRAM: 275.7 MiB

Fixed-10 FP64 scaling:

| Cells | ms/SIMPLE | MIUPS | Explicit VRAM |
| ---: | ---: | ---: | ---: |
| 768,530 | 100.373 | 7.657 | 2004.3 MiB |
| 1,143,041 | 144.073 | 7.934 | 3003.8 MiB |
| 2,104,005 | 271.305 | 7.755 | 5583.8 MiB |

The existing `fp64_reference/` subtree is intentionally retained unchanged as
the earlier historical FP64 checkpoint. Current H8 FP32 and FP64 builds share
the live `gpu/serial_cuda/` source.

Validation runners:

- `RUN_H8_FP64_111K_CONVERGENCE.sh`
- `RUN_H8_FP64_FIXED10_3MESH.sh`
