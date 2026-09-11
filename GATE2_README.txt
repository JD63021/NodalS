NodalS GPU RANS Gate 2: DG numerical-trace plug inlet

Scope:
- keeps Gate-1 Nikuradse mixing length active
- releases inlet P1/BF3 velocity entities from strong Dirichlet elimination
- retains the strong no-slip wall unchanged
- adds the CPU-RANS DG weak inlet momentum form:
    inflow + nonsymmetric Nitsche, nuEff = nu + lagged nu_t
- changes the canonical pressure coupling to
    B_eff = B_volume - DG inlet face trace
  with exact P1 A/3 and BF3 9A/20 trace coefficients
- the same effective-B feeds B, B^T, fine Schur refresh and AMG hierarchy setup
- initializes free P1 velocity vertices to the prescribed plug; BF3 starts at zero
- SUPG remains OFF
- no O(N) host/device transfer is introduced inside SIMPLE

This gate intentionally retains the strong wall. The weak Spalding wall is Gate 3,
so exact constant-plug cancellation at inlet/wall edge vertices is not required yet.

Build:
  cd $HOME/NodalS_GPU_RANS/gpu/serial_cuda
  make clean && make fp64 && make fp32

Run first in FP64:
  PREC=fp64 ./RUN_RANS_GATE2_10D.sh

Gate-2 PASS target:
- setup reports dgInlet=1 and 852 inlet faces
- run completes fixed10, finite, pressureAll=PASS
- NODALS_GPU_H8_RESIDENCY retains O_N_H2D_inside_SIMPLE=0 and O_N_D2H_inside_SIMPLE=0
- pressure/AMG operates normally with the DG effective-B coefficients

Do not judge final pressure drop/friction at Gate 2: the strong wall is intentionally
still present. Gate 3 adds the validated weak Spalding wall function.
