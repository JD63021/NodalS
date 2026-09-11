NodalS GPU RANS Gate 1 overlay
==============================

Scope:
- Nikuradse smooth-pipe mixing-length eddy viscosity only.
- 64-point CPU-matching collapsed quadrature.
- Raw strain sqrt(2 S:S).
- Lagged current SIMPLE velocity.
- Adds only int(nu_t grad(phi_a).grad(phi_b)) to the shared momentum CSR.
- Fixed-column nu_t terms go into existing c0/c1/c2 dynamic RHS buffers.
- Existing inlet and strong wall are unchanged.
- SUPG remains off.
- Fine pressure Schur continues to refresh from live rAU each SIMPLE iteration.
- No O(N) host/device transfer is introduced inside SIMPLE.

Apply from repo root:
  tar -xzf NodalS_GPU_RANS_GATE1_OVERLAY.tar.gz -C "$HOME/NodalS_GPU_RANS"

Build:
  cd "$HOME/NodalS_GPU_RANS/gpu/serial_cuda"
  make clean
  make fp64
  make fp32

Run zero-scale wiring regression first:
  CASE=scale0 PREC=fp64 ./RUN_RANS_GATE1_10D.sh

Then active FP64:
  CASE=active PREC=fp64 ./RUN_RANS_GATE1_10D.sh

Then active FP32 after FP64 is finite:
  CASE=active PREC=fp32 ./RUN_RANS_GATE1_10D.sh
