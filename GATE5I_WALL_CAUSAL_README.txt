Gate 5I: 5-slabs/D wall-causality experiment
=============================================

Purpose
-------
Run exactly the same 20D, 5-slabs/D, no-SUPG RANS case twice:

1) baseline:
   existing weak Spalding wall law.

2) reference_shear:
   force the wall to apply the ideal reference Darcy friction f_D=0.0284003.
   The corresponding target is:
      u_tau = Ubulk*sqrt(f_D/8) = 2.97910955656 m/s
      u_tau^2 = 8.87509375
   The GPU assembles a lagged Picard Robin coefficient
      beta_ref = u_tau^2 / max(|U_trace|, slipFloor)
   so the converged wall traction magnitude is exactly the reference shear.

This is a causal control, not a production wall model.

Final wall-face diagnostic
--------------------------
Both runs write one row per wall face with:
  z/D, area, sample distance y,
  U_trace, slip,
  Spalding-inferred u_tau, u_tau^2, y+, beta and f,
  actually-applied u_tau, u_tau^2, beta and f,
  wall mode.

Thus, in the forced run we can ask:
  - is f_pressure driven to f_reference?
  - what Spalding friction would the resulting velocity trace imply?
  - how much does the wall slip/profile need to change?

Install
-------
Extract this package into $HOME/NodalS_GPU_RANS and rebuild fp32.

Run
---
Baseline:
  gpu/serial_cuda/RUN_RANS_GATE5I_20D_5SLABD_BASELINE_DIAG_FP32.sh

Forced reference shear:
  gpu/serial_cuda/RUN_RANS_GATE5I_20D_5SLABD_REFERENCE_SHEAR_FP32.sh

Postprocess both:
  postprocess/wall_causal/RUN_GATE5I_WALL_CAUSAL_COMPARE.sh

Interpretation
--------------
If forced reference shear makes developed f_pressure ≈ 0.0284003, while the
baseline gives its natural Spalding value, the bulk operator is capable of
transmitting the correct drag and the axial-resolution sensitivity is entering
through the wall trace / wall-law feedback.

If forced f_wall=0.0284003 but f_pressure remains materially different in the
developed region, then the bulk momentum balance/discretization contributes an
independent error.

Caveat
------
The final 18-20D window is outlet-sensitive. 14-18D is the cleanest comparison.
