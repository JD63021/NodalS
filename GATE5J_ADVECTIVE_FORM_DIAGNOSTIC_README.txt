Gate 5J — Is the 5-slabs/D pressure/wall mismatch the advective-form term?
==============================================================================

This is a diagnostic-only rerun of the same Gate5I baseline:
  - 20D
  - 5 axial slabs/D (dz=0.2D)
  - 255,600 tets
  - SUPG OFF
  - weak Spalding wall
  - CF/PMIS + Cheb4
  - same SIMPLE settings

At the FINAL converged FE field, the solver evaluates:

  fAdvWeak
    Actual production advective convection action under a constant axial test,
    using the same central tensor as the assembled GPU operator.

  fAdvStrong
    Independent 125-point quadrature of u.grad(Uz) on the same P1+BF3 field.

  fConservativeStrong
    125-point quadrature of div(u Uz) on that SAME converged field.

  fUzDivU
    Difference conservative - advective = Uz div(u).

Therefore:
    fConservativeStrong = fAdvStrong + fUzDivU

The postprocessor compares:
    fPressure - fWall
against
    fAdvectiveDiscrete

If these match in 16-18D, the entire pressure/wall discrepancy is explained by
the advective-form convection contribution on the converged discrete field.

Workflow:
  1) Apply overlay.
  2) cd gpu/serial_cuda && make fp32
  3) ./RUN_RANS_GATE5J_20D_5SLABD_ADVECTIVE_DIAG_FP32.sh
  4) cd $HOME/NodalS_GPU_RANS
  5) postprocess/convection_form/RUN_GATE5J_ADVECTIVE_ERROR_CHECK.sh

Key output:
  NODALS_ADVECTIVE_ERROR_WINDOW ... z0D=16 z1D=18 ...

Interpretation:
  advectiveExplainsFraction ~= 1 and residualAfterAdvective ~= 0:
      yes, essentially all of the mismatch is the advective convection term.

  fConservativeSameField near 0 while fAdvectiveDiscrete is large:
      confirms the previously observed fact that conservative u⊗u evaluated
      on the advective-converged field looks much closer to the expected
      developed-pipe balance, without implying that solving with conservative
      convection is stable/accurate.
