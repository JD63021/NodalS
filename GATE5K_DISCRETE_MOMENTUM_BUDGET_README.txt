Gate 5K — Exact discrete z-momentum budget on the 5-slabs/D baseline
============================================================================

Purpose
-------
Close the final axial momentum equation using the actual assembled operators
rather than reconstructing terms from the VTU.

The run is the SAME baseline physics as Gate5H/Gate5I/Gate5J:
  - 20D pipe
  - 5 axial slabs per D (dz=0.2D), 255,600 tets
  - Nikuradse mixing length
  - weak Spalding wall
  - DG plug inlet
  - SUPG OFF
  - SIMPLE alphaU=0.7, alphaP=1, rAU scale=2
  - CF/PMIS + Cheb4 pressure AMG
  - FGS1 momentum work

What is captured
----------------
At the start of every SIMPLE outer iteration, diagnostic mode stores DEVICE
snapshots of:
  U_old = velocity used to lag nu_t, wall beta/Jacobian and advector
  p_mom = pressure used by that iteration's momentum sweep

There are no extra H2D/D2H transfers inside SIMPLE; these are D2D snapshots.

At the end, the diagnostic evaluates the final linearization:

  molecular diffusion
  + lagged nu_t diffusion
  + weak-wall Robin operator including tangent compensation
  + DG inlet operator
  + advective convection
  - B^T p_mom

It also records:
  - B^T p_final after the last pressure correction
  - under-relaxation lag delta*(U_new-U_old)
  - exact residual of the final relaxed momentum linear system

The spatial test is a smooth P1 sine window in z, with BF3 test coefficient 0:
  w(z)=sin(pi*(z-z0)/(z1-z0))
inside each requested window and zero at/outside its endpoints.

This is a valid continuous FE test and avoids a discontinuous window cutoff.
All terms are normalized to an equivalent Darcy friction contribution.

Key identity
------------
For the pressure actually used by the last momentum sweep:

  f_molecular + f_nut + f_wall + f_DG + f_advective
  - f_pressure_momentum
  = f_physical_residual

and

  f_physical_residual + f_relaxation_lag
  = f_relaxed_linear_residual.

The component/direct parity check verifies the split against the actual final
assembled matrix/rhs.

Workflow
--------
1) Apply this overlay.
2) Rebuild:
     cd "$HOME/NodalS_GPU_RANS/gpu/serial_cuda"
     make fp32

3) Run:
     ./RUN_RANS_GATE5K_20D_5SLABD_DISCRETE_BUDGET_FP32.sh

4) Postprocess:
     cd "$HOME/NodalS_GPU_RANS"
     postprocess/discrete_budget/RUN_GATE5K_DISCRETE_BUDGET_COMPARE.sh

Most important line
-------------------
Inspect 16-18D:

  NODALS_DISCRETE_BUDGET_COMPARE z0D=16.00 z1D=18.00 ...

This will tell whether the ~0.00128 remaining after the advective term is:
  - molecular/nu_t axial diffusion,
  - a difference between pairwise dp/dz and the actual discrete B^T p,
  - the final SIMPLE pressure correction,
  - momentum under-relaxation / one-sweep residual,
  - or something else in the exact operator budget.
