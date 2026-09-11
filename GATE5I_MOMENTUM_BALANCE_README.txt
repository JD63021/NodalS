Gate 5I momentum-balance postprocessor
=======================================

Purpose
-------
Close the approximate axial momentum balance for the two already-computed
5-slabs/D Gate 5I cases:

  baseline weak Spalding wall
  forced reference wall shear

The integrated steady straight-pipe balance is

  f_pressure = f_wall + f_development

with

  f_development = 2 d(beta)/d(z/D)

and

  beta = integral_A(Uz^2)dA / (A Ubulk^2).

VTU limitation
--------------
The VTU stores exact cell-average U0, not the full P1+BF3 velocity polynomial.
The script therefore estimates beta from

  sum(Vcell * U0_z^2) / (Vslab * Ubulk^2).

This is not exact quadrature of Uz^2. It is best used as a causal/comparative
diagnostic between the baseline and forced-wall cases on the identical mesh.

Run
---
cd "$HOME/NodalS_GPU_RANS"
postprocess/momentum_balance/RUN_GATE5I_MOMENTUM_BALANCE.sh

Important terminal records
--------------------------
NODALS_MOMENTUM_BALANCE_WINDOW
NODALS_MOMENTUM_BALANCE_DELTA

The most useful windows are 14-16D, 16-18D, and the aggregate 12-18D.
The final 18-19.8D window remains outlet-sensitive.

Interpretation
--------------
If

  fPressure ~= fWallApplied + fDevelopment

with a small closure residual, then the "extra" pressure gradient seen in the
forced-wall run is explained by continued profile development rather than a
separate bulk momentum defect.

If the residual remains large compared with fReference, then either:
  (a) there is a genuine discrete bulk momentum-balance defect, or
  (b) the cell-average-squared VTU approximation is insufficient.

A future exact diagnostic would integrate Uz^2 from the actual P1+BF3
coefficients during/after the solve.
