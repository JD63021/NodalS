# NodalS vs OpenFOAM Foundation 12

This directory contains validation and performance comparisons between
NodalS and OpenFOAM Foundation 12 for the tetrahedral Hagen-Poiseuille
pipe test sequence.

The comparison uses matching meshes and corresponding physical boundary
conditions. The figures are intended to document the behaviour observed
for these particular validation cases rather than to claim universal
performance or accuracy advantages for either code.

## Figures

`pipe_results/velocity_convergence.jpeg`
: Velocity-error convergence with mesh refinement.

`pipe_results/pressure_convergence.jpeg`
: Pressure-error convergence with mesh refinement.

`pipe_results/outerits.jpeg`
: SIMPLE/SIMPLEC outer-iteration counts.

`pipe_results/wall_time.jpeg`
: Measured solution wall time.

`pipe_results/schemes.jpeg`
: Summary of the numerical schemes/configuration used in the comparison.

The bundled 40k pipe mesh under `meshes/pipe/40k` can also be used as a
small NodalS validation/example mesh.

See the main `README.md` for the NodalS discretization and solver
architecture.
