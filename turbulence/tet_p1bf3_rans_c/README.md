# Tetrahedral P1+BF3/P0 RANS baseline — wall configuration C

This directory freezes the validated standalone tetrahedral turbulent-flow solver used for the VMFL003 campaign before further EMAC work.

## Frozen physics

- velocity/pressure pair: `[P1+BF3]^3/P0`
- segregated SIMPLE
- central **advective** convection (the historical stable baseline)
- Nikuradse-pipe algebraic mixing length, raw strain mode
- weak Spalding wall law
- wall sample mode: `legacy_face_trace`
- wall molecular consistency terms: **OFF**
- wall distance: `0.25 * (3V/A)`
- Spalding constants: `kappa=0.4`, `B=5.5`
- consistent-tangent wall linearization, `tangentBlend=0.5`
- SUPG off for the frozen baseline

The source is the conservation-audit version of the recovered historical solver. The momentum-budget diagnostic is optional and does not alter production physics when disabled.

Source SHA256:

`222d2fce915e7843b82bfc938abcc14d5106845af238366b4be05838cc7adf24`

## Reference VMFL003 setup

- D = 0.004 m
- R = 0.002 m
- Ubulk = 50 m/s
- nu = 1.460734693878e-5 m^2/s
- Re_D = 13691.7402481
- 10D mesh default: `~/Desktop/meshes/nodals_vmfl003_10D_radial460k/foam_case/constant/polyMesh`
- 20D mesh: `~/Desktop/meshes/nodals_vmfl003_20D_radial920k/foam_case/constant/polyMesh`

The runner exposes normal tuning variables as environment variables. For the 20D mesh use, for example:

`MESH=$HOME/Desktop/meshes/nodals_vmfl003_20D_radial920k/foam_case/constant/polyMesh SIMPLE_RTOL=1e-5 ./RUN_VMFL003_C_BASELINE.sh`

## Important baseline readings

On the 20D mesh at the tight `1e-5` SIMPLE gate, the recovered advective-C run reached 626 outer iterations, `wallFSpalding = 0.0255942888`, pressure drop `2205.162843784`, and mass-relative imbalance `2.60e-7`.

The exact P1-test momentum audit showed that the wall assembly itself agreed with the direct Spalding wall force to about 0.2%. The remaining downstream pressure resistance contained an advective contribution of roughly 8.5–10% of the pressure drive. This observation motivated the later EMAC experiments; it is not changed in this frozen baseline.

## Scope

This branch is a reproducible turbulent-flow reference. Do not merge experimental conservative or EMAC operators into this baseline without a separate validation campaign.
