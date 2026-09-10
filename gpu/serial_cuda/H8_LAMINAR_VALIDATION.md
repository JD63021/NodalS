# H8 laminar GPU validation and production freeze

Status: **LOCAL VALIDATION PASS**
Date: 2026-09-10
Scope: serial single-GPU H8 laminar path, FP32/FP64 setup and runtime; Re=20 FP64 Hagen–Poiseuille accuracy.

This file is intentionally retained in the repository as handoff context for future development. It records the numerical and performance gates used to freeze the optimized laminar GPU path.

## Production architecture

The steady SIMPLE loop is device-resident. Host-side work is retained for mesh-fixed/symbolic setup and SA hierarchy construction; compact final structures are uploaded once. The fine pressure operator remains explicit CSR because its GPU action is faster than the matrix-free alternative.

Reconstruction/setup optimizations do not change the physical discretization.

Production defaults after Gates 1–7:

- parallel A1 first-coarse construction;
- parallel P0 transfer;
- parallel recursive SA transfer/PtAP;
- parallel momentum CSR construction;
- parallel `G4CellPlanHost` construction;
- parallel fine-pressure CSR topology construction;
- setup-time autotuning disabled in production;
- FP32 H7 convection policy: `WARP_CELL`, SM factor 4;
- FP64 H7 convection policy: `SCALAR_THREAD_CELL`;
- momentum finalization: `SCALAR_THREAD_ROW`;
- fine pressure refresh / B-transpose / B action / continuity: precomputed B;
- explicit coarse AMG CSR levels: `WARP_ROW`;
- GPU spectral refresh remains enabled at setup;
- H2 action parity/benchmark is diagnostic-only.

Diagnostic setup autotuning is retained with:

`NODALS_SETUP_AUTOTUNE=1`

Intentional setup-only execution is retained with:

`NODALS_H8_SETUP_ONLY_EXPLICIT=1`

The Hagen–Poiseuille error diagnostic downloads the final state only after the SIMPLE loop; it is not part of the steady GPU iteration.

## Sparse/setup gate results

At 2,104,005 cells:

- Gate 5 fine-CSR topology, FP32: reference ~4.60 s -> parallel ~0.68 s, exact row/column/diag/incidence/contribution/slot parity.
- Gate 5 fine-CSR topology, FP64: reference ~4.57 s -> parallel ~0.67 s, exact parity.
- SA hierarchy after Gates 1–3: ~4.33 s.
- G4 setup after Gates 4A/4B: ~1.42 s.
- pre-SIMPLE after Gate 6/7: ~9.87 s FP32 and ~10.00 s FP64.

The initial pre-optimization setup was roughly 42.3 s FP32 and 47.4 s FP64 on the same 2M case.

## Gate 7: 2M real fixed-10 SIMPLE smoke

### FP32

- cells: 2,104,005
- pre-SIMPLE: 9.867850508 s
- 10-step loop: 1745.668 ms
- average SIMPLE: 174.566848 ms
- average pressure iterations: 2.9
- after-upload explicit GPU storage: 3592.318 MiB
- explicit bytes/cell: 1790.309
- sampled process GPU peak: 3770 MiB
- runtime GPU-memory drift: 0.000 MiB
- final H8 fixed10 result: PASS

### FP64

- cells: 2,104,005
- pre-SIMPLE: 10.001173339 s
- 10-step loop: 2724.667 ms
- average SIMPLE: 272.466680 ms
- average pressure iterations: 2.9
- after-upload explicit GPU storage: 5583.750 MiB
- explicit bytes/cell: 2782.781
- sampled process GPU peak: 5744 MiB
- runtime GPU-memory drift: 0.000 MiB
- final H8 fixed10 result: PASS

The fixed-10 run is a runtime/memory smoke only; its pressure drop is intentionally not treated as a converged physics result.

## Re=20 FP64 Hagen–Poiseuille accuracy validation

Physics:

- parabolic inlet;
- central convection;
- SUPG off;
- SIMPLE;
- `alphaU = 0.5`;
- `alphaP = 0.5`;
- momentum work: `fgs1`;
- pressure inexact target: rtol 0.5, atol 1e-12, max 20;
- H8 convergence gate: `simpleTol = 1e-6`;
- physical pressure operator: current exact CSR;
- final error quadrature: H8 `duffy5_125`.

### 111,183 cells

- `hEff = 2.066904832921e-03`
- outer SIMPLE iterations: 1160
- final relative continuity: `1.525877322431e-08`
- final momentum initial residuals: `[9.972e-07, 5.372e-07, 4.369e-09]`
- average pressure iterations: 1.228448
- average SIMPLE time: 11.819312 ms
- `U_L2 = 2.708504671785e-04`
- `U_relL2 = 7.486180766004e-03`
- `P_shifted_L2 = 9.975398232264e-04`
- `P_shifted_relL2 = 3.450241265911e-03`
- fitted pressure drop: `1.605600487847e+01`
- exact pressure drop: `1.600000117725e+01`
- pressure-drop relative error: `3.500231068750e-03`
- result: PASS

### 292,236 cells

- `hEff = 1.497690447682e-03`
- outer SIMPLE iterations: 2084
- final relative continuity: `9.129250078978e-09`
- final momentum initial residuals: `[9.968e-07, 9.003e-07, 2.832e-09]`
- average pressure iterations: 1.061900
- average SIMPLE time: 30.544371 ms
- `U_L2 = 1.382819325279e-04`
- `U_relL2 = 3.822047898243e-03`
- `P_shifted_L2 = 6.698904699072e-04`
- `P_shifted_relL2 = 2.315737099644e-03`
- fitted pressure drop: `1.602778174349e+01`
- exact pressure drop: `1.600000000000e+01`
- pressure-drop relative error: `1.736358968037e-03`
- result: PASS

### 111k -> 292k pair order

Using `p = ln(E111/E292) / ln(h111/h292)`:

- velocity L2 order: **2.086972**
- velocity relative-L2 order: **2.086973**
- shifted-pressure L2 order: **1.236086**
- shifted-pressure relative-L2 order: **1.237757**
- fitted pressure-drop absolute-error order: **2.176274**

Interpretation: the velocity field and integral pressure drop retain essentially second-order convergence. The P0 pressure L2 error converges at the expected lower order and decreases monotonically.

## Historical strict Re=20 40k anchor

A prior strict FP64 CPU/PETSc result on 40,620 cells gave:

- `U_L2 = 5.369827349215e-04`
- `P_shifted_L2 = 1.632459350213e-03`
- fitted pressure drop = `1.611709824860e+01`

Together with the new H8 111k and 292k results, both velocity and shifted pressure errors decrease monotonically with refinement, while the fitted pressure drop approaches 16.0 at approximately second order.

## Deferred validation

The 7M/A100 production audit is intentionally deferred because the A100 was not available during this freeze. Run it when hardware becomes available; do not reinterpret its absence as a failure of the local laminar freeze.

## Freeze decision

The serial H8 laminar GPU path is accepted for integration based on:

1. exact symbolic/parity gates for the setup optimizations;
2. stable FP32 and FP64 2M fixed-10 runtime smoke;
3. zero runtime GPU-memory drift in Gate 7;
4. strict converged FP64 Re=20 Hagen–Poiseuille validation on 111k and 292k;
5. ~second-order velocity and pressure-drop convergence.

Future changes to the frozen laminar default should preserve these checks or document why a replacement gate is stronger.
