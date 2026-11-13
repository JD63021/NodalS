# Experimental Option A: OpenFOAM-like no-slip wall-layer conductance (P1+BF3)

**Status, 2026-11-13:** ISOLATED DESIGN + LOCAL BASELINE-PREFLIGHT. The native GPU
Option-A wall operator is **not yet wired**. No modified turbulent solve has been run,
and no performance / Moody accuracy improvement is claimed. The baseline branch
remains unchanged. Do not treat a run using the existing Robin kernel as an Option-A run.

## Objective

On the **333,779-tet 50D Delaunay pipe mesh**, first reproduce the confirmed
Re = 72,600 SST reference with the unchanged solver, and then compare a new momentum
wall treatment to it. This is **not** a Re=500,000 validation experiment.

Existing geometry (workstation):
`/home/jd/Downloads/NODALS_PIPE50_3D_DELAUNAY_20261111/PIPE50_3D_DELAUNAY.msh`.
The previously prepared case may be named
`PIPE50_D3D_5PT_WIDE_5TO300.case`. If missing, start from the
earlier `PIPE_50D_MOODY.case` and use the mesh override.

## Constraints

- Separate experimental Git branch; no changes to baseline code or existing local runs.
- Preserve the P1+BF3 velocity element, SST 2003m, scalar k/omega treatment,
  SIMPLE, FGMRES/AMG and existing Spalding five-point fit initially.
- Preserve the existing turbulent volume viscosity away from the wall layer.
- Impose **all three components** of velocity as zero on the physical wall
  vertices *and* wall-face BF3 coefficient. Keep the DOF storage; enforce
  proper fixed masks in predictor, preconditioner, Schur, and pressure correction.
- Do **not** simultaneously retain tangential Spalding Robin or normal
  Nitsche wall terms from the old wall operator.
- The model wall flux is computed from lagged u_tau and the actual
  discrete wall-to-interior velocity gradient.
- Avoid duplicate diffusion: replace the unresolved wall-normal contribution
  to the existing viscous+SST operator; do not merely add a new positive penalty.
- For deviatoric stress, derive the replacement from the **same constitutive
  operator** as momentum, not an unrelated scalar Laplacian.
- Correct handling for owner tets that have more than one boundary face;
  do not subtract the same volume subspace twice. Any overlapping wall layers
  must have a partition of unity or be rejected at validation.
- Keep the viscous limit exact: as u_tau^2 = nu*U_s/y_s, wall correction -> zero.
- No model changes to k or omega for the first A/B comparison.
- Do not force under-relaxation or SUPG changes to disguise model discrepancies.

## One-dimensional consistency test

For an affine interval [0,h] with no-slip at 0, fluid velocity U at h,
the existing viscous element contributes K_base = A*nu_base/h.

If Spalding supplies u_tau from a resolved off-wall velocity U_s at
y_s=t*h and the 1-D FE field is linear, U_s=t*U, then the desired
wall shear is tau/rho=u_tau^2 and the **replacement** wall-normal
conductance is K_model=A*u_tau^2/U. Equivalently, in terms of the sample:

  K_model = A*(u_tau^2/U_s)*(y_s/h).

Thus K_new = K_base - K_base + K_model.
Using an unscaled sampled-basis penalty beta*U_s*V_s is NOT this
replacement and is NOT automatically OpenFOAM-equivalent.

For P1+BF3 and unstructured tetrahedra, extend the identity using the
actual FE gradients/test functions, wall-face quadrature, a consistent
modelled transfer and a non-overlapping wall-layer partition. A physically
plausible wall nu_t alone is insufficient if it is assembled twice.

## Acceptance tests before 500k

1. Unmodified 72.6k reference is reproducible on **the same 333,779-tet mesh**,
   with a recorded .case, binary SHA, and postprocessing result.
2. Exact trace: max |u_h(x_wall)| at 12 face quadrature points is < 1e-12 U_b;
   all wall-face BF3 traces and pressure-correction increments respect this.
3. Integrated wall flux is zero to numerical tolerance.
4. Viscous-limit patch test: no difference from baseline molecular FEM stiffness
   once u_tau^2=nu U_s/y_s; no extra Robin friction.
5. Tangential wall flux, modelled face stress and 20D-50D pressure gradient
   agree via the integrated axial momentum balance; compare against Moody.
6. Monitor Δp, wall u_tau, wall nu_t, k, omega, max nu_t/nu, sampling containment,
   and physical momentum/continuity residuals for at least the same SIMPLE span.

Run the isolated workstation discovery + baseline preflight with
`bash experiments/of12_nutwall_p1bf3/PREPARE_LOCAL.sh`.
Set `RUN_BASELINE=1` only when a known-good *existing* solver binary is found;
this is still the original Robin model, not Option A.
