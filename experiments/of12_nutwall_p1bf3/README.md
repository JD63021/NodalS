# Experimental Option A: OpenFOAM-like no-slip wall-layer conductance (P1+BF3)

**Status (2026-11-13):** The native Option-A GPU wall operator has now been wired into the isolated GitHub branch, but it has NOT been compiled or run on the user's CUDA workstation. No numerical improvement is claimed. The existing Robin model is unchanged when the new mode is off. This first version uses an isotropic replacement of viscosity in the wall-owner tetrahedron, not an exact OpenFOAM finite-volume face operator.

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


## Structured 99,552-tetrahedron Re72600 trial (current reference)

Use the already-prepared folder:

    ~/Downloads/PIPE50_STRUCTURED_RE72600_100K_20261113/

The mesh has 11,712 wall triangles, each with a UNIQUE owner tetrahedron
(validated directly from the uploaded Gmsh mesh connectivity).

Run in the existing experimental worktree:

    cd ~/NodalS_OF12_WALL
    git pull --ff-only
    bash experiments/of12_nutwall_p1bf3/RUN_STR100K.sh baseline 10
    bash experiments/of12_nutwall_p1bf3/RUN_STR100K.sh option_a 10

The runner builds a branch-local sm_86 CUDA executable. The two runs use
the same source case and exact mesh; they differ only by
    --set wall_function.momentum_bc=robin
or
    --set wall_function.momentum_bc=of12_noslip

Option A cannot silently fall back: the launcher verifies the executable
printed 'NODALS_OPTION_A enabled=1'. If compilation fails, retain the
compiler log and stop; do not run using an older Robin-only executable.

Implementation: wall P1 vertices and BF3 face bubbles are fixed to zero
for all velocity components. Scalar k/omega retain their original free
wall masks. The original tangential Robin and wall-normal Nitsche
contributions, including cached F1B and matrix-free cross-actions,
are disabled. F2 five-point Spalding still supplies u_tau.
The owner tet uses nu_eff_wall = max(nu, u_tau^2 / |P_t grad_n u|)
in the SST momentum turbulent diffusion and the implicit deviatoric
volume operator. The existing molecular base is kept, and no additional
Robin face force is applied.

LIMITATION: this is a cellwise isotropic wall-layer viscosity replacement;
it modifies more of the first tetrahedron than the FV wall-face flux.
The one-dimensional analytical patch test passes, but native GPU
compilation, actual 3D patch tests, no-slip audit and pressure-drop
comparison are PENDING. Do not infer accurate wall shear from the old
Robin-based wall diagnostic; inspect the new NODALS_OPTION_A_WALL output.


## V2 correction after the diverging 1000-step attempt (2026-11-13)

The original Option A v1 failed at approximately it=220 with growing momentum
residuals. It had replaced **the complete isotropic viscosity** inside
11,712 wall-owner tetrahedra, whereas the intended OF-like model should
alter only their wall-normal shear. The prior full-tetra replacement is
**retired** from experimental compilation (existing checkpoint remains
unchanged).

The V2 formulation retains original SST diffusion and the complete
implicit deviatoric tensor, and adds a **signed replacement**, not another
wall penalty:

    g(U) = (I-n*n^T) ((grad U + grad U^T) n)
    delta_a(U,V) = integral_owner_tet (nu_wall - nu_eff_SST) g(U).g(V) dV

The scalar case remains masked Q1, and its wall masks are never replaced
by the velocity strong-no-slip mask. The V2 correction is implemented in
both the M3 nine-block CSR and the reference matrix-free operator, before
BF3 T/Tt elimination. The original wall Robin and normal Nitsche terms
remain zero in Option A. This produces a positive total viscous shear
coefficient nu_wall even when nu_wall < nu_eff_SST.

A wall-face fail-fast rejects a non-finite or excessive (>1e6*nu) fitted
nuwall coefficient; this is **not** a silently clipped modelling choice.

**Important**: As of this source commit, V2 is **not yet compiled or
validated on NVIDIA hardware**. Start with 10 or 20 iterations and inspect
`NODALS_OPTION_A_WALL`, `NODALS_OPTION_A_NOSLIP_AUDIT`, and M3 parity
before another longer run. Do not restart the v1 checkpoint using the V2
formulation; the frozen nonlinear state may be incompatible.
