# Custom pressure AMG — 40k repository example

This example uses the OpenFOAM tetrahedral pipe mesh already tracked in
`meshes/pipe/40k/constant/polyMesh`.  It exercises the three public custom-AMG
pressure modes:

- `pcg_unsmoothed`
- `pcg_smoothed`
- `richardson_smoothed`

Run from the repository root after building, or invoke the helper script here.
The cases use the validated Re=20 SIMPLE settings and converge to the normal
`1e-3` outer gate; they are intended as a lightweight integration regression.
