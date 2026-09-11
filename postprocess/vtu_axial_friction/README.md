# VTU axial cell-center friction postprocessor

This tool is intentionally separate from the solver. It consumes a NodalS ASCII
VTU containing tetrahedra and `CellData/p_P0`.

For an extruded pipe mesh, tetrahedron centroids are grouped into transverse
`(x_c,y_c)` row keys and sorted by `z_c`. Adjacent cell centers in each row give

`pressure_loss_gradient = (p_up - p_down) / (z_down - z_up)`

and the local Darcy factor

`f_D = 2 D pressure_loss_gradient / (rho U_bulk^2)`.

For current NodalS incompressible pressure use `rho=1`. The script also solves
the Colebrook equation for the Moody friction factor at the supplied Reynolds
number and relative roughness, then writes pair-level and axially binned CSVs.

VMFL003 20D example:

```bash
python3 postprocess/vtu_axial_friction/analyze_vtu_axial_friction.py \
  gpu/serial_cuda/results_rans_20d/VMFL003_20D_GATE5_FP32.vtu \
  --diameter 0.004 --bulk 50 --rho 1 \
  --re 13691.7402481 --rel-roughness 0 \
  --reference-f 0.0284003
```

Primary outputs are `cell_center_pairwise_dp_dz.csv`,
`axial_friction_profile.csv`, and `axial_friction_profile.png`.
