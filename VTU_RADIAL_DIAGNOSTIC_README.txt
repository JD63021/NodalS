VMFL003 radial diagnostic: 10 slabs/D vs 5 slabs/D
======================================================

Purpose
-------
This compares the two no-SUPG VTUs already produced:

  10 slabs/D:
    results_rans_20d_10slabD_nosupg/VMFL003_20D_10SLABD_NOSUPG_GATE5G_FP32.vtu

  5 slabs/D:
    results_rans_20d_5slabD_nosupg/VMFL003_20D_5SLABD_NOSUPG_GATE5H_FP32.vtu

Default developed window: 14D <= z < 18D.
This avoids most inlet development and the final outlet-affected region.

What is measured from the VTUs
-------------------------------
1. Radial local pressure-gradient/friction profile:
       f_D = 2 D (-dp/dz) / (rho Ubulk^2)
   using the same cell-center row-pair concept as the existing friction
   postprocessor.

2. Radial volume-weighted U0_z profile.

3. Difference U_z(5 slabs/D) - U_z(10 slabs/D).

4. Region summaries:
       core        r/R < 0.5
       mid         0.5 <= r/R < 0.8
       nearwall    r/R >= 0.8
       wall_touch  tetrahedra with at least one vertex on r=R
       all

5. Axial f(z) by radial region from 12D to 19.5D.

Interpretation limitation
-------------------------
These diagnostics show WHERE the discrepancy manifests. They do not by
themselves prove WHERE it originates.

A wall-law error can alter wall traction, global dp/dz and the entire developed
velocity profile, so a core discrepancy can still be wall-originated.

The current VTU contains U0 and p_P0 but not the exact weak-Spalding wall-face
trace, u_tau, y+, beta or tangent. A causal wall-vs-bulk test requires those
wall-face quantities to be exported by the solver at convergence.

Run
---
  cd "$HOME/NodalS_GPU_RANS"
  postprocess/vtu_radial_diagnostic/RUN_COMPARE_10D_5D_RADIAL.sh

Useful overrides:
  Z0_D=12 Z1_D=18 RADIAL_BINS=30 \
    postprocess/vtu_radial_diagnostic/RUN_COMPARE_10D_5D_RADIAL.sh

Outputs
-------
  developed_region_summary.csv
  radial_compare.csv
  axial_f_by_region_5slabD.csv
  axial_f_by_region_10slabD.csv
  radial_f_compare.png
  radial_uz_compare.png
  radial_delta_uz.png
  axial_f_regions_5slabD.png
  axial_f_regions_10slabD.png

If Python VTK is missing
------------------------
The wrapper will say so. Install either with your normal Python environment:
  python3 -m pip install --user vtk

or on Ubuntu/Debian, if appropriate:
  sudo apt install python3-vtk9
