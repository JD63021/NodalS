# NodalS SST V2 — case-driven developed inlet (no CUDA recompilation)

The native code already accepts `STAGE27_INLET_PROFILE_CSV`. It loads only Ux, Uy, Uz, and modifies the inlet DG flux and pressure RHS. Turbulence inlet k/omega remain uniform, even though the required CSV includes k and omega columns.

The native importer requires **one input row per inlet boundary vertex** with y/z coordinates matching within 1e-9 m. Current `reference_mesh/ubend_200k_for_audit.msh` has **196 inlet vertices and 338 triangular inlet faces**. The historic `inlet_profile_developed_it1500.csv` has **169 rows**, so it cannot be passed unmodified.

This update adds an optional Python preprocessing mode. No `.cu`, `.cuh`, `.inc`, Makefile or native executable is changed.

## Find your original CSV

```bash
find "$HOME/Downloads" "$HOME/NodalS_SST_LOCAL_RUNS" -type f \
  \( -name 'inlet_profile_developed_it1500.csv' -o -name 'inlet_profile_8col.csv' \) -print 2>/dev/null
```

Previous candidate path (verify file exists):
`/home/jd/Downloads/STAGE27_DG_PROFILE_V1_PACKAGE/inlet_profile_developed_it1500.csv`.

The exact CSV header is:
`x_m,y_m,z_m,Ux_m_s,Uy_m_s,Uz_m_s,k_m2_s2,omega_1_s`

## Edit examples/ubend_200k.case

Keep `mode=velocity` and the nominal `velocity` vector. Add to `[inlet.main]`:

```ini
profile_csv = /home/jd/Downloads/STAGE27_DG_PROFILE_V1_PACKAGE/inlet_profile_developed_it1500.csv
profile_mapping = interpolate
profile_flux = preserve
profile_corner_epsilon = 1e-9
```

`profile_mapping`: `exact` (default, existing strict 196-point CSV) or `interpolate` (linear y/z interpolation using SciPy Delaunay, preparing a mesh-matched 196-row CSV). An interpolated CSV is stored in `.cache/profiles/` with hashes of input profile, mesh, and relevant settings. The cache is non-mandatory and purely a reuse optimization.

`profile_flux`: `preserve` (do not rescale interpolated velocities) or `match_bulk` (scale Ux/Uy/Uz uniformly so inlet flux matches the nominal constant-vector inlet flux). Always examine the printed actual inlet flux, as interpolation can slightly change the original precursor's flux.

`profile_corner_epsilon`: nonnegative velocity magnitude in m/s, default 1e-9. The native loader requires strictly inward face flux. On this unstructured square inlet, **two corner triangles** can have all three vertices exactly at wall-zero velocity. The Python preprocessor adds a documented 1e-9 m/s inward bias **only to vertices of such all-zero faces**, rather than changing the CUDA code. This is a tiny deviation from exact edge zero and a selectable preprocessing tolerance. Set it to 0 to refuse that tiny correction (the old native importer will then reject perfectly zero corner faces). Actual reverse-flow faces are rejected, not silently corrected.

`mode = velocity` and `velocity` remain the nominal bulk Re/nu reference and the inlet direction used by the native solver's startup and pressure RHS. The provided velocity CSV overrides the uniform **boundary** profile.

## Verify and run

```bash
cd ~/Downloads/NodalS_SST_SOFTWARE_V2
python3 -c 'import numpy, scipy; print("NumPy/SciPy OK")'
python3 -u nodals.py run --case examples/ubend_200k.case --dry-run
python3 -u nodals.py run --case examples/ubend_200k.case
RUN=$(ls -td runs/ubend_200k_* | head -1)
grep -E 'STAGE27_DG_PROFILE=ON|STAGE27_DG_PROFILE_PRESSURE|STAGE27_DG_PROFILE_GPU' "$RUN/terminal.log"
```

Do **not** use a previous PLUG-INLET checkpoint for a direct developed-inlet restart: it would represent a changed boundary condition. Start fresh to compare against plug.

The `.case` source path is resolved relative to the case file unless absolute; the preprocessed path passed to the native solver is always absolute. `--dry-run` may prepare the CSV cache but does not invoke CUDA or create a simulation run directory.

Validation performed here: 11 CPU-side software tests, including synthetic 169→196 mapping and preserve/match_bulk flux options. The user's original CSV and GPU run have not been tested in this environment.
