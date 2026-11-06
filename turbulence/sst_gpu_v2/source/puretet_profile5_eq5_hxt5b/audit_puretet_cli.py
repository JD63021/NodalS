#!/usr/bin/env python3
"""Print current health indicators; never label an unconverged run converged."""
import pathlib, re, sys
if len(sys.argv) != 2:
    sys.exit('Usage: python3 audit_puretet_cli.py PATH_TO_RUN_LOG')
p = pathlib.Path(sys.argv[1])
if not p.is_file():
    sys.exit(f'ERROR: log missing: {p}')
rows = []
for line in p.read_text(errors='replace').splitlines():
    if 'NODALS_SST_G2 it=' not in line:
        continue
    vals = dict(re.findall(r'\b([A-Za-z][A-Za-z0-9]*)=([^\s]+)', line))
    try:
        rows.append({k: float(vals[k]) for k in (
            'it','absCont','kRelUpdate','omegaRelUpdate','nutMeanRelUpdate',
            'kFloorVolumeFrac','omegaFloorVolumeFrac','wallDarcy','wallYPlusMean',
            'pressureDrop','meanNutOverNu','maxNutOverNu')})
    except (KeyError, ValueError):
        continue
text = p.read_text(errors='replace')
if not rows:
    print('PURETET_CLI_AUDIT noCompletedG2Iterations=1')
    sys.exit(0)
first, last = rows[0], rows[-1]
print(f"PURETET_CLI_AUDIT completed={len(rows)} lastIt={int(last['it'])}")
for k in ('absCont','kRelUpdate','omegaRelUpdate','nutMeanRelUpdate',
          'kFloorVolumeFrac','omegaFloorVolumeFrac','wallDarcy','wallYPlusMean',
          'pressureDrop','meanNutOverNu','maxNutOverNu'):
    print(f'  {k}: first={first[k]:.6g} last={last[k]:.6g}')
ks, ws = last['kFloorVolumeFrac'], last['omegaFloorVolumeFrac']
if ks >= .01 or ws >= .01:
    print('PURETET_CLI_HEALTH floorFractionsOver1Percent=YES status=WARNING')
else:
    print('PURETET_CLI_HEALTH floorFractionsOver1Percent=NO status=PROVISIONAL')
for m in ('SST_G2_GATE_STATUS=PASS','SST_G2_GATE_STATUS=FAIL',
          'HXT5B_GATE_STATUS=PASS','HXT5B_GATE_STATUS=FAIL'):
    if m in text:
        print('PURETET_CLI_GATE ' + m)
if 'SST_G2_GATE_STATUS=PASS' not in text:
    print('PURETET_CLI_CONVERGENCE demonstrated=NO')
