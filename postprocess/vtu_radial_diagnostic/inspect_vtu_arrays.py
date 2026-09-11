#!/usr/bin/env python3
import sys
from pathlib import Path

try:
    import vtk
except Exception as exc:
    raise SystemExit(f"ERROR: Python VTK unavailable: {exc}")

if len(sys.argv) != 2:
    raise SystemExit("usage: inspect_vtu_arrays.py file.vtu")

p = Path(sys.argv[1])
r = vtk.vtkXMLUnstructuredGridReader()
r.SetFileName(str(p))
r.Update()
g = r.GetOutput()

print(f"VTU={p}")
print(f"points={g.GetNumberOfPoints()} cells={g.GetNumberOfCells()}")

for label, data in [("PointData", g.GetPointData()), ("CellData", g.GetCellData()), ("FieldData", g.GetFieldData())]:
    print(label)
    for i in range(data.GetNumberOfArrays()):
        a = data.GetAbstractArray(i)
        if a is None:
            continue
        name = a.GetName() or f"<unnamed_{i}>"
        nc = a.GetNumberOfComponents()
        nt = a.GetNumberOfTuples()
        print(f"  {name}: components={nc} tuples={nt}")
