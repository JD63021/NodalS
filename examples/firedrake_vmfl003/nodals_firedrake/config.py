"""VMFL003 benchmark configuration.

The values here are deliberately explicit: this repository is intended as a
small, reproducible Firedrake implementation of the Q1+BF2/Q0 benchmark, not
as a general CFD framework.
"""
import os
from firedrake import dx, ds_b
from petsc4py import PETSc

Print = PETSc.Sys.Print

# Geometry / mesh
R = 0.002
D = 2.0 * R
L = 2.0
NQ = 12
NR = 1
NZ_FINE = 20
NZ_LONG = 48
NZ = NZ_FINE + NZ_LONG
DZ_FINE = D
DZ_LONG = 10.0 * D
EXPECTED_CELLS = 13056
EXPECTED_SCALAR_DOF = 55965

# Fluid / benchmark
NU = 1.4607346938775508e-5
RHO = 1.225
U_BULK = 50.0
RE = U_BULK * D / NU

# Exact integration used by the production port
QUADRATURE_DEGREE = 7
DXQ = dx(metadata={"quadrature_degree": QUADRATURE_DEGREE})
DSBQ = ds_b(metadata={"quadrature_degree": QUADRATURE_DEGREE})

# SIMPLEC
ALPHA_U = 0.7
ALPHA_P = 0.3
RAU_SCALE = 2.0
SIMPLEC_BLEND = 1.0
SIMPLEC_FLOOR_FRACTION = 1.0e-6

# Turbulence / wall law
MIXLEN_SCALE = 1.0
WALL_KAPPA = 0.4
WALL_B = 5.5
WALL_BETA_SCALE = 1.0
WALL_SAMPLE_FRACTION = 0.5
WALL_MOLECULAR_CONSISTENCY = False

# Nonlinear and linear-solver defaults. Environment variables allow rerunning
# sensitivity checks without editing source.
SIMPLE_TOL = float(os.environ.get("SIMPLE_RTOL", "1e-4"))
MAX_OUTER = int(os.environ.get("SIMPLE_MAX_ITS", "10000"))

M_RTOL = float(os.environ.get("M_RTOL", "0.5"))
M_PETSC_ATOL = float(os.environ.get("M_PETSC_ATOL", "1e-50"))
M_MAX_ITS = int(os.environ.get("M_MAX_ITS", "20000"))

P_RTOL = float(os.environ.get("P_RTOL", "0.5"))
P_PETSC_ATOL = float(os.environ.get("P_PETSC_ATOL", "1e-50"))
P_MAX_ITS = int(os.environ.get("P_MAX_ITS", "300"))
P_MG_LEVEL_ITS = int(os.environ.get("P_MG_LEVEL_ITS", "6"))

# Public benchmark references
VMFL003_TARGET_DP_PA = 21744.0
SMOOTH_MOODY_FD = 0.0284587039

# Loose regression gate for a refactor/reproduction run. This is intentionally
# far wider than the verified benchmark error (~0.004%).
BENCHMARK_DP_TOL_PCT = float(os.environ.get("BENCHMARK_DP_TOL_PCT", "0.5"))
