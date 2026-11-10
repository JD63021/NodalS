# NodalS Experimental Multicore CPU k-omega SST

Research implementation of steady incompressible RANS using
P1+BF3 velocity, P0 pressure and masked P1+BF3 turbulence transport.

Features:
- MPI multicore CPU implementation
- SIMPLE pressure-velocity coupling
- Coupled momentum FGMRES
- Pressure PCG-AMG
- k-omega SST turbulence
- Spalding wall-function sampling
- Omega near-wall treatment
- Editable .case configuration

Purpose:
Experimental reference for developing and testing wall-function
approaches before porting validated changes to the CUDA solver.

Status:
Experimental. Tested on the 207k-tetrahedron spiral using 16 MPI
ranks and on a small pipe using a single-process test environment.
Full CPU/GPU numerical parity and parallel performance optimization
remain incomplete.

This branch must not alter the established CPU laminar or CUDA SST
implementations.
