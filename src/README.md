# NodalS v1.00 source modules

The production solver remains one C++ translation unit assembled by
`app/nodals_main.cpp` from the ordered `.inc` fragments in this directory.

## Numerical/performance oracle

The authoritative v1.00 production oracle is now
`reference/p1bf3_simple_foam_mpi.cpp` with SHA256:

`0de1a33840d0f36fa84dd7a301bb2073c3dc344cb958defeb0bd7390334d4d01`

This is the accepted FULLFAST-FP64-RICH-SCALE source.  It supersedes the older
`d4388409...` monolith that was accidentally used for the initial modular split.

`tests/check_source_freeze.py` concatenates the ordered source fragments and
requires a byte-for-byte match with the `0de1...` oracle.  This protects both
numerics and the accepted memory architecture, including:

- root-only global OpenFOAM mesh read plus distributed local support packets;
- compact fixed Dirichlet-value storage;
- M1 release of setup-only `colGid`;
- M2 elimination of persistent `kNu`;
- PLAN-C1 `CustomDynamicRuntimeCellPlan`;
- release of setup-only full-Schur `vertexCells`;
- custom FP64 factored physical pressure operator with full explicit Pmat only
  as the GAMG preconditioning snapshot;
- scalar momentum solves with processor block-Jacobi plus local SGS.

Do not independently reorder the fragments.  Any production edit must retain
`NODALS_ACTIVE_ORACLE ... exact_text_match=1` until a new oracle is explicitly
accepted.
