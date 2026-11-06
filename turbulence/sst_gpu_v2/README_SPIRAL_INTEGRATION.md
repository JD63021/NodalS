# NodalS V2 GPU SST source import

This subtree is copied from the active local
`~/Downloads/NodalS_SST_SOFTWARE_V2` installation by the
`COMMIT_TO_LOCAL_NODALS.sh` integration helper.

The existing NodalS laminar solver under `/src`, `/app`, etc. remains
untouched. This SST code is a separate pure-tetrahedral CUDA solver and
uses its own `nodals.py` front-end and CUDA build code.

The imported tree intentionally excludes generated files: caches,
CUDA binaries, run folders, checkpoint/VTU data, local mesh datasets,
backups, and logs. It may still need build and dependency documentation
before a public release. The current live installation is the source
of truth for this commit; no numerical source edits are applied here.

See `examples/spiral_swept_207k/` for the example mesh and `.case`.
