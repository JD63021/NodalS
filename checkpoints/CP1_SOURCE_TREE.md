# Checkpoint 1 — frozen baseline + source tree

Status target: mechanical-only refactor.

- Frozen original minimal tar and every supplied source/script are under `reference/`.
- `app/nodals_main.cpp` assembles role-specific source fragments under `src/`.
- No numerical statement, PETSc option, BC implementation, solver path or diagnostic was intentionally changed.
- `tests/check_source_freeze.py` must report `exact_text_match=1`.
