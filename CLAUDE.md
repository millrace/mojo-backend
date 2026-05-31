# Working conventions for this repo

## Scratch / temporary files

Use the repo-local **`.scratch/`** directory for all ad-hoc output (test logs,
throwaway scripts, captured stdout) — **never `/tmp`**. `.scratch/` is
gitignored and pre-approved, so writing there needs no permission prompt.

- Redirect captured output to `.scratch/<name>.out`, not `/tmp/...`.
- Put any throwaway script in `.scratch/`; promote it to `tests/` only if it
  becomes a real, reusable test.

## Tool usage

Avoid low level shell tools when:

1. they may provide a security risk
2. other tools are available.

## Manual / GPU tests

Inference tests that need weights + the Metal GPU live in `tests/manual/` and
run via pixi tasks (e.g. `pixi run gpu-check`), not the pure-Python unittest
suite (`pixi run test`).
