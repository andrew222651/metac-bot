Everything should be made as simple as possible, but not simpler.

Run only the fast tests locally: `bash scripts/python.sh -m unittest discover -s tests -v`. The slow tests in `tests/slow/` run separately in CI; do not run them locally unless the user explicitly asks.

When `<ROOT>/SPEC.md` or the code is modified, also modify the other one so that
the code is always in line with the spec.
