# Tests

Run the fast suite locally from the repository root:

```bash
bash scripts/python.sh -m unittest discover -s tests -v
```

The slow suite in `tests/slow/` covers scientific dependencies, PyMC sampling,
and real MCP stdio startup. GitHub Actions runs both suites. The slow directory
intentionally has no `__init__.py`, so local discovery does not enter it.
