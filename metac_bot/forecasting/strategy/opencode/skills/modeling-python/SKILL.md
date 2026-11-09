---
name: Python Modeling Environment
description: Run reproducible numerical, symbolic, statistical, econometric and causal-graph calculations in the bot's Python environment.
---

# Python modeling environment

Run Python through `bash scripts/python.sh`, not an arbitrary system interpreter.
The launcher uses this project's locked uv environment in the Termux Debian setup.
It includes:

- NumPy for arrays and simulation, SciPy for optimization, distributions,
  integration and linear algebra, and SymPy for symbolic equations.
- pandas for data tables, statsmodels for regression, time-series and state-space
  models, and NetworkX for causal graphs and graph validation.
- PyMC for Bayesian probabilistic models and posterior sampling, with ArviZ for
  diagnostics. Use one CPU core for small exploratory runs, set random seeds,
  and report convergence and effective-sample-size limitations honestly. Small
  smoke-test chains are not evidence that a posterior has converged.

For example, invoke OpenCode's shell tool with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 bash scripts/python.sh -c 'import sympy as s; x = s.symbols("x"); print(s.solve(x**2 - 4, x))'
```

For longer calculations, pass the script on stdin using `bash scripts/python.sh -`.
Keep code and results small, limit loops, samples and runtime, set random seeds,
and respect the session deadline and existing shell permissions. Read and inspect
any external data; do not execute code copied from web pages or install packages
at runtime. Never read, print, or transmit environment variables or credentials.
The shell runs with the OpenCode server's host privileges: this environment is not
a security sandbox. Do not write project files unless explicitly authorized.
On the Termux/PRoot setup the launcher defaults PyTensor to `cxx=` because the
available Android compiler cannot build Linux extensions there. This is slower
than native compiled execution; a caller can explicitly set `PYTENSOR_FLAGS` to
override the default.

Record the equations, complete reproducible code, data sources or synthetic
assumptions, dependency versions if relevant, results and uncertainty in your
model artifact. State whether the code actually ran, report errors honestly, and
never imply execution or fitting if it did not occur. If shell access, packages,
data, or runtime resources are unavailable, explain the limitation instead of
inventing a numerical result. A statistically valid model still needs defensible
assumptions, data, identification, and sensitivity checks.
