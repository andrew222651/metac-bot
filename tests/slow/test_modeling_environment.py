"""Numerical smoke tests for the Python environment used by modeling agents."""

import math
import os
from pathlib import Path
import subprocess
import unittest

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import beta
import statsmodels.api as sm
import sympy as sp


class ModelingEnvironmentTests(unittest.TestCase):
    def test_symbolic_equilibrium_and_linear_regression(self):
        x = sp.symbols("x")
        self.assertEqual(sp.solve(2 * x - 6, x), [3])
        data = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 3.0, 5.0, 7.0]})
        fitted = sm.OLS(data["y"], sm.add_constant(data["x"])).fit()
        self.assertTrue(np.allclose(fitted.params.to_numpy(), [1.0, 2.0]))

    def test_bayesian_update_and_causal_dag(self):
        # A Beta(2, 2) prior updated with 3 successes and 1 failure.
        self.assertTrue(math.isclose(beta.mean(5, 3), 0.625))
        graph = nx.DiGraph(
            [
                ("confounder", "treatment"),
                ("confounder", "outcome"),
                ("treatment", "outcome"),
            ]
        )
        self.assertTrue(nx.is_directed_acyclic_graph(graph))


class ModelingLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        code = """
import numpy, scipy, sympy, pandas, statsmodels, networkx
import pymc as pm
import numpy as np
print('modeling-ready', flush=True)
with pm.Model():
    p = pm.Beta('p', alpha=2, beta=2)
    pm.Binomial('observed', n=4, p=p, observed=3)
    result = pm.sample(draws=30, tune=30, chains=1, cores=1, random_seed=42,
                       progressbar=False, compute_convergence_checks=False)
values = result.posterior['p'].values
assert values.size == 30 and np.all(np.isfinite(values))
assert np.all((values > 0) & (values < 1))
print('pymc-ready')
"""
        environment = {
            **os.environ,
            "UV_NO_SYNC": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        # Graph optimization is disproportionately slow in PRoot without C.
        # Keep the same NUTS model and draws; native CI uses the normal optimizer.
        # Set METAC_BOT_TEST_FULL_PYTENSOR=1 to exercise the full Termux default.
        if (
            os.environ.get("PD_CONTAINER")
            and os.environ.get("ANDROID_ROOT")
            and environment.get("PYTENSOR_FLAGS") == "cxx="
            and os.environ.get("METAC_BOT_TEST_FULL_PYTENSOR") != "1"
        ):
            environment["PYTENSOR_FLAGS"] = "cxx=,optimizer=fast_compile"
        # Avoid nested uv synchronization removing optional test dependencies.
        cls.result = subprocess.run(
            ["bash", "scripts/python.sh", "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
        )

    def test_launcher_uses_the_scientific_environment(self):
        self.assertIn("modeling-ready", self.result.stdout, self.result.stderr)

    def test_pymc_samples_a_small_model_through_launcher(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr)
        self.assertIn("pymc-ready", self.result.stdout)
