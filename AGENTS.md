# AGENTS.md

## Cursor Cloud specific instructions

This is a Python-based numerical mathematics research project for Gaussian quadrature rule optimization. It consists of standalone Python scripts and Jupyter notebooks — no web app, no external services.

### Key files

- `test_opt_ab.py` — Gauss-Newton optimizer for 1D Gaussian quadrature (main runnable script)
- `iteratively.ipynb` — Projected BFGS and alternating least squares for constrained quadrature
- `Spectrum.ipynb` / `Spectrum1.ipynb` — 2D quadrature via spectral/eigenvalue methods
- `LP method/LP_quad.ipynb` — LP-based quadrature with KMeans clustering
- `LP method/sqp.py` — SQP stub (imports only)

### Running

- **Script:** `python3 test_opt_ab.py` — runs the Gauss-Newton optimizer and shows convergence results
- **Notebooks:** `jupyter lab --no-browser --ip=0.0.0.0 --port=8888` to start JupyterLab

### Important caveats

- **scipy version:** The codebase uses `scipy.sparse.linalg.cg(..., tol=...)` which was deprecated in scipy 1.12 and removed in 1.14. Pin `scipy<1.14` to avoid `TypeError`.
- **Computationally heavy notebooks:** Several notebook cells (e.g., `iteratively.ipynb` cell 12 with n=99 BFGS, `Spectrum.ipynb` cell 10) are very slow and may time out. This is expected for research code.
- **No formal dependency manifest:** There is no `requirements.txt` or `pyproject.toml`. Dependencies are: `numpy`, `scipy<1.14`, `matplotlib`, `pandas`, `sympy`, `colorcet`, `scikit-learn`, `jupyterlab`.
- **No test suite:** Despite `test_opt_ab.py` having "test" in its name, it is a standalone computational script, not a test suite. There are no automated tests.
- **No linter configured:** No linting tools or configurations exist in the repository.
