"""
Notebook 00: Environment Test
Run this first to confirm all dependencies are installed correctly.
Expected output: a green "Environment OK" summary.
"""

import sys

checks = {}

try:
    import scanpy as sc
    checks["scanpy"] = sc.__version__
except ImportError as e:
    checks["scanpy"] = f"MISSING — {e}"

try:
    import anndata as ad
    checks["anndata"] = ad.__version__
except ImportError as e:
    checks["anndata"] = f"MISSING — {e}"

try:
    import decoupler as dc
    checks["decoupler"] = dc.__version__
except ImportError as e:
    checks["decoupler"] = f"MISSING — {e}"

try:
    import pandas as pd
    checks["pandas"] = pd.__version__
except ImportError as e:
    checks["pandas"] = f"MISSING — {e}"

try:
    import numpy as np
    checks["numpy"] = np.__version__
except ImportError as e:
    checks["numpy"] = f"MISSING — {e}"

try:
    import scipy
    checks["scipy"] = scipy.__version__
except ImportError as e:
    checks["scipy"] = f"MISSING — {e}"

try:
    import matplotlib
    checks["matplotlib"] = matplotlib.__version__
except ImportError as e:
    checks["matplotlib"] = f"MISSING — {e}"

try:
    import plotly
    checks["plotly"] = plotly.__version__
except ImportError as e:
    checks["plotly"] = f"MISSING — {e}"

try:
    import GEOparse
    checks["GEOparse"] = GEOparse.__version__
except ImportError as e:
    checks["GEOparse"] = f"MISSING — {e}"

# Print summary
print(f"\nPython: {sys.version}\n")
all_ok = True
for pkg, version in checks.items():
    status = "OK" if "MISSING" not in str(version) else "FAIL"
    if status == "FAIL":
        all_ok = False
    print(f"  [{status}] {pkg:<20} {version}")

print()
if all_ok:
    print("Environment OK — ready to proceed to notebook 01.")
else:
    print("Some packages are missing. Run: conda env update -f environment.yml")
