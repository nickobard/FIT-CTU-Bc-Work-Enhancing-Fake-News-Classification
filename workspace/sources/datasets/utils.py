import sys
import importlib
import site
from pathlib import Path


def import_hf_datasets():
    module_name = "datasets"
    # 1) Find site-packages dirs
    site_dirs = site.getsitepackages() + [site.getusersitepackages()]
    hf_path = None
    for d in site_dirs:
        candidate = Path(d) / module_name
        if candidate.is_dir():
            hf_path = str(d)
            break
    if hf_path is None:
        raise ImportError(f"Could not find installed '{module_name}' in site-packages")

    # 2) Temporarily prioritize that site-packages
    sys.path.insert(0, hf_path)
    hf = importlib.import_module(module_name)
    sys.path.pop(0)

    return hf
