"""
Standardised results persistence.

All results are stored under  results/{model_name}/{device}/{dtype}/  as JSON
files so that cross-model, cross-device comparisons can be done from any
notebook without re-running experiments.

  {model_name} : Registry key (e.g. "biogpt_large")
  {device}     : Compute backend used ("cuda", "mps", "cpu")
  {dtype}      : Numeric precision   ("float32", "float16", "bfloat16")
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime


RESULTS_ROOT = Path("results")


def _json_safe(obj):
    """Recursively convert numpy scalars/arrays to Python builtins."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_results(model_name: str, tag: str, data: dict,
                 device: str = "cpu",
                 dtype: str = "float32") -> Path:
    """
    Persist *data* as JSON under
    results/{model_name}/{device}/{dtype}/{tag}.json.

    Parameters
    ----------
    model_name : Registry key (e.g. "biogpt_large").
    tag        : Filename stem (e.g. "experiment", "probe_accuracy").
    data       : Serialisable dict.
    device     : Compute backend used for the run ("cuda", "mps", "cpu").
    dtype      : Numeric precision used for the run (e.g. "float32", "float16").

    Returns the Path where the file was written.
    """
    out_dir = RESULTS_ROOT / model_name / device / dtype
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}.json"
    with open(out_path, "w") as f:
        json.dump(_json_safe(data), f, indent=2)
    print(f"Saved → {out_path}")
    return out_path


def load_results(model_name: str, tag: str,
                 device: str = "cpu",
                 dtype: str = "float32") -> dict:
    """Load previously saved results."""
    path = RESULTS_ROOT / model_name / device / dtype / f"{tag}.json"
    if not path.exists():
        raise FileNotFoundError(f"No results found at {path}")
    with open(path) as f:
        return json.load(f)


def results_exist(model_name: str, tag: str,
                  device: str = "cpu",
                  dtype: str = "float32") -> bool:
    return (RESULTS_ROOT / model_name / device / dtype / f"{tag}.json").exists()


def list_saved(model_name: str,
               device: str = "cpu",
               dtype: str = "float32") -> list:
    d = RESULTS_ROOT / model_name / device / dtype
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def load_all_probe_results(device: str = "cpu",
                           dtype: str = "float32") -> dict:
    """
    Aggregate probe accuracy results across all models that have been run.

    Returns {model_name: probe_results_dict}.
    """
    out = {}
    for model_dir in sorted(RESULTS_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        tag_path = model_dir / device / dtype / "experiment.json"
        if tag_path.exists():
            data = json.loads(tag_path.read_text())
            if "probe_results" in data:
                out[model_dir.name] = data["probe_results"]
    return out
