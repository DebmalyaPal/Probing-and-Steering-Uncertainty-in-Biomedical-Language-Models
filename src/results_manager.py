"""
Standardised results persistence.

All results are stored under  results/{model_name}/  as JSON files so that
cross-model comparisons can be done from any notebook without re-running
experiments.
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


def save_results(model_name: str, tag: str, data: dict) -> Path:
    """
    Persist *data* as JSON under  results/{model_name}/{tag}.json.

    Parameters
    ----------
    model_name : Registry key (e.g. "biogpt_large").
    tag        : Filename stem (e.g. "experiment", "probe_accuracy").
    data       : Serialisable dict.

    Returns the Path where the file was written.
    """
    out_dir = RESULTS_ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}.json"
    with open(out_path, "w") as f:
        json.dump(_json_safe(data), f, indent=2)
    print(f"Saved → {out_path}")
    return out_path


def load_results(model_name: str, tag: str) -> dict:
    """Load previously saved results."""
    path = RESULTS_ROOT / model_name / f"{tag}.json"
    if not path.exists():
        raise FileNotFoundError(f"No results found at {path}")
    with open(path) as f:
        return json.load(f)


def results_exist(model_name: str, tag: str) -> bool:
    return (RESULTS_ROOT / model_name / f"{tag}.json").exists()


def list_saved(model_name: str) -> list:
    d = RESULTS_ROOT / model_name
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def load_all_probe_results() -> dict:
    """
    Aggregate probe accuracy results across all models that have been run.

    Returns {model_name: probe_results_dict}.
    """
    out = {}
    for model_dir in sorted(RESULTS_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        tag_path = model_dir / "experiment.json"
        if tag_path.exists():
            data = json.loads(tag_path.read_text())
            if "probe_results" in data:
                out[model_dir.name] = data["probe_results"]
    return out
