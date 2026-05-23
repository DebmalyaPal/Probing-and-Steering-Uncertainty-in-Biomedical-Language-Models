"""
Lightweight correctness tests that run without GPU or real model weights.

Tests cover:
  - model_registry: structure and layer resolution
  - bioscope_parser: parsing produces correct output
  - probing: probe math on random features
  - orthogonalization: orthogonality condition holds
  - steering_core: seed functions work, steer_representation shape
  - results_manager: save / load round-trip
  - metrics: text-only metrics (no model needed)
"""

import sys, json, tempfile, os
from pathlib import Path
import numpy as np

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ============================================================================ #
# model_registry                                                               #
# ============================================================================ #

def test_registry_keys():
    from src.model_registry import MODEL_REGISTRY, get_config, list_models
    assert "biogpt"       in MODEL_REGISTRY
    assert "biogpt_large" in MODEL_REGISTRY
    assert "biomedlm"     in MODEL_REGISTRY
    assert "biomistral"   in MODEL_REGISTRY
    assert "meditron"     in MODEL_REGISTRY
    assert "biobert"      in MODEL_REGISTRY
    assert "clinicalbert" in MODEL_REGISTRY
    assert "bluebert"     in MODEL_REGISTRY
    assert "scibert"      in MODEL_REGISTRY
    print("PASS: registry has all 9 models")


def test_registry_fields():
    from src.model_registry import MODEL_REGISTRY
    required = {"hf_id", "display_name", "model_type", "hidden_dim",
                "num_layers", "layers_path", "probe_layers", "steer_layer",
                "quantize"}
    for name, cfg in MODEL_REGISTRY.items():
        missing = required - cfg.keys()
        assert not missing, f"{name} missing fields: {missing}"
    print("PASS: all registry entries have required fields")


def test_list_models():
    from src.model_registry import list_models
    decoders = list_models("decoder")
    encoders = list_models("encoder")
    assert len(decoders) == 5
    assert len(encoders) == 4
    print(f"PASS: {len(decoders)} decoders, {len(encoders)} encoders")


# ============================================================================ #
# bioscope_parser                                                               #
# ============================================================================ #

def test_bioscope_parser_exists():
    from src.bioscope_parser import build_balanced_contrast_set
    bioscope_path = ROOT / "data" / "bioscope"
    if not bioscope_path.exists():
        print("SKIP: data/bioscope not found")
        return
    U, C = build_balanced_contrast_set(str(bioscope_path), max_per_class=10, seed=42)
    assert len(U) > 0 and len(C) > 0
    assert len(U) == len(C)
    print(f"PASS: bioscope parsed {len(U)} uncertain + {len(C)} certain")


def test_bioscope_cached():
    cached = ROOT / "data" / "processed" / "bioscope_contrast.json"
    if not cached.exists():
        print("SKIP: cached contrast set not found")
        return
    with open(cached) as f:
        data = json.load(f)
    assert "uncertain" in data and "certain" in data
    assert data["seed"] == 42
    print(f"PASS: cached contrast set has {len(data['uncertain'])} uncertain sentences")


# ============================================================================ #
# probing (mock data — no model needed)                                        #
# ============================================================================ #

def test_probe_at_layer():
    from src.probing import probe_at_layer, build_probe_vector
    rng = np.random.RandomState(42)
    # Linearly separable mock data
    X_u = rng.randn(50, 32) + 1.0   # uncertain class shifted +1
    X_c = rng.randn(50, 32) - 1.0   # certain class shifted -1

    stats = probe_at_layer(X_u, X_c, n_splits=3, seed=42)
    assert stats["mean_acc"] > 0.8, f"Expected >80%, got {stats['mean_acc']:.2%}"
    assert "ci_low" in stats and "ci_high" in stats
    print(f"PASS: probe_at_layer acc={stats['mean_acc']:.2%}")

    vec, acc = build_probe_vector(X_u, X_c, seed=42)
    assert vec.shape == (32,)
    assert acc > 0.8
    print(f"PASS: build_probe_vector acc={acc:.2%}")


# ============================================================================ #
# orthogonalization                                                             #
# ============================================================================ #

def test_orthogonalize():
    from src.orthogonalization import orthogonalize, cosine_similarity
    rng = np.random.RandomState(0)
    a = rng.randn(128)
    b = rng.randn(128)
    c = orthogonalize(a, b)
    # c should be orthogonal to b
    dot = float(np.dot(c, b / np.linalg.norm(b)))
    assert abs(dot) < 1e-9, f"Not orthogonal: dot={dot:.2e}"
    print(f"PASS: orthogonalize dot(result, confound)={dot:.2e}")


def test_cosine_similarity():
    from src.orthogonalization import cosine_similarity
    v = np.array([1.0, 0.0, 0.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9
    assert abs(cosine_similarity(v, -v) + 1.0) < 1e-9
    print("PASS: cosine_similarity")


# ============================================================================ #
# steering_core — seed function                                                #
# ============================================================================ #

def test_probe_layers_all_layers():
    """Every model should now probe every layer, not a coarse sample."""
    from src.model_registry import MODEL_REGISTRY
    for name, cfg in MODEL_REGISTRY.items():
        n = cfg["num_layers"]
        expected = list(range(n + 1))   # 0 through num_layers inclusive
        assert cfg["probe_layers"] == expected, (
            f"{name}: expected probe_layers=range({n+1}), "
            f"got {cfg['probe_layers']}"
        )
    print("PASS: all models probe every layer (0 to num_layers)")


def test_hook_index_conversion():
    from src.steering_core import _probe_layer_to_hook_index
    # hidden_states[1] → layers_list[0]
    assert _probe_layer_to_hook_index(1) == 0
    # hidden_states[16] → layers_list[15]
    assert _probe_layer_to_hook_index(16) == 15
    # hidden_states[24] → layers_list[23]
    assert _probe_layer_to_hook_index(24) == 23
    # layer 0 should raise
    try:
        _probe_layer_to_hook_index(0)
        assert False, "Expected ValueError"
    except ValueError:
        pass
    print("PASS: _probe_layer_to_hook_index correct")


def test_set_all_seeds():
    from src.steering_core import set_all_seeds
    import torch, random
    set_all_seeds(42)
    r1 = random.random()
    n1 = np.random.randn()
    t1 = torch.randn(1).item()
    set_all_seeds(42)
    r2 = random.random()
    n2 = np.random.randn()
    t2 = torch.randn(1).item()
    assert r1 == r2 and n1 == n2 and t1 == t2
    print("PASS: set_all_seeds is deterministic")


# ============================================================================ #
# metrics — text-only (no model)                                               #
# ============================================================================ #

def test_metrics_text_only():
    from src.metrics import hedge_score, lexical_diversity, token_validity, token_length
    text = "The patient may have pneumonia, which might indicate a serious condition."
    hs = hedge_score(text)
    ld = lexical_diversity(text)
    tv = token_validity(text)
    tl = token_length(text)
    assert hs >= 2.0, f"hedge_score={hs}"
    assert 0 < ld <= 1.0
    assert 0 < tv <= 1.0
    assert tl > 0
    print(f"PASS: metrics text-only  hedge={hs}  ld={ld:.2f}  tv={tv:.2f}  len={tl}")


def test_metrics_empty():
    from src.metrics import hedge_score, lexical_diversity, token_validity, token_length
    assert hedge_score("") == 0.0
    assert lexical_diversity("") == 0.0
    assert token_validity("") == 0.0
    assert token_length("") == 0
    print("PASS: metrics handle empty string")


# ============================================================================ #
# results_manager                                                               #
# ============================================================================ #

def test_results_roundtrip():
    from src.results_manager import save_results, load_results, results_exist
    import tempfile, os

    # Temporarily redirect RESULTS_ROOT
    import src.results_manager as rm
    orig_root = rm.RESULTS_ROOT
    with tempfile.TemporaryDirectory() as tmpdir:
        rm.RESULTS_ROOT = Path(tmpdir)
        data = {"foo": 1, "bar": [1, 2, 3], "nested": {"x": 0.5}}
        save_results("test_model", "test_tag", data)
        assert results_exist("test_model", "test_tag")
        loaded = load_results("test_model", "test_tag")
        assert loaded == data
    rm.RESULTS_ROOT = orig_root
    print("PASS: results_manager save/load round-trip")


# ============================================================================ #
# Runner                                                                       #
# ============================================================================ #

if __name__ == "__main__":
    tests = [
        test_registry_keys,
        test_registry_fields,
        test_list_models,
        test_probe_layers_all_layers,
        test_hook_index_conversion,
        test_bioscope_parser_exists,
        test_bioscope_cached,
        test_probe_at_layer,
        test_orthogonalize,
        test_cosine_similarity,
        test_set_all_seeds,
        test_metrics_text_only,
        test_metrics_empty,
        test_results_roundtrip,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
