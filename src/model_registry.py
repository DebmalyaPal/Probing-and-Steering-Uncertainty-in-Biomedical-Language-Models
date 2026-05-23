"""
Central registry of all supported biomedical language models.

Each entry defines:
  hf_id        : HuggingFace model identifier
  display_name : Human-readable name for figures/logs
  model_type   : "decoder" (causal LM) or "encoder" (masked LM)
  hidden_dim   : Size of hidden state vectors
  num_layers   : Number of transformer layers (excluding embedding layer)
  layers_path  : Dot-separated attribute path from the model object to the
                 list of transformer layer modules, used for hook injection.
                 Resolved via get_layers(model).
  probe_layers : Layer indices to sweep during probing (0 = embedding output).
  steer_layer  : Default layer for steering. None means "pick by probing first".
  quantize     : Whether to load in 8-bit (for large models on limited VRAM).
"""

MODEL_REGISTRY = {
    # ------------------------------------------------------------------ #
    # Decoder (causal LM) models                                          #
    # ------------------------------------------------------------------ #
    "biogpt": {
        "hf_id": "microsoft/biogpt",
        "display_name": "BioGPT",
        "model_type": "decoder",
        "hidden_dim": 1024,
        "num_layers": 24,
        "layers_path": "biogpt.layers",
        # Every layer: 0 = embedding output, 1–24 = transformer block outputs.
        # Layer index here corresponds directly to hidden_states[i].
        "probe_layers": list(range(25)),   # 0–24, all 25 hidden states
        "steer_layer": 16,                 # from original paper
        "quantize": False,
    },
    "biogpt_large": {
        "hf_id": "microsoft/BioGPT-Large",
        "display_name": "BioGPT-Large",
        "model_type": "decoder",
        "hidden_dim": 1600,
        "num_layers": 48,
        "layers_path": "biogpt.layers",
        "probe_layers": list(range(49)),   # 0–48, all 49 hidden states
        "steer_layer": None,               # determined by probing
        "quantize": False,
    },
    "biomedlm": {
        "hf_id": "stanford-crfm/BioMedLM",
        "display_name": "BioMedLM (PubMedGPT)",
        "model_type": "decoder",
        "hidden_dim": 2560,
        "num_layers": 32,
        # GPT-2 architecture; transformer layers live at model.transformer.h
        "layers_path": "transformer.h",
        "probe_layers": list(range(33)),   # 0–32
        "steer_layer": None,
        "quantize": False,
    },
    "biomistral": {
        "hf_id": "BioMistral/BioMistral-7B",
        "display_name": "BioMistral-7B",
        "model_type": "decoder",
        "hidden_dim": 4096,
        "num_layers": 32,
        # Mistral/LLaMA architecture; layers live at model.model.layers
        "layers_path": "model.layers",
        "probe_layers": list(range(33)),   # 0–32
        "steer_layer": None,
        "quantize": True,                  # 7 B → load in 8-bit
    },
    "meditron": {
        "hf_id": "epfl-llm/meditron-7b",
        "display_name": "Meditron-7B",
        "model_type": "decoder",
        "hidden_dim": 4096,
        "num_layers": 32,
        # LLaMA-2 architecture
        "layers_path": "model.layers",
        "probe_layers": list(range(33)),   # 0–32
        "steer_layer": None,
        "quantize": True,
    },
    # ------------------------------------------------------------------ #
    # Encoder (masked LM) models                                          #
    # Steering produces representation shifts measured via probe           #
    # projection; text generation is not available.                       #
    # ------------------------------------------------------------------ #
    "biobert": {
        "hf_id": "dmis-lab/biobert-v1.1",
        "display_name": "BioBERT",
        "model_type": "encoder",
        "hidden_dim": 768,
        "num_layers": 12,
        "layers_path": "bert.encoder.layer",
        "probe_layers": list(range(13)),   # 0–12 (embedding + 12 blocks)
        "steer_layer": None,
        "quantize": False,
    },
    "clinicalbert": {
        "hf_id": "emilyalsentzer/Bio_ClinicalBERT",
        "display_name": "ClinicalBERT",
        "model_type": "encoder",
        "hidden_dim": 768,
        "num_layers": 12,
        "layers_path": "bert.encoder.layer",
        "probe_layers": list(range(13)),
        "steer_layer": None,
        "quantize": False,
    },
    "bluebert": {
        "hf_id": "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
        "display_name": "BlueBERT",
        "model_type": "encoder",
        "hidden_dim": 768,
        "num_layers": 12,
        "layers_path": "bert.encoder.layer",
        "probe_layers": list(range(13)),
        "steer_layer": None,
        "quantize": False,
        # Very old checkpoint: config.json has no model_type field, so
        # AutoModelForMaskedLM cannot resolve it. Load with explicit class.
        "model_class": "BertForMaskedLM",
        "tokenizer_class": "BertTokenizer",
    },
    "scibert": {
        "hf_id": "allenai/scibert_scivocab_uncased",
        "display_name": "SciBERT",
        "model_type": "encoder",
        "hidden_dim": 768,
        "num_layers": 12,
        "layers_path": "bert.encoder.layer",
        "probe_layers": list(range(13)),
        "steer_layer": None,
        "quantize": False,
    },
}


def get_config(model_name: str) -> dict:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]


def get_layers(model, model_name: str):
    """
    Return the list of transformer layer modules for a loaded model.
    Resolves the dot-separated layers_path from the registry.
    """
    path = get_config(model_name)["layers_path"]
    obj = model
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


def list_models(model_type: str = None):
    """List model names, optionally filtered by type ('decoder'/'encoder')."""
    return [
        k for k, v in MODEL_REGISTRY.items()
        if model_type is None or v["model_type"] == model_type
    ]
