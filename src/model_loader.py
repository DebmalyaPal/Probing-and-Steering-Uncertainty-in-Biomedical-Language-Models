"""
Generic model and tokenizer loader that works for all models in the registry.

Design decisions:
  - Decoders  → AutoModelForCausalLM  (supports generate() and hidden states)
  - Encoders  → AutoModelForMaskedLM  (supports MLM loss for pseudo-perplexity
                                       and hidden states)
  - 7B models → loaded in 8-bit via bitsandbytes when config.quantize=True
  - Device    → auto-detected (CUDA > MPS > CPU)
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM,
    BertTokenizer, BertForMaskedLM,
)

from src.model_registry import get_config


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name: str, device: str = None, force_cpu: bool = False):
    """
    Load tokenizer and model for *model_name*.

    Parameters
    ----------
    model_name  : Key in MODEL_REGISTRY (e.g. "biogpt", "biobert").
    device      : Override auto-detected device string.
    force_cpu   : Useful for unit tests; ignores quantization flags.

    Returns
    -------
    tok   : Tokenizer (with pad token set when missing).
    model : Model in eval mode, placed on *device*.
    """
    cfg = get_config(model_name)
    hf_id = cfg["hf_id"]
    quantize = cfg["quantize"] and not force_cpu

    if device is None:
        device = "cpu" if force_cpu else get_device()

    # ---- tokenizer --------------------------------------------------------
    # Load order:
    #   1. Registry-specified tokenizer class (for old checkpoints like BlueBERT)
    #   2. AutoTokenizer fast
    #   3. AutoTokenizer slow  (no tokenizer.json)
    #   4. BertTokenizer slow  (final fallback for BERT variants)
    tok_class_name = cfg.get("tokenizer_class")
    tok = None
    if tok_class_name == "BertTokenizer":
        tok = BertTokenizer.from_pretrained(hf_id)
    else:
        for kwargs in [{"use_fast": True}, {"use_fast": False}]:
            try:
                tok = AutoTokenizer.from_pretrained(
                    hf_id, trust_remote_code=True, **kwargs
                )
                break
            except (ValueError, OSError):
                continue
        if tok is None:
            tok = BertTokenizer.from_pretrained(hf_id)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- model ------------------------------------------------------------
    load_kwargs = dict(output_hidden_states=True, trust_remote_code=True)

    if quantize:
        # Requires bitsandbytes: pip install bitsandbytes
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"] = "auto"
    else:
        # float16 on MPS is incomplete — many ops (e.g. matmul variants used
        # in the final BERT layer) raise dtype assertion errors at runtime.
        # Use float16 only on CUDA where it is fully supported.
        # Always float32 for non-quantized models.
        # float16 causes ~10pp probe accuracy loss on deep models (accumulated
        # rounding errors wash out the uncertainty signal in later layers).
        # float16 on MPS raises dtype assertion errors at runtime.
        # Memory cost is acceptable: largest non-quantized model (BioMedLM,
        # 2.7B) uses ~11 GB float32, within a 16 GB GPU budget.
        load_kwargs["dtype"] = torch.float32

    # Use registry-specified class for old checkpoints that lack model_type
    model_class_name = cfg.get("model_class")
    if model_class_name == "BertForMaskedLM":
        model = BertForMaskedLM.from_pretrained(hf_id, **load_kwargs)
    elif cfg["model_type"] == "decoder":
        model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
    else:
        model = AutoModelForMaskedLM.from_pretrained(hf_id, **load_kwargs)

    if not quantize:
        model = model.to(device)

    model.eval()
    return tok, model
