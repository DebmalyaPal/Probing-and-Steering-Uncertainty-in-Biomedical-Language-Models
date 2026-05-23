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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM

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
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- model ------------------------------------------------------------
    load_kwargs = dict(
        output_hidden_states=True,
        trust_remote_code=True,
    )

    if quantize:
        # Requires bitsandbytes: pip install bitsandbytes
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"] = "auto"
    else:
        # float16 on MPS is incomplete — many ops (e.g. matmul variants used
        # in the final BERT layer) raise dtype assertion errors at runtime.
        # Use float16 only on CUDA where it is fully supported.
        load_kwargs["torch_dtype"] = (
            torch.float16 if device == "cuda" else torch.float32
        )

    if cfg["model_type"] == "decoder":
        model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
    else:
        model = AutoModelForMaskedLM.from_pretrained(hf_id, **load_kwargs)

    if not quantize:
        model = model.to(device)

    model.eval()
    return tok, model
