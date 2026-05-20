import torch
from transformers import BioGptTokenizer, BioGptForCausalLM

def load_model(device="mps"):
    tok = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    model = BioGptForCausalLM.from_pretrained(
        "microsoft/biogpt",
        output_hidden_states=True,
    ).to(device)
    model.eval()
    return tok, model

def get_last_token_hidden_state(text, tok, model, layer=-1, device="mps"):
    """Hidden state of the final non-padding token at the given layer."""
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer]
    return hidden[0, -1, :].cpu()

def get_mean_hidden_state(text, tok, model, layer=-1, device="mps"):
    """Mean hidden state across all non-padding tokens at the given layer."""
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer]  # (1, seq_len, hidden_dim)
    mask = inputs.attention_mask.unsqueeze(-1).float()  # (1, seq_len, 1)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return pooled[0].cpu()

if __name__ == "__main__":
    tok, model = load_model()
    h = get_mean_hidden_state("The patient may have pneumonia.", tok, model)
    print(f"Mean-pooled hidden state shape: {h.shape}")