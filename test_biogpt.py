import torch
from transformers import BioGptTokenizer, BioGptForCausalLM

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading BioGPT...")
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt").to(device)
model.eval()

prompt = "The patient presents with chest pain and shortness of breath."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
    )

print("\n--- Generated output ---")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))