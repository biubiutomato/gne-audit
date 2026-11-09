#!/usr/bin/env python3
from transformers import AutoModel, AutoTokenizer
import torch

name = "LongSafari/hyenadna-tiny-1k-seqlen-hf"
print("download", name)
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
print("tokenizer ok", type(tok))
model = AutoModel.from_pretrained(name, trust_remote_code=True)
model.eval()
print("model ok", type(model), sum(p.numel() for p in model.parameters()))
s = ["ACGTACGTACGTACGTACGT", "GATTACAGATTACAGATTACA"]
inputs = tok(s, return_tensors="pt", padding=True, truncation=True, max_length=128)
with torch.inference_mode():
    out = model(**inputs)
print(out.last_hidden_state.shape)
