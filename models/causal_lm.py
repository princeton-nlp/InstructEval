import collections
from typing import Optional, Tuple

import torch
import transformers
from tqdm import tqdm

from models.base import BaseModel


class CausalLM(BaseModel):

    def __init__(self, name: str):
        if torch.cuda.is_available():
            model = transformers.AutoModelForCausalLM.from_pretrained(
                name, device_map="auto", torch_dtype=torch.float16)
            device = f"cuda:{list(model.hf_device_map.values())[0]}"
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(name)
            device = "cpu"

        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token

        super().__init__(name, model, tokenizer, device)

    def forward(self, *args, **kwargs):
        return self.hf_model.forward(*args, **kwargs)
