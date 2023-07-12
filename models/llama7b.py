import torch

from models.causal_lm import CausalLM


class LLaMA7B(CausalLM):

    def __init__(self):
        super().__init__(name="/path/to/llama7b/")
