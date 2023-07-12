import torch

from models.causal_lm import CausalLM


class LLaMA13B(CausalLM):

    def __init__(self):
        super().__init__(name="/path/to/llama13b/")
