import torch

from models.causal_lm import CausalLM


class GPTNeo1B3(CausalLM):

    def __init__(self):
        super().__init__("EleutherAI/gpt-neo-1.3B")
