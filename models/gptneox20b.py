import torch

from models.causal_lm import CausalLM


class GPTNeoX20B(CausalLM):

    def __init__(self):
        super().__init__("EleutherAI/gpt-neox-20b")
