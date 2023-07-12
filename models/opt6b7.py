import torch

from models.causal_lm import CausalLM


class OPT6B7(CausalLM):

    def __init__(self):
        super().__init__("facebook/opt-6.7b")
