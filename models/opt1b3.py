import torch

from models.causal_lm import CausalLM


class OPT1B3(CausalLM):

    def __init__(self):
        super().__init__("facebook/opt-1.3b")
