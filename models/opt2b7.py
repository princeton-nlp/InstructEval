import torch

from models.causal_lm import CausalLM


class OPT2B7(CausalLM):

    def __init__(self):
        super().__init__("facebook/opt-2.7b")
