import torch

from models.causal_lm import CausalLM


class Bloom1B1(CausalLM):

    def __init__(self):
        super().__init__("bigscience/bloom-1b1")
