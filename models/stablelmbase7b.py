from models.causal_lm import CausalLM


class StableLMBase7B(CausalLM):

    def __init__(self):
        super().__init__("StabilityAI/stablelm-base-alpha-7b")
