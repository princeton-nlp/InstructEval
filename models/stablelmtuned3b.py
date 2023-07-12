from models.causal_lm import CausalLM


class StableLMTuned3B(CausalLM):

    def __init__(self):
        super().__init__("StabilityAI/stablelm-tuned-alpha-3b")
