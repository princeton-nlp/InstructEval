from models.causal_lm import CausalLM


class StableLMTuned7B(CausalLM):

    def __init__(self):
        super().__init__("StabilityAI/stablelm-tuned-alpha-7b")
