from models.causal_lm import CausalLM


class GPTNeo2B7(CausalLM):

    def __init__(self):
        super().__init__("EleutherAI/gpt-neo-2.7B")
