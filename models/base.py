import torch


class BaseModel:

    def __init__(self, name: str, model, tokenizer, device: str):
        self.name = name
        self.hf_model = model
        self.tokenizer = tokenizer
        self.device = device

    def forward(self, *args, **kwargs):
        raise NotImplementedError
