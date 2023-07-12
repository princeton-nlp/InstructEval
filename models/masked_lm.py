import torch
import transformers

from models.base import BaseModel


class MaskedLM(BaseModel):

    def __init__(self, name: str):
        if torch.cuda.is_available():
            model = transformers.AutoModelForMaskedLM.from_pretrained(
                name, device_map="auto")
        else:
            model = transformers.AutoModelForMaskedLM.from_pretrained(name)

        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        tokenizer.pad_token = self.tokenizer.eos_token
        super().__init__(name, model, tokenizer)
