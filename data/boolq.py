import datasets

from data.dataset import Dataset


class BoolQ(Dataset):

    def __init__(self):
        dataset = datasets.load_dataset("boolq")
        super().__init__("boolq",
                         "CLS",
                         dataset["train"],
                         dataset["validation"],
                         text_keys=["question", "passage"],
                         label_key="answer")
