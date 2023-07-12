import datasets

from data.dataset import Dataset


class NQOpen(Dataset):

    def __init__(self):
        dataset = datasets.load_dataset("nq_open")
        super().__init__("nq_open",
                         "GQA",
                         dataset["train"],
                         dataset["validation"],
                         text_keys=["question"],
                         label_key="answer")
