import datasets

from data.dataset import Dataset


class TriviaQA(Dataset):

    def __init__(self):
        dataset = datasets.load_dataset("trivia_qa", "rc.web.nocontext")
        super().__init__("trivia_qa",
                         "GQA",
                         dataset["train"],
                         dataset["validation"],
                         text_keys=["question"],
                         label_key="answer")
