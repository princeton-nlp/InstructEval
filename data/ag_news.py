import datasets

from data.dataset import Dataset


class AGNews(Dataset):

    def __init__(self):
        dataset = datasets.load_dataset("ag_news")
        super().__init__("ag_news", "CLS", dataset["train"], dataset["test"])
