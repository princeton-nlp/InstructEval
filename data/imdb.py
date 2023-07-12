import datasets
from data.dataset import Dataset

class IMDB(Dataset):
    def __init__(self):
        dataset = datasets.load_dataset("imdb", ignore_verifications=True)
        super().__init__("imdb", "CLS", dataset["train"], dataset["test"])
