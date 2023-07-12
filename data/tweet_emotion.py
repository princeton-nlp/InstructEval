import datasets

from data.dataset import Dataset


class TweetEmotion(Dataset):

    def __init__(self):
        dataset = datasets.load_dataset("tweet_eval", "emotion")
        super().__init__("tweet_emotion",
                         "CLS",
                         dataset["train"],
                         dataset["test"],
                         text_keys=["text"],
                         label_key="label")
