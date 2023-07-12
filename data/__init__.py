from data.ag_news import AGNews
from data.anli import ANLI
from data.boolq import BoolQ
from data.cosmos_qa import CosmosQA
from data.dataset import Dataset
from data.hellaswag import HellaSwag
from data.nq_open import NQOpen
from data.imdb import IMDB
from data.trivia_qa import TriviaQA
from data.tweet_emotion import TweetEmotion


def get_dataset(name: str) -> Dataset:
    name2dataset = {
        "ag_news": AGNews,
        "imdb": IMDB,
        "anli": ANLI,
        "boolq": BoolQ,
        "tweet_emotion": TweetEmotion,
        "hellaswag": HellaSwag,
        "cosmos_qa": CosmosQA,
        "nq_open": NQOpen,
        "trivia_qa": TriviaQA
    }
    if not name in name2dataset:
        raise KeyError(f"Unrecognized dataset {name}")
    return name2dataset[name]()
