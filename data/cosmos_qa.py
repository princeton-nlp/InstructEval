import datasets

from data.dataset import Dataset


class CosmosQA(Dataset):

    def __init__(self):
        dataset = datasets.load_dataset("cosmos_qa")
        super().__init__("cosmos_qa",
                         "MCQ",
                         dataset["train"],
                         dataset["validation"],
                         text_keys=["context", "question"],
                         label_key="label")

    def get_choices_per_instance(self, instance):
        return [
            instance['answer0'], 
            instance['answer1'], 
            instance['answer2'],
            instance['answer3']
        ]
