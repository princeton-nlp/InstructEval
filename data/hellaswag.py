import datasets

from data.dataset import Dataset


class HellaSwag(Dataset):

    def __init__(self):
        dataset = datasets.load_dataset("hellaswag")
        super().__init__("hellaswag",
                         "MCQ",
                         dataset["train"],
                         dataset["validation"],
                         text_keys=["ctx"],
                         label_key="label")

    def get_choices_per_instance(self, instance):
            return instance['endings']