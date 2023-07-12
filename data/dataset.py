import datasets
import random
from typing import List, Dict, Any

class Dataset:

    def __init__(self, 
                 name: str, 
                 task_type: str, 
                 train_split: datasets.Dataset, 
                 test_split: datasets.Dataset, 
                 text_keys: List[str] = ["text"], 
                 label_key: str = "label"):
        
        """
            Parent class for all datasets.
            
            name: str corresponding to the name of the dataset. (usually matches with filename)
            task_type: str corresponding to the type of task. (CLS, MCQ, GQA)
            train_split: huggingface dataset corresponding to the training set. 
            test_split: huggingface dataset corresponding to the testing set.
            text_keys: keys representing long text fields.
            label_key: key that represents the prediction label.
        """

        self.name = name
        self.task_type = task_type
        self.label_key = label_key
        self.text_keys = text_keys
        self.splits = {"train": train_split, "test": test_split}

        # get unique classes and indices by class for each split for classification datasets
        if self.task_type == "CLS":    
            self.classes = set(self.splits["train"][self.label_key])
            self.num_classes = len(self.classes)
            self.idxs_by_class = {split_name: self._get_idxs_by_class(split) for split_name, split in self.splits.items()}

    def _get_idxs_by_class(self, dataset: datasets.Dataset) -> Dict[str, List[int]]:
        idxs_by_class = {label: [] for label in self.classes}
        for i, label in enumerate(dataset[self.label_key]):
            idxs_by_class[label].append(i)
        return idxs_by_class

    def sample_instances(self,
                         split: str,
                         sample_size: int,
                         seed: int = 0,
                         balanced_sampling: bool = True,
                         max_words: int = 100) -> List[Dict[str, Any]]:
        
        dataset = self.splits[split]
        if sample_size > len(dataset):
            raise ValueError(f"Sample size {sample_size} is larger than dataset size {len(dataset)}")
        
        random.seed(seed)
        sampled_indices = []
        # balanced sampling is only used for classification datasets
        if (self.task_type == "CLS") and balanced_sampling:
            dataset_by_label = self.idxs_by_class[split]

            # compute number of instances to sample per class
            naive_count_per_class = sample_size // self.num_classes
            num_remaining = sample_size % self.num_classes
            counts_per_class = \
                [naive_count_per_class] * (self.num_classes - num_remaining) + \
                [naive_count_per_class + 1] * num_remaining
            
            # sample instances from each class
            for count, label in zip(counts_per_class, self.classes):
                if len(dataset_by_label[label]) < count:
                    raise ValueError(f"Insufficient number of instances for label {label} in split {split}. {len(dataset_by_label[label])} < {count}")
                sampled_indices.extend(random.sample(dataset_by_label[label], count))
            
            # shuffle the sampled indices
            random.shuffle(sampled_indices)
        else:
            sampled_indices.extend(random.sample(range(len(dataset)), sample_size))

        # get instances corresponding to sampled_indices from dataset
        sampled_instances = dataset[sampled_indices]
        sampled_instances = [dict(zip(sampled_instances, t)) for t in zip(*sampled_instances.values())]  # convert to List[Dict]

        # truncate the text field at each instance
        if max_words:
            for instance in sampled_instances:
                for text_key in self.text_keys:
                    words = instance[text_key].split(" ") 
                    if len(words) > max_words:
                        instance[text_key] = " ".join(words[:max_words]) + "..."

        return sampled_instances
