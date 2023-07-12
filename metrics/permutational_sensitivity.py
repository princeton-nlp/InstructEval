import copy
import random
import statistics
from typing import Any, List, Tuple, Dict

import datasets
from tqdm import tqdm

from data import Dataset
from decoders.decoder import Decoder
from metrics.metric import Metric
from models import BaseModel
from templates import FewShotTemplate


class PermutationalSensitivityMetric(Metric):
    def __init__(
        self,
        model: BaseModel,
        dataset: Dataset,
        template: FewShotTemplate,
        decoder: Decoder,
        num_demonstrations: int,
        num_combinations: int,
        num_permutations: int,
        num_test_instances: int,
    ):
        """
            Metric for evaluating permutational sensitivity.

            model: model to evaluate.
            dataset: dataset to evaluate on.
            template: template to use for generating prompts.
            decoder: decoder to use for decoding.
            num_demonstrations: K for K-shot learning.
            num_combinations: number of combinations of K-shot learning to try.
            num_permutations: number of permutations to try for each combination.
            num_test_instances: number of test instances to evaluate on.
        """

        super().__init__(model, dataset, template, decoder)
        self.num_demonstrations = num_demonstrations
        self.num_combinations = num_combinations
        self.num_permutations = num_permutations
        self.num_test_instances = num_test_instances

    def create_inputs(self) -> Tuple[List[List[List[Dict[str, Any]]]], List[Dict[str, Any]]]:
        # create inputs for calculating permutational robustness

        combinations_list = []
        # for each combination of demonstrations
        for seed in range(self.num_combinations):
            demonstration_instances = self.dataset.sample_instances("train", self.num_demonstrations, seed=seed)
            permutations = []
            # create num_permutations permutations
            for _ in range(self.num_permutations):
                random.shuffle(demonstration_instances)
                permutations.append(copy.deepcopy(demonstration_instances))
            combinations_list.append(permutations) 
        
        # sample test instances
        test_instances = self.dataset.sample_instances(split="test", sample_size=self.num_test_instances)
        return (combinations_list,  # list of combinations; each combination is a list of permutations; each permutation is a list of demonstrations
                test_instances)

    def evaluate(
        self, 
        inputs: Tuple[List[List[List[Dict[str, Any]]]], List[Dict[str, Any]]]
    ) -> Dict[str, Any]:

        # unpack inputs
        combinations_list, test_instances = inputs

        # remove labels from test instances
        test_instances_no_label = datasets.Dataset.from_list(test_instances).remove_columns([self.dataset.label_key])
        test_instance_labels = [test_instance[self.dataset.label_key] for test_instance in test_instances]

        # evaluate each combination of demonstrations
        permutation_stdevs = []
        for permutations_list in tqdm(combinations_list):
            permutationwise_accuracies = []
            # evaluate each permutation on full test set
            for demonstrations_list in permutations_list:
                predicted_outputs = [
                    output["prediction"]
                    for output in self.decoder.decode(
                        self.model,
                        demonstrations_list,
                        test_instances_no_label,
                    )
                ]
                # This metric uses exact match for correctness
                correctness_indicators = [
                    self.eq_metric(predicted_output, gt_output)
                    for gt_output, predicted_output in zip(
                        test_instance_labels, predicted_outputs
                    )
                ]
                # calculate accuracy for this permutation
                permutationwise_accuracies.append(statistics.mean(correctness_indicators))
            permutation_stdevs.append(statistics.stdev(permutationwise_accuracies))

        # return mean permutational stdev, and all permutational stdevs
        return {
            "permutational_sensitivity": statistics.mean(permutation_stdevs),
            "all_permutational_stdevs": permutation_stdevs
        }
