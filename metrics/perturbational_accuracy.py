from typing import List, Dict, Tuple, Any
import copy
import statistics
from random import Random

import datasets
from helm.benchmark.augmentations.mild_mix_perturbation import MildMixPerturbation
from tqdm import tqdm

from data import Dataset
from decoders import Decoder
from metrics.metric import Metric
from models.base import BaseModel
from templates.few_shot_template import FewShotTemplate


class PerturbationalAccuracyMetric(Metric):
    """From Holistic Evaluation of Language Models
    Credit to: https://github.com/stanford-crfm/helm
    """

    def __init__(
        self,
        model: BaseModel,
        dataset: Dataset,
        template: FewShotTemplate,
        decoder: Decoder,
        num_demonstrations: int,
        num_combinations: int,
        num_test_instances: int,
        seed: int = 0
    ):
        """
            Metric for evaluating few-shot perturbation accuracy.

            model: model to evaluate.
            dataset: dataset to evaluate on.
            template: template to use for generating prompts.
            decoder: decoder to use for decoding.
            num_demonstrations: K for K-shot learning.
            num_combinations: number of combinations of K-shot learning to try.
            num_test_instances: number of test instances to evaluate on.
        """

        super().__init__(model, dataset, template, decoder)
        self.num_demonstrations = num_demonstrations
        self.num_combinations = num_combinations
        self.num_test_instances = num_test_instances
        
        # initialize random number generator
        self.rng = Random()
        self.rng.seed(seed)

        # initialize HELM perturbation object
        self.mild_mix_perturbation = MildMixPerturbation()
    
    def _apply_perturbation(self, example: Dict[str, Any]) -> Dict[str, Any]:
        # apply perturbation to all text fields in an example
        
        example_copy = copy.deepcopy(example)
        for text_key in self.dataset.text_keys:
            example_copy[text_key] = self.mild_mix_perturbation.perturb(
                example_copy[text_key], self.rng
            )
            
        return example_copy

    def create_inputs(self) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        # create inputs for calculating perturbation accuracy

        demonstrations_list = []
        for seed in range(self.num_combinations):
            demonstration_instances = self.dataset.sample_instances("train", self.num_demonstrations, seed=seed)
            demonstrations_list.append(demonstration_instances)
        
        test_instances = self.dataset.sample_instances("test", self.num_test_instances)
        return (demonstrations_list, test_instances)

    def evaluate(
        self, 
        inputs: Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]
    ) -> Dict[str, Any]:

        # unpack inputs
        demonstrations_list, test_instances = inputs

        # remove labels from test instances
        test_instances_no_label = datasets.Dataset.from_list(test_instances).remove_columns([self.dataset.label_key])
        test_instance_labels = [test_instance[self.dataset.label_key] for test_instance in test_instances]

        # apply perturbation to all text fields in test instances
        test_instances_perturbed_no_label = test_instances_no_label.map(self._apply_perturbation)

        # evaluate each combination of demonstrations on perturbed and unperturbed test instances
        accuracies_unperturbed = []
        accuracies_perturbed = []
        for demonstrations in tqdm(demonstrations_list):
            predicted_outputs_unperturbed = [
                output["prediction"]
                for output in self.decoder.decode(
                    self.model,
                    demonstrations,
                    test_instances_no_label,
                )
            ]
            predicted_outputs_perturbed = [
                output["prediction"]
                for output in self.decoder.decode(
                    self.model,
                    demonstrations,
                    test_instances_perturbed_no_label,
                )
            ]

            # This metric uses exact match for correctness
            correctness_indicators_unperturbed = [
                self.eq_metric(predicted_output, gt_output)
                for gt_output, predicted_output in zip(
                    test_instance_labels, predicted_outputs_unperturbed
                )
            ]
            correctness_indicators_perturbed = [
                self.eq_metric(predicted_output, gt_output)
                for gt_output, predicted_output in zip(
                    test_instance_labels, predicted_outputs_perturbed
                )
            ]

            # compute accuracy
            accuracies_unperturbed.append(statistics.mean(correctness_indicators_unperturbed))
            accuracies_perturbed.append(statistics.mean(correctness_indicators_perturbed))
        
        # compute accuracy statistics
        mean_accuracy_unperturbed = statistics.mean(accuracies_unperturbed)
        mean_accuracy_perturbed = statistics.mean(accuracies_perturbed)
        mean_accuracy_drop = mean_accuracy_unperturbed - mean_accuracy_perturbed
        
        # return accuracies
        return {
            "unperturbed_accuracy": mean_accuracy_unperturbed,
            "perturbed_accuracy": mean_accuracy_perturbed,
            "perturbation_drop_in_accuracy": mean_accuracy_drop,
            "all_unperturbed_accuracies": accuracies_unperturbed,
            "all_perturbed_accuracies": accuracies_perturbed
        }
