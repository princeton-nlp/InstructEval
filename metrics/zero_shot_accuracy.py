import statistics
from typing import Any, List, Dict

import datasets
from tqdm import tqdm

from data.dataset import Dataset
from decoders.decoder import Decoder
from metrics.metric import Metric
from models.base import BaseModel
from templates.few_shot_template import FewShotTemplate


class ZeroShotAccuracyMetric(Metric):

    def __init__(
        self,
        model: BaseModel,
        dataset: Dataset,
        template: FewShotTemplate,
        decoder: Decoder,
        num_test_instances: int,
    ):
        """
            Metric for evaluating zero-shot accuracy.

            model: model to evaluate.
            dataset: dataset to evaluate on.
            template: template to use for generating prompts.
            decoder: decoder to use for decoding.
            num_test_instances: number of test instances to evaluate on.
        """

        super().__init__(model, dataset, template, decoder)
        self.num_test_instances = num_test_instances

    def create_inputs(self) -> List[Dict[str, Any]]:
        # create inputs for calculating zero-shot accuracy
        
        test_instances = self.dataset.sample_instances("test", self.num_test_instances)
        return test_instances

    def evaluate(
        self, 
        inputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        # unpack inputs
        test_instances = inputs

        # remove labels from test instances
        test_instances_no_label = datasets.Dataset.from_list(test_instances).remove_columns([self.dataset.label_key])
        test_instance_labels = [test_instance[self.dataset.label_key] for test_instance in test_instances]

        # get predictions
        predicted_outputs = [
            output["prediction"]
            for output in self.decoder.decode(
                self.model,
                [],
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

        # compute accuracy
        accuracy = statistics.mean(correctness_indicators)

        # return mean few-shot accuracy, and all few-shot accuracies
        return {
            "zero_shot_accuracy": accuracy
        }
