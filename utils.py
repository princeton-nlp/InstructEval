from typing import Any, Dict
import argparse
import hashlib
import json
import os
import re
import unicodedata

from data import *
from decoders import *
from metrics import *
from models import *
from templates import *


def slugify(value: Any, allow_unicode=False) -> str:
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (unicodedata.normalize("NFKD",
                                       value).encode("ascii",
                                                     "ignore").decode("ascii"))
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def dict2namespace(config: dict) -> argparse.Namespace:
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def hash_dict(dictionary: Dict[Any, Any]) -> str:
    dict_string = "\n".join([f"{key}: {value}" for key, value in dictionary.items()])
    sha = hashlib.sha256()
    sha.update(dict_string.encode())
    hashed_dict = sha.hexdigest()[:16]
    return hashed_dict

def get_filename_from_metadata(metadata: Dict[str, Any]) -> str:
    hashed_metadata = hash_dict(metadata)
    return f"{hashed_metadata}.json"

def write_results(results_dir: str, filename: str, metadata: Dict[str, Any], results: Dict[str, Any]) -> None:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    result_path = os.path.join(results_dir, filename)
    print(f"Writing results to {result_path}...")

    log_dict = {
        "metadata": metadata,
        "results": results
    }

    with open(result_path, "w", encoding='utf-8') as f:
        json.dump(log_dict, f, ensure_ascii=False, indent=4)


def default_decoder_name(task_type: str) -> str:
    if task_type == "CLS":
        return "constrained_label_generation"
    elif task_type == "MCQ":
        return "constrained_per_example_label_generation"
    elif task_type == "GQA":
        return "greedy_generation"
    else:
        raise KeyError(f"Unrecognized task type {task_type}")


def get_model(model_name: str) -> BaseModel:
    model_name = slugify(model_name)
    model_to_class_map = {
        "gptneo1b3": GPTNeo1B3,
        "gptneo2b7": GPTNeo2B7,
        "gptneox20b": GPTNeoX20B,
        "bloom1b1": Bloom1B1,
        "bloom1b7": Bloom1B7,
        "bloom3b": Bloom3B,
        "bloom7b1": Bloom7B1,
        "llama7b": LLaMA7B,
        "llama13b": LLaMA13B,
        "opt1b3": OPT1B3,
        "opt2b7": OPT2B7,
        "opt6b7": OPT6B7,
        "opt13b": OPT13B,
        "stablelmbase3b": StableLMBase3B,
        "stablelmbase7b": StableLMBase7B,
        "stablelmtuned3b": StableLMTuned3B,
        "stablelmtuned7b": StableLMTuned7B,
    }
    if model_name not in model_to_class_map:
        raise KeyError(f"Unrecognized model {model_name}")

    return model_to_class_map[model_name]()


def get_decoder(decoder_name: str, template: FewShotTemplate, dataset: Dataset) -> Decoder:
    decoder_name = slugify(decoder_name)
    if decoder_name == "constrained_label_generation":
        return ConstrainedLabelGeneration(template)
    elif decoder_name == "nucleus_generation":
        return NucleusGeneration(template)
    elif decoder_name == "greedy_generation":
        return GreedyGeneration(template)
    elif decoder_name == "constrained_per_example_label_generation":
        return ConstrainedPerExampleLabelGeneration(template, dataset)
    else:
        raise KeyError("Unrecognized decoder {decoder_name}")


def get_metric(
    metric_name: str,
    model: BaseModel,
    dataset: Dataset,
    template: FewShotTemplate,
    decoder: Decoder,
    metric_config: dict
) -> Metric:
    metric_name = slugify(metric_name)
    metric_to_class_map = {
        "zero_shot_accuracy": ZeroShotAccuracyMetric,
        "few_shot_accuracy": FewShotAccuracyMetric,
        "perturbational_accuracy": PerturbationalAccuracyMetric,
        "selectional_sensitivity": SelectionalSensitivityMetric,
        "permutational_sensitivity": PermutationalSensitivityMetric
    }
    if metric_name not in metric_to_class_map:
        raise KeyError(f"Unrecognized metric {metric_name}")

    metric_class = metric_to_class_map[metric_name]
    return metric_class(
        model=model,
        dataset=dataset,
        template=template,
        decoder=decoder,
        **metric_config[metric_name],
    )
