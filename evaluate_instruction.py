from typing import Tuple, Dict, Any
import argparse
import sys

import yaml

from data import *
from decoders import *
from metrics import *
from models import *
from templates import *
from utils import *

def get_metric_name_config(args) -> Tuple[str, Dict[str, Any]]:
    with open(args.metric_config, "r") as f:
        metric_config = yaml.safe_load(f)
        metric_name = list(metric_config.keys())[0]
    return metric_name, metric_config

def get_instruction(args) -> str:
    if os.path.isfile(os.path.join(args.instructions_dir, args.dataset + ".yaml")):
        # model-agnostic instructions found
        instructions_file = os.path.join(args.instructions_dir, args.dataset + ".yaml")
    elif os.path.isfile(os.path.join(args.instructions_dir, args.model, args.dataset + ".yaml")):
        # model-specific instructions found
        instructions_file = os.path.join(args.instructions_dir, args.model, args.dataset + ".yaml")
    else:
        # no instructions found
        raise ValueError(f"No matching instructions file in {args.instructions_dir}")
     
    with open(instructions_file, "r") as f:
        instructions_list = yaml.safe_load(f)
        if args.index < 0 or args.index >= len(instructions_list):
            raise ValueError(f"Index {args.index} out of bounds for {len(instructions_list)} instructions in {instructions_file}.")
        instruction = instructions_list[args.index]
    return instruction

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--decoder", type=str, required=False, help="Decoder name")
    parser.add_argument("--metric_config", type=str, required=True, help="Metric config file")
    parser.add_argument("--instructions_dir", type=str, required=True, help="Directory containing instruction files")
    parser.add_argument( "--index", type=int, required=True, help="Index of instruction to evaluate in dataset's instruction file")
    parser.add_argument( "--prompt_template_dir", type=str, required=False, default="configs/default_prompts", help="Directory containing prompt templates for each dataset")
    parser.add_argument("--results_dir", type=str, required=False, default="results/", help="Directory to write results to")
    args = parser.parse_args()

    # initialize objects
    metric_name, metric_config = get_metric_name_config(args)
    instruction = get_instruction(args)
    prompt_template = InstructionBasedFewShotTemplate(
        instruction=instruction,
        jinja2_file_path=os.path.join(args.prompt_template_dir, args.dataset + ".j2")
    )
    dataset_name = prompt_template.dataset_name
    model = get_model(args.model)
    dataset = get_dataset(dataset_name)
    decoder_name = args.decoder or default_decoder_name(dataset.task_type)
    decoder = get_decoder(decoder_name, prompt_template, dataset)
    metric = get_metric(metric_name, model, dataset, prompt_template, decoder, metric_config)
    
    metadata_dict = {
        "model": args.model,
        "dataset": dataset_name,
        "metric": metric_name,
        "decoder": decoder_name,
        "metric_config": metric_config,
        "instruction": instruction,
        "instructions_dir": args.instructions_dir,
    }

    result_filename = get_filename_from_metadata(metadata_dict)
    result_path = os.path.join(args.results_dir, result_filename)
    if os.path.exists(result_path):
        print(f"Results already exist for this configuration at {result_path}.")
        print("Exiting...")
    else:
        # Evaluate metric
        inputs = metric.create_inputs()
        results = metric.evaluate(inputs)
        # Write results to disk
        write_results(args.results_dir, result_filename, metadata_dict, results)
