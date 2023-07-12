from typing import Tuple, List, Optional, Dict, Any, Union

import torch

from decoders.decoder import Decoder
from models.causal_lm import CausalLM
from templates.few_shot_template import FewShotTemplate


class GreedyGeneration(Decoder):
    """
    Decoder that uses the language model to greedily generate tokens when
    conditioned on the prompt. Meant for generation tasks where the label-space
    is unconstrained.
    
    Does not assume any label_map in the template. Uses only the language model 
    to generate the output.
    """

    def __init__(self, template: FewShotTemplate, max_length: int = 10):
        self.max_length = max_length
        super().__init__(template)

    def decode(
        self,
        model: CausalLM,
        demonstrations: List[Dict[str, Any]],
        test_examples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        """
            model: model to use for decoding.
            demonstrations: list of in-context demonstrations to use for decoding.
            test_examples: list of test examples to decode.
        """

        def tokenize(text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
            return model.tokenizer(text).input_ids

        # generate prompts for each test example and tokenize them
        prompts = [self.template.render(demonstrations, test_example) for test_example in test_examples]
        prompt_ids = tokenize(prompts)

        # get the longest common prefix of the prompts. Contains the few-shot demonstrations.
        lc_prefix_ids = self._longest_common_prefix(prompt_ids)
        past_key_values, _ = self._get_forward_cache(model, lc_prefix_ids)

        results = []
        for prompt in prompts:
            # find tokens remaining after removing the prefix
            input_ids = tokenize(prompt.rstrip())[len(lc_prefix_ids):]

            # generate continuation using greedy sampling
            generated_ids = self._greedy_sampling(
                model,
                input_ids,
                max_length=self.max_length,
                past_key_values=past_key_values,
            )

            # convert generated tokens to text
            generated_text = model.tokenizer.decode(generated_ids)
            prediction = generated_text.splitlines()[0]
            results.append({"prediction": prediction})

        return results

    def _greedy_sampling(
        self,
        model: CausalLM,
        input_ids: List[int],
        max_length: int,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ) -> List[int]:
        """Generate tokens using greedy sampling."""

        input_ids = torch.tensor([input_ids], dtype=int).to(model.device)
        generated_ids = []

        # Generate the next token using greedy sampling
        for _ in range(max_length):
            with torch.no_grad():
                outputs = model.forward(input_ids, past_key_values=past_key_values)
                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

            # Set input_ids to generated token and update past_key_values
            input_ids = next_token
            past_key_values = outputs.past_key_values

            # Append generated token to the list
            generated_ids.append(next_token.squeeze().item())

        # Return generated tokens
        return generated_ids

    def _get_forward_cache(
        self,
        model: CausalLM,
        input_ids: List[int],
    ) -> Tuple[Optional[Tuple[Tuple[torch.FloatTensor]]], Optional[torch.Tensor]]:
        # computes a forward pass on the input_ids, and returns the  
        # corresponding past_key_values and past_last_logit

        if len(input_ids) == 0:
            return None, None
        
        with torch.no_grad():
            input_ids = torch.tensor([input_ids], dtype=int).to(model.device)
            model_output = model.hf_model.forward(
                input_ids=input_ids,
                use_cache=True
            )

        past_key_values = model_output["past_key_values"]
        past_last_logit = model_output["logits"][:, -1, :]

        return past_key_values, past_last_logit

    def _longest_common_prefix(self, id_lists: List[List[int]]):
        if len(id_lists) == 1:
            return id_lists[0]
        ids_sorted = sorted(id_lists)
        first = ids_sorted[0]
        last = ids_sorted[-1]
        for i in range(min(len(first), len(last))):
            if first[i] != last[i]:
                return first[:i]
        return first if len(first) < len(last) else last