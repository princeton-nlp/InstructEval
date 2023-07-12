from typing import Optional, Dict, Any, List
from jinja2 import Environment

from data import Dataset, get_dataset


class FewShotTemplate:

    def __init__(self,
                 jinja2_file_path: Optional[str] = None,
                 jinja2_string: Optional[str] = None):
        """
        General few shot template class.

        jinja2_file_path: path to a jinja2 template file.
        jinja2_string: string containing a jinja2 template.

        Accepts either a path to a jinja2 template file or a string containing the template.

        - The file must also define `dataset_name` which represents the name
          of the dataset, used in data.get_dataset.
        - To reference few shot examples, the template should use `demonstrations`.
        - To reference the test example, the template should use `test_example`.
        """

        if not (jinja2_file_path or jinja2_string):
            raise ValueError("Neither path to jinja2 template or string jinja2 template were provided.")
        elif jinja2_file_path and jinja2_string:
            raise ValueError("You only need to specify one of jinja2_file_path or jinja2_string not both.")
        elif jinja2_file_path:
            jinja2_string = open(jinja2_file_path, "r").read()

        self.template = Environment().from_string(jinja2_string)
        module = self._get_dummy_module()

        if "dataset_name" not in dir(module):
            raise ValueError("You must ensure your jinja2 template sets a `dataset_name`.")
        self.dataset_name = module.dataset_name
        
        # label_map is only expected for classification datasets
        if "label_map" in dir(module):
            self.label_map = module.label_map


    def _get_dummy_module(self) -> Any:
        # dummy context to extract the label map and the dataset name
        # `endings` is used in MCQ datasets
        dummy_context = {
            "dataset_name": None,
            "label_map": None,
            "label": None,
            "demonstrations": [],
            "test_example": {
                "": "",
                "endings": [],
                "answer": None,
                "label": None
            }
        }
        module = self.template.make_module(dummy_context)
        return module

    def get_dataset(self) -> Dataset:
        # Return the dataset specified in the prompt template
        return get_dataset(self.dataset_name)

    def render(self, 
        demonstrations: List[Dict[str, Any]], 
        test_example: Optional[Dict[str, Any]] = None
    ) -> str:
        # Render an open prompt using a list of demonstrations and a test example.
        return self.template.render(demonstrations=demonstrations,
                                    test_example=test_example)
