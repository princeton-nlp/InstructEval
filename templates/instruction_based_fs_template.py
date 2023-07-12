from typing import Optional

from templates.few_shot_template import FewShotTemplate


class InstructionBasedFewShotTemplate(FewShotTemplate):

    def __init__(self,
                 instruction: str,
                 jinja2_file_path: Optional[str] = None,
                 jinja2_string: Optional[str] = None):
        """
        Few shot template class supporting instructions.

        instruction: string containing the instruction.
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

        if "{{instruction}}" not in jinja2_string:
            raise ValueError("Your prompt template must contain the placeholder {{instruction}}.")

        jinja2_string = jinja2_string.replace("{{instruction}}", instruction)
        super().__init__(jinja2_string=jinja2_string)
