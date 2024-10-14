from pathlib import Path
import re
import json


def extract_code(response: str):
    code_blocks = re.findall(r'```(.*?)```', response, re.DOTALL)
    # Combine code to be one block
    code = '\n'.join(code_blocks)
    return code


def get_project_root() -> Path:
    return Path(__file__).parent.parent
