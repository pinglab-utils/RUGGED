import re
import json
from pathlib import Path

def extract_code(response: str):
    code_blocks = re.findall(r'```(.*?)```', response, re.DOTALL)
    # Combine code to be one block
    code = '\n'.join(code_blocks)
    return code


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def format_node_name(node_id, node_names):
    if node_id not in node_names:
        return f"{node_id} No known names"
    if node_names[0] == "No known names":
        return f"{node_id} ({node_names[0]})"
    return f"{node_names[0]} ({node_id})"

