import yaml
import os

def load_config(path: str):
    """
    Load YAML config file and return as dictionary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
