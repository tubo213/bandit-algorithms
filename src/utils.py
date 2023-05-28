import numpy as np
import yaml


def set_seed(seed):
    np.random.seed(seed)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f, Loader=yaml.FullLoader)
