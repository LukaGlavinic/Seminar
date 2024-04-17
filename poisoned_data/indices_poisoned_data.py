import os
from pathlib import Path

import torch

PATH_TO_INDECES = Path(__file__).parent / 'cifar10_poisoned_0.1_0_train.pt'

def load_indices_poisoned_data(path=PATH_TO_INDECES):
    indices_poisoned_data = None
    if os.path.exists(path):
        indices_poisoned_data = torch.load(path)
    else:
        print(f"File not found: {path}")
    return indices_poisoned_data

def check_poisoned_data():
    indices_poisoned_data = load_indices_poisoned_data()
    print(f"Učitani podaci su tenzor s veličinom: {str(indices_poisoned_data.shape)}")
    element = indices_poisoned_data[0]
    print(f"Prvi element je {str(element)}")


# check_poisoned_data()
