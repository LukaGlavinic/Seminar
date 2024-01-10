import torch
import os

def load_indices_poisoned_data(path='C:\\Users\\Luka\\GitHub\\Seminar\\poisoned_data\\cifar10_poisoned_0.1_0_train.pt'):
    poisoned_data = None
    if os.path.exists(path):
        poisoned_data = torch.load(path)
    else:
        print(f"File not found: {path}")
    return poisoned_data

def check_poisoned_data():
    poisoned_data = load_indices_poisoned_data()
    print("Učitani podaci su tenzor s veličinom:", poisoned_data.shape)
    element = poisoned_data[0]
    print(element)
    print(len(poisoned_data))


check_poisoned_data()