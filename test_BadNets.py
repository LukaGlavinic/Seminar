'''
This is the test code of poisoned training under BadNets.
'''
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor

from BadNets import BadNets
from models.resnet import ResNet


def train_BadNets_noCleaning(poisoned_indeces):
    # ========== Set global settings ==========
    global_seed = 666
    deterministic = True
    torch.manual_seed(global_seed)
    # CUDA_SELECTED_DEVICES = '0,3'
    CUDA_SELECTED_DEVICES = '0'
    SAVE_DIR_EXP = Path(__file__).parent / 'experiments'
    DATA_DIR_ROOT = Path(__file__).parent / 'datasets'

    # ========== ResNet-18_CIFAR-10_BadNets ==========
    dataset = torchvision.datasets.CIFAR10

    transform_train = Compose([
        RandomHorizontalFlip(),
        ToTensor()
    ])
    trainset = dataset(DATA_DIR_ROOT, train=True, transform=transform_train, download=True)

    transform_test = Compose([
        ToTensor()
    ])
    testset = dataset(DATA_DIR_ROOT, train=False, transform=transform_test, download=True)

    pattern = torch.zeros((32, 32), dtype=torch.uint8)
    pattern[-3:, -3:] = 255
    weight = torch.zeros((32, 32), dtype=torch.float32)
    weight[-3:, -3:] = 1.0

    # Train Attacked Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/CIFAR/train_watermarked.py)
    schedule = {
        # 'device': 'GPU',
        # 'CUDA_SELECTED_DEVICES': CUDA_SELECTED_DEVICES,

        'benign_training': False,
        'batch_size': 1024,
        'num_workers': 4,

        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'schedule': [150, 180],

        'epochs': 200,

        'log_iteration_interval': 100,
        'test_epoch_interval': 10,
        'save_epoch_interval': 10,

        'save_dir': SAVE_DIR_EXP,
        'experiment_name': 'ResNet18_CIFAR10_BadNets_noCleaning'
    }

    badnets = BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=ResNet(18),
    loss=nn.CrossEntropyLoss(),
    y_target=0,
    poisoned_rate=0.05,
    poisoned_indeces=poisoned_indeces,
    pattern=pattern,
    weight=weight,
    seed=global_seed,
    deterministic=deterministic
)
    badnets.train(schedule)
    badnets.test(schedule)