import torch
import meerkat as mk
from meerkat.datasets.imagenette import download_imagenette
from domino import explore, DominoSlicer
import meerkat as mk
from torchvision.models import resnet18
import torchvision.transforms as transforms
import pandas as pd
import numpy  as np

print('cuda available:', torch.cuda.is_available())

dp = mk.datasets.get(
    "celeba",
    dataset_dir="C:/Users/rohil/Downloads/Uni Bonn/WiSe 2022-23/LabBD/Code/meerkat_test/meerkat-main/meerkat/datasets/celeba"
)

print('dataset shape:', dp.shape)
assert 'split' in dp.columns

dp = dp.lz[dp["split"] == "train"]
print('dataset shape after split:', dp.shape)