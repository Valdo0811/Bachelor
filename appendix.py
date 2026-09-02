import glob
import torch
import sys
import os
from anomalib.metrics import AUROC, PRO, AUPRO
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd


df = torch.load("figures/every_image.pt", weights_only=False)

group_cols = ["category"]
print(len(df.index))
# Indices of lowest and highest AUPRO score per group
idx_min = df.groupby(group_cols)["AUPRO score"].idxmin()
idx_max = df.groupby(group_cols)["AUPRO score"].idxmax()

lowest = df.loc[idx_min].reset_index(drop=True)
highest = df.loc[idx_max].reset_index(drop=True)

print(highest)
print(lowest)
for i in range(len(lowest.index)):
    low_entry = lowest.loc[i]
    low = f'predictions/{low_entry["category"]}/{low_entry["error type"]}/{low_entry["prompt"]}.pt'
    
    
    high_entry = highest.loc[i]
    high = f'predictions/{high_entry["category"]}/{high_entry["error type"]}/{high_entry["prompt"]}.pt'
    