import glob
import torch
import sys
import os
from anomalib.metrics import AUROC, PRO, AUPRO
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd


data = []


folder = "predictions/*/*" + ".pt"
subdirectories = [os.path.basename(path) for path in glob.glob(f'predictions/*')]
for dir in subdirectories:
        subs = [os.path.basename(path) for path in glob.glob(f'predictions/{dir}/*')]
        for sub in subs:
            cat_err = dir + "/" + sub
            for filename in sorted(glob.glob(f'predictions/{dir}/{sub}/*.pt')):
                batch = torch.load(filename, weights_only=False)
                
                x = filename.split('\\')
                x = x[-1].split('.')
                prompt = x[0]
                
                for image in batch:
                    
                    aupro = AUPRO(fields=["anomaly_map", "gt_mask"])
                    aupro.update(batch)
                    aupro_res = aupro.compute()
                    data.append([dir, sub, prompt, image.image, image.gt_mask, image.anomaly_map, aupro_res])
                del batch
                print(data)
                df = pd.DataFrame(data=data, columns=["category", "error type", "prompt", "image", "gt mask", "anomaly map", "AUPRO score"])
                torch.save(df, "figures/every_image.pt")
                    

df = pd.DataFrame(data=data, columns=["category", "error type", "prompt", "image", "gt mask", "anomaly map", "AUPRO score"])

group_cols = ["category"]

# Indices of lowest and highest AUPRO score per group
idx_min = df.groupby(group_cols)["AUPRO score"].idxmin()
idx_max = df.groupby(group_cols)["AUPRO score"].idxmax()

lowest = df.loc[idx_min].reset_index(drop=True)
highest = df.loc[idx_max].reset_index(drop=True)

print(highest)

torch.save(highest, "figures/highest_per_cat")
torch.save(lowest, "figures/lowest_per_cat")

group_cols = ["category", "error type"]

# Indices of lowest and highest AUPRO score per group
idx_min = df.groupby(group_cols)["AUPRO score"].idxmin()
idx_max = df.groupby(group_cols)["AUPRO score"].idxmax()

lowest = df.loc[idx_min].reset_index(drop=True)
highest = df.loc[idx_max].reset_index(drop=True)

print(highest)

torch.save(highest, "figures/highest_per_type")
torch.save(lowest, "figures/lowest_per_type")

torch.save(df, "figures/every_image.pt")

