import glob
import torch
import sys
import os
from anomalib.metrics import AUROC, PRO, AUPRO
import matplotlib.pyplot as plt
from collections import OrderedDict
import dataframe_image as dfi
import pandas as pd

'''
data = []

folder = sys.argv[1] + "/*/*" + ".pt"
subdirectories = [os.path.basename(path) for path in glob.glob(f'{sys.argv[1]}/*')]
for dir in subdirectories:
        
        subs = [os.path.basename(path) for path in glob.glob(f'{sys.argv[1]}/{dir}/*')]
        
        
        for sub in subs:
            cat_err = dir + "/" + sub
            for filename in sorted(glob.glob(f'{sys.argv[1]}/{dir}/{sub}/*.pt')):
                x = filename.split('\\')
                x = x[-1].split('.')
                prompt = x[0]
                
                if prompt not in ["broken", "damage", "damaged", "error"]:
                    continue
                
                res = torch.load(filename, weights_only=False)
                val = res["im_auroc_res"].item()
                del res
                
                
                data.append([dir, dir+"/"+sub, sub+"/"+prompt, sub, prompt, val])
                    


df = pd.DataFrame(data=data, columns=["Category", "Category/Errortype", "Errortype/Prompt", "Errortype", "Prompt", "AUROC Score"])

torch.save(df, "figures/generics.pt")
'''

df = torch.load("figures/generics.pt", weights_only=False)

df = df.sort_values(
    ["AUROC Score"],
    ascending=[False]
)

dfi.export(df[["Category", "Errortype", "Prompt", "AUROC Score"]].iloc[:25].style.hide(), "figures/generics.png")

comparison = (
    df[df["Prompt"].isin(["damaged", "damage"])]
    .pivot(index="Category/Errortype", columns="Prompt", values="AUROC Score")
    .sort_values("damaged", ascending=False)
)

print(df[df["Prompt"].isin(["damaged", "damage"])]
    .groupby("Prompt", as_index=False)["AUROC Score"]
    .mean())

#comp = comparison.style.set_caption("Count of errortypes with given Image-AUROC Score")
dfi.export(comparison.iloc[:20], "figures/damage_vs_damaged.png")





data = []
columns = ["Prompt", "0.5", "(0.5, 0.6]", "(0.6, 0.7]", "(0.7, 0.8]", "(0.8, 0.9]", "(0.9, 1]"]

for prompt in ["broken", "damage", "damaged"]:
    row = [prompt]
    row.append(((df["AUROC Score"] <= 0.50) & (df["Prompt"] == prompt)).sum())
    row.append(((0.50 < df["AUROC Score"]) & (df["AUROC Score"] <= 0.60) & (df["Prompt"] == prompt)).sum())
    row.append(((0.60 < df["AUROC Score"]) & (df["AUROC Score"] <= 0.70) & (df["Prompt"] == prompt)).sum())
    row.append(((0.70 < df["AUROC Score"]) & (df["AUROC Score"] <= 0.80) & (df["Prompt"] == prompt)).sum())
    row.append(((0.80 < df["AUROC Score"]) & (df["AUROC Score"] <= 0.90) & (df["Prompt"] == prompt)).sum())
    row.append(((df["AUROC Score"] > 0.90) & (df["Prompt"] == prompt)).sum())
    data.append(row)

df_2 = pd.DataFrame(data=data, columns=columns)
df_2 = df_2.style.set_caption("Count of errortypes with given Image-AUROC Score").hide()

dfi.export(df_2, "figures/generics_counts.png")


