import torch
import pandas as pd
import dataframe_image as dfi

df = pd.read_csv("results.csv")

good = df[df["image_auroc"] > 0.85]["error_type/prompt"]
mid = df[(df["image_auroc"] > 0.70) & (df["image_auroc"] <= 0.85)]["error_type/prompt"]
bad = df[df["image_auroc"] <= 0.70]["error_type/prompt"]

df = pd.DataFrame({
    "bad score (<= 0.70)": bad.reset_index(drop=True),
    "decent score (0.70 - 0.85)": mid.reset_index(drop=True),
    "good score (>0.85)": good.reset_index(drop=True),
})

print(df.to_latex(index=True
                  ))
dfi.export(df, "figures/classification.png")
