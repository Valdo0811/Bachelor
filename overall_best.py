import glob
import torch
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
import pandas as pd


pix_auroc_prompts = {}
im_auroc_prompts = {}
aupro_prompts = {}
im_auroc = {}
im_auroc_res = {}
pix_auroc = {}
pix_auroc_res = {}
aupro = {}
aupro_res = {}
category_average = {}
prompts = {}

bests = {}

data = []

folder = sys.argv[1] + "/*/*" + ".pt"
subdirectories = [os.path.basename(path) for path in glob.glob(f'{sys.argv[1]}/*')]
for dir in subdirectories:
        nr_of_prompts = 0
        subs = [os.path.basename(path) for path in glob.glob(f'{sys.argv[1]}/{dir}/*')]
        cat_bests = {}
        cat_worsts = {}
        cat_best = 0
        for sub in subs:
            cat_err = dir + "/" + sub
            for filename in sorted(glob.glob(f'{sys.argv[1]}/{dir}/{sub}/*.pt')):
                res = torch.load(filename, weights_only=False)
                val = res["im_auroc_res"].item()
                del res
                
                x = filename.split('\\')
                x = x[-1].split('.')
                prompt = x[0]
                 
                nr_of_prompts = nr_of_prompts + 1
                if cat_best <= val:
                        cat_best = val
                
                if cat_err in prompts:
            
                    if cat_bests[cat_err] <= val:
                        cat_bests[cat_err] = val
                        prompts[cat_err] = prompt
                    
                    if cat_worsts[cat_err] >= val:
                        cat_worsts[cat_err] = val
                        
                else:   
                    cat_worsts[cat_err] = val
                    cat_bests[cat_err] = val
                    prompts[cat_err] = prompt
            
            data.append([dir, sub+"/"+prompts[cat_err], prompts[cat_err], cat_bests[cat_err], cat_worsts[cat_err]])
                    
                
        cat_bests["best"] = cat_best            
        bests[dir] = cat_bests

df = pd.DataFrame(data=data, columns=["category", "Errortype/Prompt", "prompt", "AUROC Score", "min"])
#torch.save(df, "best.pt")
#df = torch.load("best.pt", weights_only=False)
xlim = (0.0, 1.0)
ylim = (0.0, 1.0)


category_order = (
    df.groupby("category")["AUROC Score"]
      .max()
      .sort_values(ascending=False)
      .index
)

df["category"] = pd.Categorical(
    df["category"],
    categories=category_order,
    ordered=True,
)

df = df.sort_values(
    ["category", "AUROC Score"],
    ascending=[True, False]
)


im_auroc_fig, ax = plt.subplots(figsize=(12,15))    

colors = ["#4875ff",
            "#63bf2d",
            "#881c9d",
            "#c49b00",
            "#2c4ab2",
            "#f27f0c",
            "#6eb9ff",
            "#7f5d00",
            "#ff6ee5",
            "#1d6009",
            "#d0006a",
            "#019c66",
            "#9c2146",
            "#739463",
            "#ff8098"]

duplicates = df["Errortype/Prompt"].duplicated(keep=False)


df.loc[duplicates, "Errortype/Prompt"] = (
    df.loc[duplicates, "category"].astype(str) + ": " + df.loc[duplicates, "Errortype/Prompt"]
)

p = sns.barplot(x="AUROC Score", y="Errortype/Prompt", data=df, hue="category", dodge=False, palette=colors)
sns.scatterplot(
    data=df,
    y="Errortype/Prompt",
    x="min",
    color="black",
    marker=".",
    s=60,
    zorder=3,
    ax=ax,
)
handles, labels = ax.get_legend_handles_labels()

handles.append(
    Line2D(
        [0],
        [0],
        marker=".",
        color="w",
        markerfacecolor="black",
        markersize=8,
        label="worst score"
    )
)
for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f",
        padding=3
    )
im_auroc_fig.set_layout_engine("tight")
plt.title("Best Image-AUROC Scores per Errortype")
plt.ylabel("Errortype/Prompt")
plt.xticks(size="small")
ax.legend(handles=handles, title="Category")
ax.set_ylim(len(df), -1)
plt.savefig("figures/best.png")


