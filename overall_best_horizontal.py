import glob
import torch
import sys
import os
from anomalib.metrics import AUROC, PRO, AUPRO
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

im_auroc_sorted = torch.load("figures/overall_best.pt", weights_only=False)
worst = torch.load("figures/overall_worst.pt", weights_only=False)
df = torch.load("figures/df.pt", weights_only=False)
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
#ax.margins(y=0.5)
ax.set_ylim(len(df), -1)
plt.savefig("figures/best.png")

'''
counter = 0   
for entry in worst:
    
    for val in entry["values"]:
        p = ax.barh(
            val["label"],
            val["value"],
            color=colors[counter],
            figure=im_auroc_fig,
            hatch='/'
        )
        ax.bar_label(p, label_type="center", fmt="%.2f")
    counter = counter + 1
counter = 0
for entry in im_auroc_sorted:
    
    legend = True
    for val in entry["values"]:
        if legend:
            p = ax.barh(
                val["label"],
                val["value"],
                color=colors[counter],
                label=entry["class"],
                figure=im_auroc_fig,
            )
            legend = False
        else:
            p = ax.barh(
                val["label"],
                val["value"],
                color=colors[counter],
                figure=im_auroc_fig,
            )
        ax.bar_label(p, label_type="center", fmt="%.2f")
    counter = counter + 1
 


#im_auroc_axs.set_xlim(xlim)
im_auroc_fig.set_layout_engine("tight")
plt.setp(ax.get_yticklabels(), size="small")
ax.set_xlim(ylim)
ax.set_xlabel("Image AUROC Score")
box = ax.get_position()
    #a.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc="center left", fontsize="small", bbox_to_anchor=(1, 0.5))

ax.set_title(f"Best and Worst Results per Category")
ax.set_xlabel("Error-Type/Prompt")
plt.savefig(f"figures/best_and_worst.png")
'''