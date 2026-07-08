import glob
import torch
import sys
import os
from anomalib.metrics import AUROC, PRO, AUPRO
import matplotlib.pyplot as plt
from collections import OrderedDict

im_auroc_sorted = torch.load("figures/best.pt", weights_only=False)

xlim = (0.0, 1.0)
ylim = (0.0, 1.0)

im_auroc_fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(10,10))    

colors = ["red", "green", "blue", "yellow", "purple"]
counter = 0
ax = ax1
for entry in im_auroc_sorted:
    if counter > 4:
        counter = 0
        if ax == ax1:
            ax = ax2
        else:
            ax = ax3
    
    
    p = ax.bar(
        entry["label"],
        entry["average"].item(),
        color=colors[counter],
        figure=im_auroc_fig,
        label=entry["class"],
        width=0.5
    )
    ax.bar_label(p, label_type="center", fmt="%.2f")
    for val in entry["values"]:
        p = ax.bar(
            val["label"],
            val["value"].item(),
            color=colors[counter],
            figure=im_auroc_fig,
            width=0.5
        )
        ax.bar_label(p, label_type="center", fmt="%.2f")
    counter = counter + 1
   
#im_auroc_axs.set_xlim(xlim)

plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', size="small")
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', size="small")
plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', size="small")
im_auroc_fig.set_layout_engine("constrained")
ax1.set_ylim(ylim)
ax2.set_ylim(ylim)
ax3.set_ylim(ylim)
ax3.set_xlabel("Error-Type/Prompt")
ax2.set_ylabel("True Positive Rate")
ax1.legend(loc="upper right", fontsize="x-small")  
ax1.set_title(f"Best results per Category")

plt.savefig(f"figures/best.png")
