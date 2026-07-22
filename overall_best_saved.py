import glob
import torch
import sys
import os
from anomalib.metrics import AUROC, PRO, AUPRO
import matplotlib.pyplot as plt
from collections import OrderedDict

im_auroc_sorted = torch.load("figures/best_new.pt", weights_only=False)

xlim = (0.0, 1.0)
ylim = (0.0, 1.0)

im_auroc_fig, ax = plt.subplots(4,1, figsize=(13,15))    

colors = ["red", "green", "pink", "yellow", "purple"]
counter = 0
axis_nr = 0
axis = ax[0]
for entry in im_auroc_sorted:
    if counter > 3:
        counter = 0
        axis_nr = axis_nr + 1
        axis = ax[axis_nr]
    
    legend = True
    for val in entry["values"]:
        if legend:
            p = axis.bar(
                val["label"],
                val["value"],
                color=colors[counter],
                label=entry["class"],
                figure=im_auroc_fig,
                width=0.5
            )
            legend = False
        else:
            p = axis.bar(
                val["label"],
                val["value"],
                color=colors[counter],
                figure=im_auroc_fig,
                width=0.5
            )
        axis.bar_label(p, label_type="center", fmt="%.2f")
    counter = counter + 1
   
#im_auroc_axs.set_xlim(xlim)
im_auroc_fig.set_layout_engine("tight")
for a in ax:
    plt.setp(a.get_xticklabels(), rotation=45, ha='right', size="small")
    a.set_ylim(ylim)
    a.set_ylabel("Image AUROC Score")
    box = a.get_position()
    #a.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    a.legend(loc="center left", fontsize="small", bbox_to_anchor=(0.965, 0.5))

ax[0].set_title(f"Best results per Category")
ax[-1].set_xlabel("Error-Type/Prompt")
plt.savefig(f"figures/best_new.png")
