import glob
import torch
import sys
from anomalib.metrics import AUROC, PRO, AUPRO
import matplotlib.pyplot as plt
from collections import OrderedDict

folder_path = sys.argv[1] + "/*" + ".pt"

prompts = []
im_auroc = {}
pix_auroc = {}
aupro = {}
split_folder_path = folder_path.split('/')
category, error_type = split_folder_path[-3], split_folder_path[-2]

for filename in sorted(glob.glob(folder_path)):
        x = filename.split('\\')
        x = x[-1].split('.')
        x = x[0]
        
        prompts.append(x)
        res = torch.load(filename, weights_only=False)
        aupro[x] = res



xlim = (0.0, 1.0)
ylim = (0.0, 1.0)

'''
for key in im_auroc:
    value = im_auroc[key]
    fpr, tpr = value._compute()
    auroc = value.compute()

    xlim = (0.0, 1.0)
    ylim = (0.0, 1.0)

for key in pix_auroc:
    value = pix_auroc[key]
    fpr, tpr = value._compute()
    auroc = value.compute()

    xlim = (0.0, 1.0)
    ylim = (0.0, 1.0)
'''
aupro_fig, aupro_axs = plt.subplots()    

aupro_sorted = []

for key in aupro:
    value = aupro[key]
    fpr, tpr = value._compute()
    aupro_ = value.compute()
    xlim = (0.0, float(value.fpr_limit.detach().cpu().item()))
    
    aup = {"value": aupro_, "fpr": fpr.detach().cpu(), "tpr": tpr.detach().cpu(), "label": f"{key}: {aupro_.detach().cpu():0.2f}"}
    added = False
    if len(aupro_sorted) > 0:
        for i in range(len(aupro_sorted) - 1):
            if aupro_sorted[i]["value"] >= aupro_:
                if len(aupro_sorted) > i + 1:
                    if aupro_sorted[i+1]["value"] <= aupro_:
                        aupro_sorted.insert(i+1, aup)
                        added = True
                else:
                    aupro_sorted.append(aup) 
                    added = True
                break 
        if not added:
            aupro_sorted.append(aup)
               
    else:
        aupro_sorted.append(aup)

        

for entry in aupro_sorted:
    aupro_axs.plot(
        entry["fpr"],
        entry["tpr"],
        label=entry["label"],
        figure=aupro_fig,
        lw=2
    )
   
aupro_axs.set_xlim(xlim)
aupro_axs.set_ylim(ylim)
aupro_axs.set_xlabel("Global FPR")
aupro_axs.set_ylabel("Averaged Per-Region TPR")
aupro_axs.legend(loc="lower right", fontsize="x-small")  
aupro_axs.set_title(f"Aupro scores for \n{category}/{error_type}")

plt.savefig(f"figures/aupro/{category}/{error_type}/combined.png")