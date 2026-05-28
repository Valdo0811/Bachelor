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
        im_auroc[x] = res["image_auroc"]
        pix_auroc[x] = res["pixel_auroc"]



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
im_auroc_fig, im_auroc_axs = plt.subplots()    

im_auroc_sorted = []

for key in im_auroc:
    value = im_auroc[key]
    fpr, tpr = value._compute()
    im_auroc_ = value.compute()
    
    im_aur = {"value": im_auroc_, "fpr": fpr.detach().cpu(), "tpr": tpr.detach().cpu(), "label": f"{key}: {im_auroc_.detach().cpu():0.2f}"}
    added = False
    
    if len(im_auroc_sorted) > 0:
        for i in range(len(im_auroc_sorted)):
            if im_auroc_sorted[i]["value"] >= im_auroc_:
                if len(im_auroc_sorted) > i + 1:
                    if im_auroc_sorted[i+1]["value"] <= im_auroc_:
                        im_auroc_sorted.insert(i+1, im_aur)
                        added = True
                    else:
                        im_auroc_sorted.insert(i, im_aur) 
                        added = True
                else:
                    im_auroc_sorted.insert(i, im_aur) 
                    added = True
                break 
        if not added:
            im_auroc_sorted.append(im_aur)
               
    else:
        im_auroc_sorted.append(im_aur)

        

for entry in im_auroc_sorted:
    im_auroc_axs.plot(
        entry["fpr"],
        entry["tpr"],
        label=entry["label"],
        figure=im_auroc_fig,
        lw=2
    )
   
im_auroc_axs.set_xlim(xlim)
im_auroc_axs.set_ylim(ylim)
im_auroc_axs.set_xlabel("False Positive Rate")
im_auroc_axs.set_ylabel("True Positive Rate")
im_auroc_axs.legend(loc="lower right", fontsize="x-small")  
im_auroc_axs.set_title(f"Image Auroc scores for \n{category}/{error_type}")

plt.savefig(f"figures/image_auroc/{category}/{error_type}/combined.png")


pix_auroc_fig, pix_auroc_axs = plt.subplots()    

pix_auroc_sorted = []

for key in pix_auroc:
    value = pix_auroc[key]
    fpr, tpr = value._compute()
    pix_auroc_ = value.compute()
    
    pix_aur = {"value": pix_auroc_, "fpr": fpr.detach().cpu(), "tpr": tpr.detach().cpu(), "label": f"{key}: {pix_auroc_.detach().cpu():0.2f}"}
    added = False
    if len(pix_auroc_sorted) > 0:
        for i in range(len(pix_auroc_sorted)):
            if pix_auroc_sorted[i]["value"] >= pix_auroc_:
                if len(pix_auroc_sorted) > i + 1:
                    if pix_auroc_sorted[i+1]["value"] <= pix_auroc_:
                        pix_auroc_sorted.insert(i+1, pix_aur)
                        added = True
                    else:
                        pix_auroc_sorted.insert(i, pix_aur) 
                        added = True
                else:
                    pix_auroc_sorted.insert(i, pix_aur) 
                    added = True
                break 
        if not added:
            pix_auroc_sorted.append(pix_aur)
               
    else:
        pix_auroc_sorted.append(pix_aur)

        

for entry in pix_auroc_sorted:
    pix_auroc_axs.plot(
        entry["fpr"],
        entry["tpr"],
        label=entry["label"],
        figure=pix_auroc_fig,
        lw=2
    )
   
pix_auroc_axs.set_xlim(xlim)
pix_auroc_axs.set_ylim(ylim)
pix_auroc_axs.set_xlabel("False Positive Rate")
pix_auroc_axs.set_ylabel("True Positive Rate")
pix_auroc_axs.legend(loc="lower right", fontsize="x-small")  
pix_auroc_axs.set_title(f"Pixel-level Auroc scores for \n{category}/{error_type}")

plt.savefig(f"figures/pixel_auroc/{category}/{error_type}/combined.png")