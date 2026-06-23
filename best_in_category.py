import glob
import torch
import sys
from anomalib.metrics import AUROC, PRO, AUPRO
import matplotlib.pyplot as plt
from collections import OrderedDict

folder_path = sys.argv[1] + "/*/*" + ".pt"

pix_auroc_prompts = {}
im_auroc_prompts = {}
aupro_prompts = {}
im_auroc = {}
im_auroc_res = {}
pix_auroc = {}
pix_auroc_res = {}
aupro = {}
aupro_res = {}
split_folder_path = folder_path.split('/')
category = split_folder_path[-3]

for filename in sorted(glob.glob(folder_path)):
        x = filename.split('\\')
        error_type = x[-2]
        x = x[-1].split('.')
        prompt = x[0]
        
        res = torch.load(filename, weights_only=False)
        
        if error_type in im_auroc_prompts:
            '''
            if im_auroc_res[error_type] <= res["im_auroc_res"]:
                im_auroc_res[error_type] = res["im_auroc_res"]
                im_auroc[error_type] = res["im_auroc"]
                im_auroc_prompts[error_type] = prompt
            
            if pix_auroc_res[error_type] <= res["pix_auroc_res"]:
                pix_auroc_res[error_type] = res["pix_auroc_res"]
                pix_auroc[error_type] = res["pix_auroc"]
                pix_auroc_prompts[error_type] = prompt
            '''    
            if aupro_res[error_type] <= res["aupro_res"]:
                aupro_res[error_type] = res["aupro_res"]
                aupro[error_type] = res["aupro"]
                aupro_prompts[error_type] = prompt
        
        else:   
            '''
            im_auroc_res[error_type] = res["im_auroc_res"]
            im_auroc[error_type] = res["im_auroc"]
            pix_auroc_res[error_type] = res["pix_auroc_res"]
            pix_auroc[error_type] = res["pix_auroc"]
            '''
            aupro_res[error_type] = res["aupro_res"]
            aupro[error_type] = res["aupro"]
            
            im_auroc_prompts[error_type] = prompt
            pix_auroc_prompts[error_type] = prompt
            aupro_prompts[error_type] = prompt
        
        del res



xlim = (0.0, 1.0)
ylim = (0.0, 1.0)

'''
im_auroc_fig, im_auroc_axs = plt.subplots()    

im_auroc_sorted = []

for key in im_auroc:
    value = im_auroc[key]
    fpr, tpr = value._compute()
    im_auroc_ = im_auroc_res[key]
    
    im_aur = {"value": im_auroc_, "fpr": fpr.detach().cpu(), "tpr": tpr.detach().cpu(), "label": f"{key}/{im_auroc_prompts[key]}: {im_auroc_.detach().cpu():0.2f}"}
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
im_auroc_axs.set_title(f"Best Image Auroc scores for \n{category} per Error-Type")

plt.savefig(f"figures/image_auroc/{category}/best.png")

del im_auroc
del im_auroc_res
del im_auroc_fig
del im_auroc_axs    
del im_auroc_sorted

pix_auroc_fig, pix_auroc_axs = plt.subplots()    

pix_auroc_sorted = []

for key in pix_auroc:
    value = pix_auroc[key]
    fpr, tpr = value._compute()
    pix_auroc_ = pix_auroc_res[key]
    
    pix_aur = {"value": pix_auroc_, "fpr": fpr.detach().cpu(), "tpr": tpr.detach().cpu(), "label": f"{key}/{pix_auroc_prompts[key]}: {pix_auroc_.detach().cpu():0.2f}"}
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
pix_auroc_axs.set_title(f"Best Pixel-Level Auroc scores for \n{category} per Error-Type")

plt.savefig(f"figures/pixel_auroc/{category}/best.png")

del pix_auroc
del pix_auroc_res
del pix_auroc_fig
del pix_auroc_axs    
del pix_auroc_sorted


'''
aupro_fig, aupro_axs = plt.subplots()    

aupro_sorted = []

for key in aupro:
    value = aupro[key]
    fpr, tpr = value._compute()
    aupro_ = aupro_res[key]
    xlim = (0.0, float(value.fpr_limit.detach().cpu().item()))
    
    aup = {"value": aupro_, "fpr": fpr.detach().cpu(), "tpr": tpr.detach().cpu(), "label": f"{key}/{aupro_prompts[key]}: {aupro_.detach().cpu():0.2f}"}
    added = False
    if len(aupro_sorted) > 0:
        for i in range(len(aupro_sorted)):
            if aupro_sorted[i]["value"] >= aupro_:
                if len(aupro_sorted) > i + 1:
                    if aupro_sorted[i+1]["value"] <= aupro_:
                        aupro_sorted.insert(i+1, aup)
                        added = True
                    else:
                        aupro_sorted.insert(i, aup) 
                        added = True
                else:
                    aupro_sorted.insert(i, aup) 
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
aupro_axs.set_title(f"Best Aupro scores for \n{category} per Error-Type")

plt.savefig(f"figures/aupro/{category}/best.png")
