import glob
import torch
import sys
import os
from anomalib.metrics import AUROC, PRO, AUPRO
import matplotlib.pyplot as plt
from collections import OrderedDict

#folder_path = sys.argv[1] + "/*/*/*" + ".pt"

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
#split_folder_path = folder_path.split('/')
#category = split_folder_path[-3]
bests = {}


folder = sys.argv[1] + "/*/*" + ".pt"
subdirectories = [os.path.basename(path) for path in glob.glob(f'{sys.argv[1]}/*')]
for dir in subdirectories:
        sum = 0
        nr_of_prompts = 0
        subs = [os.path.basename(path) for path in glob.glob(f'{sys.argv[1]}/{dir}/*')]
        cat_bests = {}
        for sub in subs:
            cat_err = dir + "/" + sub
            for filename in sorted(glob.glob(f'{sys.argv[1]}/{dir}/{sub}/*.pt')):
                res = torch.load(filename, weights_only=False)
                val = res["im_auroc_res"]
                del res
                
                x = filename.split('\\')
                x = x[-1].split('.')
                prompt = x[0]
                
                
                if val != 0.5:
                    nr_of_prompts = nr_of_prompts + 1
                    sum = sum + val
                   
                '''    
                nr_of_prompts = nr_of_prompts + 1
                sum = sum + val
                '''
                if cat_err in prompts:
            
                    if cat_bests[cat_err] <= val:
                        cat_bests[cat_err] = val
                        prompts[cat_err] = prompt
                    
                    '''
                    if im_auroc_res[cat_err] <= res["im_auroc_res"]:
                        im_auroc_res[cat_err] = res["im_auroc_res"]
                        im_auroc[cat_err] = res["im_auroc"]
                        im_auroc_prompts[cat_err] = prompt
                    
                    if pix_auroc_res[cat_err] <= res["pix_auroc_res"]:
                        pix_auroc_res[cat_err] = res["pix_auroc_res"]
                        pix_auroc[cat_err] = res["pix_auroc"]
                        pix_auroc_prompts[cat_err] = prompt
                        
                    '''
                    
                else:   
                    cat_bests[cat_err] = val
                    prompts[cat_err] = prompt
                    
                    '''
                    im_auroc_res[cat_err] = res["im_auroc_res"]
                    im_auroc[cat_err] = res["im_auroc"]
                    pix_auroc_res[cat_err] = res["pix_auroc_res"]
                    pix_auroc[cat_err] = res["pix_auroc"]
                    
                    im_auroc_prompts[cat_err] = prompt
                    pix_auroc_prompts[cat_err] = prompt
                    aupro_prompts[cat_err] = prompt
                    '''
        if nr_of_prompts == 0:
            average = 0.5
        else:
            average = sum / nr_of_prompts
        cat_bests["average"] = average
        bests[dir] = cat_bests



im_auroc_sorted = []

bests_sorted = []

for key in bests:
    categ = bests[key]
    error_types_sorted = []
    for error_type in categ:
        if error_type == "average":
            continue
        res = {"value": categ[error_type], "label": f"{error_type}: {categ[error_type]:0.2f}"}
        val = categ[error_type]
        added = False
        
        if len(error_types_sorted) > 0:
            for i in range(len(error_types_sorted)):
                if error_types_sorted[i]["value"] >= val:
                    if len(error_types_sorted) > i + 1:
                        if error_types_sorted[i+1]["value"] <= val:
                            error_types_sorted.insert(i+1, res)
                            added = True
                        else:
                            error_types_sorted.insert(i, res) 
                            added = True
                    else:
                        error_types_sorted.insert(i, res) 
                        added = True
                    break 
            if not added:
                error_types_sorted.append(res)
                
        else:
            error_types_sorted.append(res)
    
    average = bests[key]["average"]
    if len(error_types_sorted) > 3:
        error_types_sorted = error_types_sorted[-3:]
    print(len(error_types_sorted))
    im_aur = {"average": average, "values": error_types_sorted, "label": f"{key} Ø: {average:0.2f}", "class": key}
    added = False
    
    if len(im_auroc_sorted) > 0:
        for i in range(len(im_auroc_sorted)):
            if im_auroc_sorted[i]["average"] >= average:
                if len(im_auroc_sorted) > i + 1:
                    if im_auroc_sorted[i+1]["average"] <= average:
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
        

torch.save(im_auroc_sorted, f'figures/best_found.pt')


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
    
    for val in entry["values"]:
        p = ax.bar(
            val["label"],
            val["value"].item(),
            color=colors[counter],
            figure=im_auroc_fig,
            width=0.5
        )
        ax.bar_label(p, label_type="center", fmt="%.2f")
    
    p = ax.bar(
        entry["label"],
        entry["average"].item(),
        color=colors[counter],
        figure=im_auroc_fig,
        label=entry["class"],
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




