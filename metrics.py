import sys
import glob
import torch
import json
from anomalib.data.dataclasses.torch import ImageBatch
from anomalib.metrics import AUROC, PRO, AUPRO

file = sys.argv[1]
    
prompts = []
preds = {}    
batch = torch.load(file, weights_only=False)   


x = file.split('/')
prompt = x[-1].split(".")[0].replace(' ', "_")
y = x[-3]
x = x[-2]

pix_auroc = AUROC(fields=["anomaly_map", "gt_mask"])

pix_auroc.update(batch)

im_auroc = AUROC(fields=["pred_score", "gt_label"])

im_auroc.update(batch)

aupro = AUPRO(fields=["anomaly_map", "gt_mask"])

aupro.update(batch)

pix_auroc_res = pix_auroc.compute()

im_auroc_res = im_auroc.compute()

aupro_res = aupro.compute()

res = {"pix_auroc_res": pix_auroc_res, "pix_auroc": pix_auroc, "im_auroc_res": im_auroc_res, "im_auroc": im_auroc, "aupro_res": aupro_res, "aupro": aupro}

torch.save(res, f'metrics/{y}/{x}/{prompt}.pt')



aupro_fig, title = aupro.generate_figure()
        
im_aur_fig, title = im_auroc.generate_figure()
        
pix_aur_fig, title = pix_auroc.generate_figure()
        
aupro_title = "AUPRO \n Category: " + y + "_" + x + "  Prompt: " + prompt
aupro_fig.suptitle(aupro_title)
aupro_fig.set_layout_engine("tight")
aupro_fig.savefig(f'figures/aupro/{y}/{x}/{prompt}')
        
im_aur_title = "Image AUROC \n Category: " + y + "_" + x + "  Prompt: " + prompt
im_aur_fig.suptitle(im_aur_title)
im_aur_fig.set_layout_engine("tight")
im_aur_fig.savefig(f'figures/image_auroc/{y}/{x}/{prompt}')
        
pix_aur_title = "Pixel AUROC \n Category: " + y + "_" + x + "  Prompt: " + prompt
pix_aur_fig.suptitle(pix_aur_title)
pix_aur_fig.set_layout_engine("tight")
pix_aur_fig.savefig(f'figures/pixel_auroc/{y}/{x}/{prompt}')
