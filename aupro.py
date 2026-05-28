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
prompt = x[-1].split(".")[0]
y = x[-3]
x = x[-2]

aupro = AUPRO(fields=["anomaly_map", "gt_mask"])

aupro.update(batch)

torch.save(aupro, f'aupro/{y}/{x}/{prompt}.pt')

aupro_res = aupro.compute()

results = {"category": x + "_" + y, "prompt": prompt.replace("_", " "), "aupro": aupro_res.item()}

with open("aupro.json", 'a') as f:
            json.dump(results, f)

aupro_fig, title = aupro.generate_figure()
aupro_title = "AUPRO \n Category: " + x + "_" + y + "  Prompt: " + prompt
aupro_fig.suptitle(aupro_title)
aupro_fig.set_layout_engine("tight")
aupro_fig.savefig(f'figures/aupro/{y}/{x}/{prompt}')
