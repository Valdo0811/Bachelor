import glob
import torch
import sys
import os
from anomalib.metrics import AUROC, PRO, AUPRO
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd
import dataframe_image as dfi
import math
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
import seaborn as sns


im_auroc_res = {}
rows = []

prompts = 0
indices = [0]
cat_errs = 0
folder = sys.argv[1] + "/*/*" + ".pt"
subdirectories = [os.path.basename(path) for path in glob.glob(f'{sys.argv[1]}/*')]
for dir in subdirectories:
        subs = [os.path.basename(path) for path in glob.glob(f'{sys.argv[1]}/{dir}/*')]
        for sub in subs:
            cat_err = dir + "/" + sub
            entries = []
            cat_errs = cat_errs + 1
            im_auroc_res[cat_err] = []
            for filename in sorted(glob.glob(f'{sys.argv[1]}/{dir}/{sub}/*.pt')):
                prompts = prompts + 1
                res = torch.load(filename, weights_only=False)
                rows.append([cat_err, res["im_auroc_res"].item()])
                im_auroc_res[cat_err].append(res["im_auroc_res"])
                del res
            #rows.append([cat_err, entries])
            if cat_errs%20 == 0:
                indices.append(prompts)
                print(prompts)            

            

torch.save(rows, "figures/boxplot.pt")


headers = ["Category/Error-Type", "Image Auroc"]
#rows = torch.load("figures/boxplot_new.pt", weights_only=False)

df = pd.DataFrame(rows, columns=headers)
#dfi.export(df, "figures/boxplot.png")
nr_of_subplots = 4

#indices = [0, 112, 239, 351]

fig, ax = plt.subplots(nr_of_subplots, 1, figsize=(11,11)) 

for i in range(0,nr_of_subplots):
    min = indices[i]
    plt.setp(ax[i].get_xticklabels(), rotation=45, ha='right', size="small")
    if i == nr_of_subplots -1:
        sns.boxplot(ax=ax[i], data=df.loc[min:], x="Category/Error-Type", y="Image Auroc")
        continue
    max = indices[i+1]-1
    sns.boxplot(ax=ax[i], data=df.loc[min:max], x="Category/Error-Type", y="Image Auroc")
    
fig.set_layout_engine("constrained")

plt.savefig("figures/boxplot.png")
