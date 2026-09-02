import glob
import torch
import sys
import os
from anomalib.metrics import AUROC, PRO, AUPRO
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd

data = []

folder = sys.argv[1] + "/*/*" + ".pt"
subdirectories = [os.path.basename(path) for path in glob.glob(f'{sys.argv[1]}/*')]
for dir in subdirectories:
        subs = [os.path.basename(path) for path in glob.glob(f'{sys.argv[1]}/{dir}/*')]
        for sub in subs:
            cat_err = dir + "/" + sub
            for filename in sorted(glob.glob(f'{sys.argv[1]}/{dir}/{sub}/*.pt')):
                res = torch.load(filename, weights_only=False)
                im_aur = res["im_auroc_res"].item()
                pix_aur = res["pix_auroc_res"].item()
                aupro = res["aupro_res"].item()
                del res
                
                x = filename.split('\\')
                x = x[-1].split('.')
                prompt = x[0]
                
                data.append([dir, sub, prompt, im_aur, pix_aur, aupro])
                    

df = pd.DataFrame(data=data, columns=["category", "error type", "prompt", "image AUROC score", "pixel AUROC score", "AUPRO score"])
print(df)
torch.save(df, "figures/overall_df.pt")

