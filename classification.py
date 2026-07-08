import glob
import torch
import sys
import os
from anomalib.metrics import AUROC, PRO, AUPRO
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd
import dataframe_image as dfi

from PIL import Image, ImageDraw, ImageFont
from IPython.display import display


averages = torch.load("figures/table_found.pt", weights_only=False)

good = []
mid = []
bad = []
cat = ""
for entry in averages:
    cur_cat = entry[0]
    cat_err_type = cur_cat
    if (cur_cat != cat) and (cur_cat != ""):
        cat = cur_cat
    else:
        cat_err_type = cat + "/" + entry[1]
    average = float(entry[3])
    
    
    if average < 0.7:
        bad.append(cat_err_type)
    elif average > 0.95:
        good.append(cat_err_type)
    else:
        mid.append(cat_err_type)

length = max(len(good), len(bad), len(mid))

data = []

for i in range(length):
    row = []
    if i >= len(bad):
        row.append("")
    else:
        row.append(bad[i])
      
    if i >= len(mid):
        row.append("")
    else:
        row.append(mid[i])  
        
    if i >= len(good):
        row.append("")
    else:
        row.append(good[i])
    
    data.append(row)
        
df = pd.DataFrame(data, columns=["bad score (< 0.70)", "decent score (0.70 - 0.95)", "good score (>0.95)"])
dfi.export(df, "figures/classification_found.png")
        
print(f"bad: {len(bad)}")
print(f"mid: {len(mid)}")
print(f"good: {len(good)}")
print(good)