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
rows = []


folder = sys.argv[1] + "/*/*" + ".pt"
subdirectories = [os.path.basename(path) for path in glob.glob(f'{sys.argv[1]}/*')]
for dir in subdirectories:
        rows_index = len(rows)
        cur_rows = []
        cat_row = [dir, ""]
        sum_cat = 0
        nr_of_prompts_cat = 0
        cat_best = 0
        cat_best_prompt = ""
        subs = [os.path.basename(path) for path in glob.glob(f'{sys.argv[1]}/{dir}/*')]
        cat_bests = {}
        for sub in subs:
            row = ["", sub]
            sum_err_type = 0
            nr_of_prompts_err_type = 0
            err_type_best = 0
            err_type_best_prompt = ""
            cat_err = dir + "/" + sub
            for filename in sorted(glob.glob(f'{sys.argv[1]}/{dir}/{sub}/*.pt')):
                res = torch.load(filename, weights_only=False)
                val = res["im_auroc_res"]
                del res
                
                x = filename.split('\\')
                x = x[-1].split('.')
                prompt = x[0]
                
                    
                nr_of_prompts_err_type = nr_of_prompts_err_type + 1
                sum_err_type = sum_err_type + val
                
                
                
                if val >= err_type_best:
                    err_type_best = val
                    err_type_best_prompt = prompt
                
                if val >= cat_best:
                    cat_best = val
                    cat_best_prompt = prompt
                
                if nr_of_prompts_err_type == 0:
                    average_err_type = 0.5
                else:
                    average_err_type = sum_err_type / nr_of_prompts_err_type
                
            row.extend([nr_of_prompts_err_type, f'{average_err_type:0.2f}', f'{err_type_best:0.2f}', err_type_best_prompt])
            rows.append(row)
        if nr_of_prompts_cat == 0:
            average_cat = 0.5
        else:
            average_cat = sum_cat / nr_of_prompts_cat
        
        cat_row.extend([nr_of_prompts_cat, f'{average_cat:0.2f}', f'{cat_best:0.2f}', cat_best_prompt])
        rows.insert(rows_index, cat_row)
        
print(rows)
headers = ["category", "error-type", "number of prompts", "average image auroc", "best image auroc", "best prompt"]
torch.save(rows, "figures/table.pt")
df = pd.DataFrame(rows, columns=headers)
dfi.export(df, "figures/table.png")





