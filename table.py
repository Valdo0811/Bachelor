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
#prompts = {}
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
            prompts = ""
            for filename in sorted(glob.glob(f'{sys.argv[1]}/{dir}/{sub}/*.pt')):
                res = torch.load(filename, weights_only=False)
                val = res["im_auroc_res"]
                del res
                
                x = filename.split('\\')
                x = x[-1].split('.')
                prompt = x[0]
                
                if prompt not in ["damage", "damaged", "broken", "error"]:
                    if prompts == "":
                        prompts = prompt
                    else:
                        prompts = prompts + f', {prompt}'
                
                nr_of_prompts_cat = nr_of_prompts_cat + 1
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
                
            row.extend([nr_of_prompts_err_type, f'{average_err_type:0.2f}', f'{err_type_best:0.2f}', err_type_best_prompt, prompts])
            rows.append(row)
        if nr_of_prompts_cat == 0:
            average_cat = 0.5
        else:
            average_cat = sum_cat / nr_of_prompts_cat
        
        cat_row.extend([nr_of_prompts_cat, f'{average_cat:0.2f}', f'{cat_best:0.2f}', cat_best_prompt, ""])
        rows.insert(rows_index, cat_row)
        
print(rows)

torch.save(rows, "figures/table_new.pt")

headers = ["category", "error type", "nr. of prompts", "average image auroc", "best image auroc", "best prompt", "specific prompts", "nr. of faulty images", "nr. of good images"]
#rows = torch.load("figures/table.pt", weights_only=False)
cat = ""
for row in rows:
    is_cat = False
    if row[0] != "":
        cat = row[0]
    path = "dataset/mvtec_anomaly_detection/" + cat + "/test/" + row[1]
    if row[1] == "":
        is_cat = True
        path = "dataset/mvtec_anomaly_detection/" + cat + "/test/*"
    images = len(glob.glob(f'{path}/*.png'))
    good_images = len(glob.glob("dataset/mvtec_anomaly_detection/" + cat + "/test/good/*.png"))
    if is_cat:
        images = images - good_images
    row.append(images)
    row.append(good_images)
df = pd.DataFrame(rows, columns=headers)
df = df[["category", "error type", "nr. of prompts", "specific prompts", "nr. of faulty images", "nr. of good images"]]
print(df.to_latex(index=False,
                  float_format="{:.2f}".format,
                  ))

#dfi.export(df, "figures/table_new.png")





