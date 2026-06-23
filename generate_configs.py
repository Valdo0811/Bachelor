import sys
import glob
import torch
import os
from anomalib.metrics import AUROC, PRO, AUPRO
import json

with open('prompts.json', 'r') as file:
    data = json.load(file)

f = open("bat_files/run_inference_pipelines.bat", "w")
metrics = open("bat_files/run_metrics.bat", "w")

base_infer_dict = {
        "folder_path": "",
        "gt_path": "",
        "image_type": ".png",
        "good_pictures": "",
        "prompts": []
        }

base_dict = {
    "infer": base_infer_dict,
    "evaluate":""
}

for key in data:
    category = data[key]
    for k in category:
        error_type = category[k] 
        for prompt in error_type:
            prompt = prompt.replace(" ", "_")
            config = open(f'configs/{key}_{k}_{prompt}.yaml', "w")
            base_infer_dict["folder_path"] = f'dataset/mvtec_anomaly_detection/{key}/test/{k}'
            base_infer_dict["gt_path"] = f'dataset/mvtec_anomaly_detection/{key}/ground_truth/{k}'
            base_infer_dict["good_pictures"] = f'dataset/mvtec_anomaly_detection/{key}/test/good'
            base_infer_dict["prompts"] = [f'{prompt}']
            base_dict["infer"] = base_infer_dict
            config.write(str(base_dict))
            metrics.write(f'python metrics.py predictions/{key}/{k}/{prompt}.pt\n')
            f.write(f'python experiment.py --config configs/{key}_{k}_{prompt}.yaml\n')

