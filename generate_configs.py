import json
import os

with open('prompts.json', 'r') as file:
    data = json.load(file)

f = open("run_inference_pipelines.bat", "w")

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

os.makedirs(f'configs', exist_ok=True)

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
            f.write(f'python experiment.py --config configs/{key}_{k}_{prompt}.yaml\n')
            os.makedirs(f'figure/aupro/{key}/{k}', exist_ok=True)
            os.makedirs(f'figure/image_auroc/{key}/{k}', exist_ok=True)
            os.makedirs(f'figure/pixel_auroc/{key}/{k}', exist_ok=True)

