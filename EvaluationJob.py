import torch
import json
import matplotlib as plt
import pandas as pd
from anomalib.metrics import AUROC, AUPRO
from anomalib.data.dataclasses.torch import ImageBatch
from anomalib.pipelines.components import Job

class EvaluationJob(Job):
    name = "evaluate"

    def __init__(self, results: list, prompt: str):
        self.results = results
        self.prompt = prompt

    def run(self, task_id: int | None = None) -> dict:
        
        images = []
        masks = []
        image_paths = []
        gt_masks = []
        gt_mask_paths = []
        anomaly_maps = []
        pred_labels = []
        scores = []
        gt_labels = []
        
        for entry in self.results:
            img = entry["image"]
            images.append(img)
            gt_masks.append(entry["ground_truth"])
            gt_mask_paths.append(entry["gt_path"])
            image_paths.append(entry["image_path"])
            gt_labels.append(entry["gt_label"])
            
            mask = entry["masks"]
            map = entry["anomaly_maps"]
            score = entry["scores"]
            
            if(mask.size(dim=0) > 1):
                scores.append(torch.max(score, dim=0).values)
                masks.append(torch.max(mask.to(torch.float), dim=0).values)
                anomaly_maps.append(torch.max(map, dim=0).values)
                pred_labels.append(torch.tensor(1, dtype=int).to(device="cuda"))
                
            elif (mask.size(dim=0) == 1):
                scores.append(score.squeeze(0))
                masks.append(mask.squeeze(0))
                anomaly_maps.append(map.squeeze(0))
                pred_labels.append(torch.tensor(1, dtype=int).to(device="cuda"))
                
            else:
                scores.append(torch.tensor(0).to(device="cuda"))
                masks.append(torch.zeros(1,img.size()[1], img.size()[2]).to(device="cuda"))
                anomaly_maps.append(torch.zeros(1,img.size()[1], img.size()[2]).to(device="cuda"))
                pred_labels.append(torch.tensor(0, dtype=int).to(device="cuda"))
        
        
        x = image_paths[0].replace('\\', '/')   
        x = x.split('/')
        error_type,category = x[-2], x[-4]  
        
        fixed_prompt = self.prompt.replace(" ", "_")  
        
        batch = ImageBatch(
            image=torch.stack(images,dim=0).to(device="cuda"),
            gt_label=torch.stack(gt_labels,dim=0).to(device="cuda"),
            image_path=image_paths,
            pred_label=torch.stack(pred_labels, dim=0).to(device="cuda"),
            pred_mask=torch.stack(masks,dim=0).to(device="cuda"),
            pred_score=torch.stack(scores,dim=0).to(device="cuda"),
            gt_mask=torch.stack(gt_masks,dim=0).to(device="cuda"),
            mask_path=gt_mask_paths,
            anomaly_map=torch.stack(anomaly_maps, dim=0).to(device="cuda"),
        )
        
        #torch.save(batch, f'predictions/{y}/{x}/{fixed_prompt}.pt')
        
        del images
        del gt_labels
        del image_paths
        del pred_labels
        del masks
        del scores
        del gt_masks
        del gt_mask_paths
        del anomaly_maps
        
        
        pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel")
        
        image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image")
        
        
        aupro = AUPRO(fields=["anomaly_map", "gt_mask"])
        
        
        pixel_auroc.update(batch)
        image_auroc.update(batch)
        aupro.update(batch)
        
        del batch   
        
        pixel_auroc_res = pixel_auroc.compute()
        image_auroc_res = image_auroc.compute()
        
        print(pixel_auroc_res)
        print(image_auroc_res)
        aupro_res = aupro.compute()
        print(aupro_res)
        
        metrics = {"image_auroc": image_auroc, "pixel_auroc": pixel_auroc, "aupro": aupro}
        
        torch.save(metrics, f'metrics/{category}/{error_type}/{fixed_prompt}.pt')
        
        aupro_fig, title = aupro.generate_figure()
        
        im_aur_fig, title = image_auroc.generate_figure()
        
        pix_aur_fig, title = pixel_auroc.generate_figure()
        
        aupro_title = "AUPRO \n Category: " + error_type + "_" + category + "  Prompt: " + fixed_prompt
        aupro_fig.suptitle(aupro_title)
        aupro_fig.set_layout_engine("tight")
        aupro_fig.savefig(f'figures/aupro/{category}/{error_type}/{fixed_prompt}')
        
        im_aur_title = "Image AUROC \n Category: " + error_type + "_" + category + "  Prompt: " + fixed_prompt
        im_aur_fig.suptitle(im_aur_title)
        im_aur_fig.set_layout_engine("tight")
        im_aur_fig.savefig(f'figures/image_auroc/{category}/{error_type}/{fixed_prompt}')
        
        pix_aur_title = "Pixel AUROC \n Category: " + error_type + "_" + category + "  Prompt: " + fixed_prompt
        pix_aur_fig.suptitle(pix_aur_title)
        pix_aur_fig.set_layout_engine("tight")
        pix_aur_fig.savefig(f'figures/pixel_auroc/{category}/{error_type}/{fixed_prompt}')
        
        
        return {"category": category, "error_type": error_type, "prompt": self.prompt, "pixel_auroc": pixel_auroc_res.item(), "image_auroc": image_auroc_res.item(), "aupro": aupro_res.item()}
    
    @staticmethod
    def collect(results: list[dict]) -> list[dict]:
        output: dict = {}
        for key in results[0]:
            output[key] = []
        for result in results:
            for key, value in result.items():
                output[key].append(value)
        return pd.DataFrame(output)
    
    @staticmethod
    def save(results: pd.DataFrame) -> None:
        with open(file="results.csv", mode='a', newline='') as file:
            results.to_csv(file, index=False, header=False, )