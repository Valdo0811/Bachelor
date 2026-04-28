import torch
import json
from anomalib.metrics import AUROC
from anomalib.data.dataclasses.torch import ImageBatch
from anomalib.pipelines.components import Job

class EvaluationJob(Job):
    name = "evaluate"

    def __init__(self, results: list, prompt: str, threshold: float):
        self.results = results
        self.prompt = prompt
        self.threshold = threshold

    def run(self, task_id: int | None = None) -> dict:
        
        images = []
        masks = []
        image_paths = []
        gt_masks = []
        gt_mask_paths = []
        anomaly_maps = []
        pred_labels = []
        
        for entry in self.results:
            images.append(entry["image"])
            gt_masks.append(entry["ground_truth"])
            gt_mask_paths.append(entry["gt_path"])
            image_paths.append(entry["image_path"])
            
            mask = entry["masks"]
            map = entry["anomaly_maps"]
            
            if(mask.size(dim=0) > 1):
                masks.append(torch.max(mask.to(torch.float), dim=0).values)
                anomaly_maps.append(torch.max(map, dim=0).values)
                pred_labels.append(torch.tensor(1, dtype=int).to(device="cuda"))
                
            elif (mask.size(dim=0) == 1):
                masks.append(mask)
                anomaly_maps.append(map.squeeze(0))
                pred_labels.append(torch.tensor(1, dtype=int).to(device="cuda"))
                
            else:
                masks.append(mask)
                anomaly_maps.append(torch.zeros(1,1024,1024).to(device="cuda"))
                pred_labels.append(torch.tensor(0, dtype=int).to(device="cuda"))
            
            
        batch = ImageBatch(
            image=torch.stack(images,dim=0).to(device="cuda"),
            gt_label=torch.ones(len(images), dtype=int).to(device="cuda"),
            image_path=image_paths,
            pred_label=torch.stack(pred_labels, dim=0).to(device="cuda"),
            #pred_mask=,
            #pred_score=scores,
            gt_mask=torch.stack(gt_masks,dim=0).to(device="cuda"),
            mask_path=gt_mask_paths,
            anomaly_map=torch.stack(anomaly_maps, dim=0).to(device="cuda")
        )
        
        
        auroc = AUROC(fields=["anomaly_map", "gt_mask"])
        for item in batch:
            auroc.update(item)
        result = auroc.compute()
        print(self.prompt)
        print(result)
        return {"prompt": self.prompt, "threshold":self.threshold, "pixel_auroc": result.item()}
    
    @staticmethod
    def collect(results: list[dict]) -> list[dict]:
        """Collect all individual runs into a dict of lists."""
        return results
    
    @staticmethod
    def save(results: list[dict]) -> None:
        with open("results.json", 'w') as f:
            json.dump(results, f)