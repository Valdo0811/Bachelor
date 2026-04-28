import torch
from PIL import Image
from torchvision import transforms
from sam3.model.sam3_image_processor import Sam3Processor
from anomalib.pipelines.components import Job

class InferenceJob(Job):
    name = "infer"

    def __init__(self, processor: Sam3Processor, image: str, threshold: float, prompt: str, ground_truth: str):
        self.processor = processor
        self.image = image
        self.threshold = threshold
        self.prompt = prompt
        self.ground_truth = ground_truth

    def run(self, task_id: int | None = None) -> dict:
        processor = self.processor
        im = Image.open(self.image)
        gt = Image.open(self.ground_truth)
        inference_state = processor.set_image(im)
        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_confidence_threshold(state=inference_state, threshold=self.threshold)
        inference_state = processor.set_text_prompt(state=inference_state, prompt=self.prompt)
        masks, boxes, scores, maps = inference_state["masks"].to(torch.float), inference_state["boxes"], inference_state["scores"], inference_state["masks_logits"]
        
        transform = transforms.ToTensor()
        im = transform(im).to(device="cuda")
        gt = transform(gt).to(device="cuda")
        res = {"masks": masks, "scores": scores, "anomaly_maps": maps, "image": im, "ground_truth": gt, "image_path": self.image, "gt_path": self.ground_truth, "prompt": self.prompt, "threshold": self.threshold}
        print(res)
        return res
    
    @staticmethod
    def collect(results: list[dict]) -> dict:
        """Collect all individual runs into a dict of lists."""
        result_dict = {}
        
        for result in results:
            prompt = result["prompt"]
            if prompt not in result_dict.keys():
                result_dict[prompt] = []
            result_dict[prompt].append(result)
        
        return result_dict

    @staticmethod
    def save(results: list[dict]) -> None:
        return