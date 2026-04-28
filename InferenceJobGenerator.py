from anomalib.pipelines.components import JobGenerator
from typing import Generator
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import numpy as np
import glob
from InferenceJob import InferenceJob

class InferenceJobGenerator(JobGenerator):

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: None = None,
    ) -> Generator[InferenceJob, None, None]:
        """Generate Jobs via random selection."""
        model = build_sam3_image_model()
        processor = Sam3Processor(model)
        
        folder_path = args["folder_path"] + "/*" + args["image_type"]
        gt_path = args["gt_path"] + "/*" + args["image_type"]

        image_paths = []
        
        for filename in sorted(glob.glob(folder_path)):
            image_paths.append(filename)
           
        gt_paths = []
        
        for filename in sorted(glob.glob(gt_path)):
            gt_paths.append(filename)
        
        min_threshold = args["min_threshold"]
        max_threshold = args["max_threshold"]
        threshold_increments = args["threshold_increments"]
        print("images: ")
        print(len(image_paths))
        print("gt masks: ")
        print(len(gt_paths))
        for i in range(len(image_paths)):
            for prompt in  args["prompts"]:      
                for threshold in np.arange(min_threshold, max_threshold, threshold_increments):
                    print("created job")
                    yield InferenceJob(processor=processor, image=image_paths[i], ground_truth=gt_paths[i], threshold=threshold, prompt=prompt)
            

    @property
    def job_class(self) -> type:
        return InferenceJob