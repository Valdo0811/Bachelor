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
        model = build_sam3_image_model()
        processor = Sam3Processor(model)
        
        folder_path = args["folder_path"] + "/*" + ".png"
        gt_path = args["gt_path"] + "/*" + ".png"
        good_pictures = args["good_pictures"] + "/*" + ".png"

        image_paths = []
        
        for filename in sorted(glob.glob(folder_path)):
            image_paths.append(filename)
           
        gt_paths = []
        
        for filename in sorted(glob.glob(gt_path)):
            gt_paths.append(filename)
            
        good_pictures_paths = []
        
        for filename in sorted(glob.glob(good_pictures)):
            good_pictures_paths.append(filename)
        
        for i in range(len(image_paths)):
            for prompt in  args["prompts"]:      
                yield InferenceJob(processor=processor, image=image_paths[i], ground_truth=gt_paths[i], prompt=prompt)
        
        for i in range(len(good_pictures_paths)):
            for prompt in  args["prompts"]:     
                yield InferenceJob(processor=processor, image=good_pictures_paths[i], ground_truth="", prompt=prompt)
                
                
    @property
    def job_class(self) -> type:
        return InferenceJob