from anomalib.pipelines.components import JobGenerator
from typing import Generator
from EvaluationJob import EvaluationJob

class EvaluationJobGenerator(JobGenerator):

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_results: dict | None = None,
    ) -> Generator[EvaluationJob, None, None]:
        """Generate Jobs via random selection."""
        assert prev_stage_results is not None, "Previous stage result is required"
        for key in prev_stage_results:
            yield EvaluationJob(results=prev_stage_results[key], prompt=key, threshold=prev_stage_results[key][0]["threshold"])

    @property
    def job_class(self) -> type:
        return EvaluationJob