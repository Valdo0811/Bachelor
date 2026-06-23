from EvaluationJobGenerator import EvaluationJobGenerator
from InferenceJobGenerator import InferenceJobGenerator
from anomalib.pipelines.components import Pipeline, Runner
from anomalib.pipelines.components.runners import SerialRunner


class ExperimentPipeline(Pipeline):

    def _setup_runners(self, args: dict) -> list[Runner]:
        return [
            SerialRunner(InferenceJobGenerator()),
            SerialRunner(EvaluationJobGenerator()),
        ]