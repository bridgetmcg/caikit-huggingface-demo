# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

# Third Party
from module_ids import MASK_FILLER
from runtime.data_model.results import (
    Text,
    MaskScore,
    MaskScoreResult
)
from runtime.hf_base import HFBase
from transformers import pipeline

# Local
from caikit.core import ModuleBase, TaskBase, module, task

DEFAULT_MODEL = "microsoft/codebert-base-mlm"
DEFAULT_MODEL_REVISION = "71b4196"  # To prevent extra downloads and surprises
TASK = "fill-mask"
TASK_NAME = TASK.replace("-", "_")


@task(
    required_parameters={"text_in": str},
    output_type=MaskScoreResult,
)
class MaskGenerationTask(TaskBase):
    pass


@module(
    id=MASK_FILLER, name=TASK_NAME, version="0.0.0", task=MaskGenerationTask
)
class MaskGeneration(HFBase, ModuleBase):
    def __init__(self, model_config_path) -> None:
        super().__init__()
        hf_model, hf_revision = self.read_config(
            model_config_path, DEFAULT_MODEL, DEFAULT_MODEL_REVISION
        )
        self.pipe = pipeline(task=TASK, model=hf_model, revision=hf_revision)

    def run(self, text_in: str) -> MaskScoreResult:  # pylint: disable=arguments-differ
        mask_filler = self.pipe(text_in, top_k=3)
        results = [
            MaskScore(score=o["score"], token=o["token"], token_str=o["token_str"], sequence=o["sequence"])
            for o in mask_filler
        ]
        return MaskScoreResult(results)

    @classmethod
    def load(cls, model_path):
        return cls(model_path)
