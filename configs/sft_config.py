from dataclasses import dataclass
from configs.base_config import BaseTrainingConfig


@dataclass
class SFTConfig(BaseTrainingConfig):
    experiment_name: str = 'sft_stage1'

