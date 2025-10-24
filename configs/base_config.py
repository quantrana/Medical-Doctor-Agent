import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseTrainingConfig:
    # Model configuration
    model_path: str = ""
    
    # Data configuration
    data_path: str = ""
    max_seq_len: int = 8192
    
    # Training hyperparameters
    learning_rate: float = 5e-6
    train_bsz_per_gpu: int = 2
    gradient_accumulation_steps: int = 8
    n_epochs: int = 3
    weight_decay: float = 0.1
    warmup_rates: float = 0.05
    
    # Output directories
    output_dir: str = './ckpts'
    log_dir: str = './train_logs'
    experiment_name: str = ''
    
    # Other settings
    seed: int = 42
    max_ckpts: int = 2
    gradient_checkpointing: bool = False
    
    def setup_directories(self):
        if self.experiment_name:
            self.output_dir = os.path.join(self.output_dir, self.experiment_name)
            self.log_dir = os.path.join(self.log_dir, self.experiment_name)
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)