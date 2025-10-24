from dataclasses import dataclass
from configs.base_config import BaseTrainingConfig


@dataclass
class RLConfig(BaseTrainingConfig):
    experiment_name: str = 'rl_stage2'
    
    # RL-specific parameters
    reward_model_path: str = ""
    value_model_path: str = ""
    response_length: int = 1300
    temperature: float = 0.5
    num_ppo_epochs: int = 3
    num_mini_batches: int = 1
    total_episodes: int = 1024
    kl_coef: float = 0.03
    local_rollout_forward_batch_size: int = 8
    per_device_train_batch_size: int = 2
    save_strategy: str = 'steps'
    save_step: int = 20
    save_total_limit: int = 1
    eval_strategy: str = 'steps'
    eval_steps: int = 20
    dataloader_num_workers: int = 4
    run_name: str = 'ppo_medical_o1'
    num_sample_generations: int = -1
    report_to: str = 'wandb'
    bf16: bool = True

