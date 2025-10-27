import os
import sys
import json
import random
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from dataclasses import dataclass, field
from typing import Optional

from trl import GRPOConfig, GRPOTrainer

from grpo_utils.reward_functions import format_reward, accuracy_reward, combined_reward
from src.data.grpo_dataset import GRPODataset


@dataclass
class ScriptArguments:
    """Arguments for model and dataset paths."""
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained policy model"}
    )
    dataset_name: str = field(
        metadata={"help": "Path to training dataset (JSON)"}
    )
    reward_model_path: str = field(
        default="FreedomIntelligence/medical_o1_verifier_3B",
        metadata={"help": "Path to reward/verifier model"}
    )


def main():
    parser = HfArgumentParser((ScriptArguments, GRPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    
    if '<|eot_id|>' in tokenizer.vocab:
        tokenizer.pad_token = '<|end_of_text|>'
        tokenizer.pad_token_id = tokenizer.encode('<|end_of_text|>', add_special_tokens=False)[0]
    
    policy = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16
    )
    
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        script_args.reward_model_path,
        num_labels=2,
        torch_dtype=torch.bfloat16
    )
    
    reward_tokenizer = AutoTokenizer.from_pretrained(script_args.reward_model_path)
    
    with open(script_args.dataset_name, 'r') as f:
        data = json.load(f)
    
    random.shuffle(data)
    eval_num = min(int(len(data) * 0.1), 200)
    
    train_dataset = GRPODataset(data[eval_num:], tokenizer)
    eval_dataset = GRPODataset(data[:eval_num], tokenizer)
    
    def compute_reward(completions, **kwargs):
       
        ground_truth = kwargs.get("ground_truth", None)
        
        if ground_truth is None:
            raise ValueError("Ground truth not provided")
        
        if isinstance(completions[0], torch.Tensor):
            completions_text = [tokenizer.decode(c, skip_special_tokens=True) for c in completions]
        else:
            completions_text = completions
        
        format_rewards = format_reward(completions_text)
        
        accuracy_rewards = accuracy_reward(
            completions=completions_text,
            ground_truth=ground_truth,
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            device=reward_model.device
        )
        
        total_rewards = combined_reward(
            format_rewards=format_rewards,
            accuracy_rewards=accuracy_rewards,
            format_weight=0.3,
            accuracy_weight=0.7
        )
        
        return total_rewards
    
    training_args.tokenizer = tokenizer
    
    trainer = GRPOTrainer(
        model=policy,
        reward_funcs=[compute_reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    trainer.train()
    
    trainer.save_model(training_args.output_dir)
    
    print("Training Complete")


if __name__ == "__main__":
    main()

