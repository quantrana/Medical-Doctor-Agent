#!/usr/bin/env python3
"""
GRPO Training for Medical Reasoning

This script implements GRPO (Group Relative Policy Optimization) using TRL library
with medical-specific reward functions.

Theoretical Foundation:
    GRPO uses group-relative advantages for policy optimization without requiring
    a value function. For each prompt, multiple completions are sampled and ranked
    by their rewards, eliminating the need for critic networks used in PPO.

Usage:
    accelerate launch scripts/train_grpo_simple.py \
        --model_name_or_path MODEL_PATH \
        --reward_model_path REWARD_MODEL_PATH \
        --dataset_name DATA_PATH \
        --output_dir OUTPUT_DIR \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --learning_rate 5e-7
"""

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
from src.data.rl_dataset import RLDataset


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
    
    train_dataset = RLDataset(data[eval_num:], tokenizer, max_length=512)
    eval_dataset = RLDataset(data[:eval_num], tokenizer, max_length=512)
    
    def compute_reward(completions, **kwargs):
        """
        Medical-specific reward function combining format and accuracy.
        
        Following Sutton & Barto Chapter 3, the reward signal R(s,a) encodes
        the goal of the learning agent. Here we decompose it into:
            R_total = 0.3 * R_format + 0.7 * R_accuracy
        
        where R_format ensures structural correctness and R_accuracy measures
        medical correctness via a learned verifier.
        """
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
    
    trainer = GRPOTrainer(
        model=policy,
        reward_funcs=[compute_reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    trainer.save_model(training_args.output_dir)
    
    print("="*80)
    print("Training Complete")
    print("="*80)


if __name__ == "__main__":
    main()

