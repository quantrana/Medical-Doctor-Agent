#!/usr/bin/env python3
import os
import sys
import json
import random
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import warnings
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser
)

from trl import ModelConfig, ScriptArguments
from ppo_utils.ppo_config_medo1 import PPOConfig
from ppo_utils.ppo_trainer_medo1 import PPOTrainer
from src.data.rl_dataset import RLDataset

# Environment setup
os.environ["WANDB_MODE"] = "offline"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    
    # Setup output directory
    output_dir = training_args.output_dir
    run_name = training_args.run_name
    if run_name not in output_dir:
        output_dir = os.path.join(output_dir, run_name)
        training_args.output_dir = output_dir
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    
    # Load reward model
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        attn_implementation="flash_attention_2",
        num_labels=2
    )
    
    # Load value model
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.value_model_path,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation="flash_attention_2",
        num_labels=1
    )
    
    # Load policy models
    ref_policy = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        attn_implementation="flash_attention_2"
    )
    policy = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        attn_implementation="flash_attention_2"
    )
    
    # Load reward tokenizer
    reward_tokenizer = AutoTokenizer.from_pretrained(training_args.reward_model_path)
    
    # Setup pad token
    if '<|eot_id|>' in tokenizer.vocab:
        assert '<|end_of_text|>' in tokenizer.vocab
        tokenizer.pad_token = '<|end_of_text|>'
        tokenizer.pad_token_id = tokenizer.encode('<|end_of_text|>', add_special_tokens=False)[0]
    assert tokenizer.pad_token_id != tokenizer.eos_token_id
    
    training_args.stop_token_id = tokenizer.eos_token_id
    
    # Load and split dataset
    eval_ratio = 0.1
    eval_max_num = 200
    
    with open(script_args.dataset_name) as f:
        data = json.load(f)
    
    random.shuffle(data)
    eval_num = min(int(len(data) * eval_ratio), eval_max_num)
    
    train_dataset = RLDataset(data[eval_num:], tokenizer, debug=1)
    eval_dataset = RLDataset(data[:eval_num], tokenizer)
    
    # Create PPO trainer
    trainer = PPOTrainer(
        config=training_args,
        processing_class=tokenizer,
        reward_processing_class=reward_tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=train_dataset.collate_fn
    )
    
    # Train
    trainer.train()

