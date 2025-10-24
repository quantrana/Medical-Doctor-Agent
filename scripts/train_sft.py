#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import torch.distributed as dist
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tqdm import tqdm
import wandb
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    get_cosine_schedule_with_warmup
)
from torch.utils.data import DataLoader

from configs.sft_config import SFTConfig
from src.data.sft_dataset import SFTDataset
from src.training.metrics import SFTMetrics
from src.training.callbacks import CheckpointCallback
from src.utils.logger import setup_logger

logger = setup_logger('sft_training')


def train(args):
    #Main SFT training function.
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Setup W&B
    if accelerator.is_main_process:
        wandb.init(
            project=args.experiment_name,
            config=vars(args),
            dir=args.log_dir,
            mode="offline"
        )
    
    accelerator.print(f'Training arguments:\n{args}')
    
    # Configure DeepSpeed
    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = (
        args.train_bsz_per_gpu * dist.get_world_size() * accelerator.gradient_accumulation_steps
    )
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Setup optimizer with weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [parameter for name, parameter in model.named_parameters() if not any(nd in name for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [parameter for name, parameter in model.named_parameters() if any(nd in name for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # Create dataset and dataloader
    train_dataset = SFTDataset(args, tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_bsz_per_gpu,
        shuffle=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn
    )
    
    # Setup learning rate scheduler
    num_training_steps = int(len(train_dataloader) * args.n_epochs) // accelerator.gradient_accumulation_steps // dist.get_world_size()
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_rates * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    accelerator.print(
        f'Training setup: gradient_accumulation_steps={accelerator.gradient_accumulation_steps}, '
        f'data_path={args.data_path}, lr={args.learning_rate}, '
        f'num_training_steps={num_training_steps}'
    )
    
    # Prepare for distributed training
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    
    # Initialize tracking
    start_epoch = 0
    start_step = 0
    global_step = 0
    metric = SFTMetrics(device=torch.cuda.current_device())
    checkpoint_callback = CheckpointCallback(args.output_dir, args.max_ckpts)
    
    accelerator.print(accelerator.deepspeed_config)
    model.train()
    
    # Training loop
    for epoch in range(start_epoch, args.n_epochs):
        train_dataloader_iterator = (
            tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            if accelerator.is_main_process
            else enumerate(train_dataloader)
        )
        
        for batch_cnt, batch in train_dataloader_iterator:
            if epoch == start_epoch and batch_cnt < start_step:
                continue
            
            if batch_cnt == 1 and epoch == 0:
                torch.cuda.empty_cache()
            
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            output = model(input_ids=input_ids, labels=labels, return_dict=True, use_cache=False)
            loss = output.loss
            
            metric(output.logits, labels, loss)
            acc, train_loss = metric.get_metric()
            
            accelerator.backward(loss)
            
            if (global_step + 1) % accelerator.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            global_step += 1
            
            if accelerator.is_main_process:
                train_dataloader_iterator.set_postfix(
                    epoch=epoch,
                    current_step=batch_cnt,
                    total_step=len(train_dataloader),
                    skip=accelerator.optimizer_step_was_skipped,
                    loss=round(train_loss, 3),
                    acc=round(acc, 3),
                    length=len(input_ids[0]),
                    lr=lr_scheduler.get_last_lr()[0]
                )
            
            if global_step % 3 == 0 and accelerator.is_main_process:
                wandb.log({
                    'skip': int(accelerator.optimizer_step_was_skipped),
                    'loss': train_loss,
                    'acc': acc,
                    'lr': lr_scheduler.get_last_lr()[0]
                }, step=global_step)
        
        accelerator.wait_for_everyone()
        checkpoint_callback.save(
            accelerator, model, tokenizer, epoch, batch_cnt, global_step, args.model_path
        )


def parse_args():
    #Parse command line arguments.
    parser = argparse.ArgumentParser(description='SFT Stage 1 Training')
    
    # Experiment
    parser.add_argument('--experiment_name', type=str, default='sft_stage1')
    
    # Model
    parser.add_argument('--model_path', required=True, type=str)
    
    # Data
    parser.add_argument('--data_path', required=True, type=str)
    
    # Training
    parser.add_argument('--output_dir', default='./ckpts', type=str)
    parser.add_argument('--max_ckpts', default=2, type=int)
    parser.add_argument('--log_dir', default='./train_logs', type=str)
    parser.add_argument('--max_seq_len', default=4096, type=int)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', default=8, type=int)
    parser.add_argument('--train_bsz_per_gpu', default=2, type=int)
    parser.add_argument('--weight_decay', default=0.1, type=float)
    parser.add_argument('--learning_rate', default=5e-6, type=float)
    parser.add_argument('--warmup_rates', default=0.05, type=float)
    parser.add_argument('--n_epochs', default=3, type=int)
    
    # Other
    parser.add_argument('--seed', default=42, type=int)
    
    args = parser.parse_args()
    
    # Setup directories
    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    train(args)

    