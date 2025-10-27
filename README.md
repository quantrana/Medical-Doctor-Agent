# Doctor-Agent: Medical Reasoning with RL

A framework for training medical reasoning models using Reinforcement Learning (PPO and GRPO).

## Repository Structure

The repository is organized into the following directories:

```
Doctor-Agent/
├── configs/              Configuration files and parameter definitions
├── src/                  Core source code
│   ├── data/            Data loading and preprocessing modules
│   ├── training/        Training utilities including metrics and callbacks
│   └── utils/           General utility functions
├── scripts/             Training entry points for SFT and RL stages
├── evaluation/          Evaluation scripts and data
├── ppo_utils/           PPO-specific training utilities
├── grpo_utils/          GRPO reward functions
└── data/                Training data directory
```

## Requirements

The framework requires Python 3.8 or higher and the following core dependencies:

- PyTorch 2.5.1
- Transformers 4.46.2
- Accelerate 0.34.2
- DeepSpeed 0.15.4
- TRL 0.14.0 (for GRPO support)
- vLLM 0.6.4

A complete list of dependencies is provided in `requirements.txt`. Install all requirements using:

```bash
pip install -r requirements.txt
```

## Training Data

The framework requires two datasets for the complete training pipeline:

1. **SFT Training Data**: Medical question-answer pairs with complex reasoning chains for supervised fine-tuning
2. **RL Training Data**: Verifiable medical problems with ground-truth answers for reinforcement learning

These datasets should be placed in the `data/` directory. The original training data used in HuatuoGPT-o1 can be obtained from:

## Training Pipeline

### Stage 1: Supervised Fine-Tuning

The first stage performs supervised fine-tuning on medical conversation data. To train on an 8-GPU setup:

```bash
accelerate launch --config_file ./configs/deepspeed_zero3.yaml \
    --num_processes 8 \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard \
    scripts/train_sft.py \
    --model_path {model path} \
    --data_path {data path} \
    --max_seq_len 4096 \
    --learning_rate 5e-6 \
    --train_bsz_per_gpu 8 \
    --output_dir ./ckpts
```

**Key Parameters:**

- `model_path`: Path or identifier for the base model to fine-tune
- `data_path`: Path to the SFT training dataset
- `max_seq_len`: Maximum sequence length for training
- `learning_rate`: Learning rate for the optimizer
- `train_bsz_per_gpu`: Training batch size per GPU
- `output_dir`: Directory for saving checkpoints

Additional parameters can be configured through the command-line interface or by modifying the configuration files in `configs/`.

### Stage 2: Reinforcement Learning

The second stage applies reinforcement learning using a medical verifier as the reward model. Two algorithms are supported:

#### Option A: PPO (Proximal Policy Optimization)

The traditional approach using policy and value models. Example training command:

```bash
accelerate launch \
    --num_processes 8 \
    --num_machines 1 \
    --machine_rank 0 \
    --config_file ./configs/deepspeed_zero3.yaml \
    --deepspeed_multinode_launcher standard \
    scripts/train_rl.py \
    --model_name_or_path ./ckpts/sft_stage1/checkpoint-final/tfmr \
    --reward_model_path FreedomIntelligence/medical_o1_verifier_3B \
    --value_model_path Qwen/Qwen3-0.6B \
    --dataset_name data/medical_o1_verifiable_problem.json \
    --response_length 1300 \
    --temperature 0.5 \
    --local_rollout_forward_batch_size 8 \
    --num_ppo_epochs 3 \
    --num_mini_batches 1 \
    --total_episodes 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --bf16 True \
    --output_dir ./ckpts \
    --save_strategy steps \
    --save_step 16 \
    --kl_coef 0.03 \
    --learning_rate 5e-7 \
    --warmup_ratio 0.05 \
    --gradient_checkpointing True \
    --run_name ppo_medical_o1 \
    --num_sample_generations -1 \
```

**Key Parameters:**

- `model_name_or_path`: Path to the SFT checkpoint from Stage 1
- `reward_model_path`: Path to the medical verifier reward model
- `value_model_path`: Path to the value model for PPO
- `dataset_name`: Path to verifiable medical problems dataset
- `total_episodes`: Total number of PPO training episodes
- `kl_coef`: KL divergence coefficient for PPO

#### Option B: GRPO (Group Relative Policy Optimization)

GRPO is an alternative RL algorithm that eliminates the need for a value model by using group-relative advantages. This approach, used in DeepSeek-R1, is simpler and more memory efficient than PPO.

```bash
accelerate launch \
    --num_processes 8 \
    --config_file ./configs/deepspeed_zero3.yaml \
    scripts/train_grpo.py \
    --model_name_or_path ./ckpts/sft_stage1/checkpoint-final/tfmr \
    --reward_model_path FreedomIntelligence/medical_o1_verifier_3B \
    --dataset_name data/medical_o1_verifiable_problem.json \
    --num_generations 4 \
    --max_completion_length 1300 \
    --temperature 0.5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-7 \
    --kl_coef 0.03 \
    --output_dir ./ckpts/grpo_medical_o1 \
    --run_name grpo_medical_o1
```

**Key Differences from PPO**:
- No value model required (simpler architecture)
- `num_generations`: Number of responses per prompt (4-8 recommended)
- Multiple generations per prompt enable better exploration
- Approximately 20% less memory usage compared to PPO

## Evaluation

The framework includes an evaluation pipeline using SGLang for efficient inference. 

**Step 1**: Deploy the trained model using SGLang:

```bash
model_name="./ckpts/ppo_medical_o1/checkpoint-final"
port=28035
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path $model_name \
    --port $port \
    --mem-fraction-static 0.8 \
    --dp 1 \
    --tp 1 > evaluation/sglang.log 2>&1 &
```

**Step 2**: Run evaluation on the deployed model:

```bash
python evaluation/eval.py \
    --model_name $model_name \
    --eval_file evaluation/data/eval_data.json \
    --port $port
```

**Step 3**: Terminate the SGLang server after evaluation:

```bash
bash evaluation/kill_sglang_server.sh
```

## License

This project is licensed under the Apache License 2.0
