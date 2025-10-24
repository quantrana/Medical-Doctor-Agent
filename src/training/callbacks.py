import os
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CheckpointCallback:
    # Handles model checkpointing with automatic cleanup.
    
    def __init__(self, output_dir: str, max_checkpoints: int = 2):
        
        #Initialize checkpoint callback.
        self.output_dir = Path(output_dir)
        self.max_checkpoints = max_checkpoints
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, accelerator, model, tokenizer, epoch: int, step: int, global_step: int, model_path: str):
        #Save model checkpoint with cleanup.
        
        save_dir = self.output_dir / f"checkpoint-{epoch}-{global_step}"
        
        if accelerator.is_main_process:
            # Cleanup old checkpoints
            if self.max_checkpoints > 0:
                checkpoint_files = [
                    file for file in os.listdir(self.output_dir)
                    if file.startswith("checkpoint-")
                ]
                num_checkpoints = len(checkpoint_files)
                
                if num_checkpoints >= self.max_checkpoints:
                    checkpoint_files.sort(
                        key=lambda x: os.path.getctime(os.path.join(self.output_dir, x))
                    )
                    oldest_checkpoint = checkpoint_files[0]
                    shutil.rmtree(os.path.join(self.output_dir, oldest_checkpoint))
            
            # Create checkpoint directory
            os.makedirs(save_dir, exist_ok=True)
            output_dir = os.path.join(save_dir, 'tfmr')
            
            # Save model and tokenizer
            if accelerator.state.deepspeed_plugin.zero_stage != 3:
                model.save_pretrained(
                    output_dir,
                    state_dict=accelerator.get_state_dict(model)
                )
            
            tokenizer.save_pretrained(output_dir)
            
            # Copy additional files from original model
            copy_files = []
            for item in os.listdir(model_path):
                if os.path.exists(os.path.join(output_dir, item)):
                    continue
                if item.startswith("pytorch_model") and item.endswith(".bin"):
                    continue
                if item.endswith(".index.json") or item.endswith(".safetensors"):
                    continue
                
                s = os.path.join(model_path, item)
                if os.path.isfile(s):
                    shutil.copy(s, os.path.join(output_dir, item))
                    copy_files.append(item)
            
            print(f'Huggingface model saved in {output_dir}, copied files: {copy_files}')
        
        # Handle DeepSpeed ZeRO Stage 3
        if accelerator.state.deepspeed_plugin.zero_stage == 3:
            unwrap_model = accelerator.unwrap_model(model)
            unwrap_model.save_pretrained(
                os.path.join(save_dir, 'tfmr'),
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model)
            )
        
        # Save training state
        accelerator.wait_for_everyone()
        accelerator.save(
            {"epoch": epoch, "step": step, "global_step": global_step},
            os.path.join(save_dir, "training_state.pt")
        )
        
        accelerator.print(f'Checkpoint checkpoint-{epoch}-{global_step} saved')

