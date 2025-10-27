import torch
from typing import Dict, List
from torch.utils.data import Dataset


class GRPODataset(Dataset):
    
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        message = [{"role": "user", "content": item['question']}]
        prompt = self.tokenizer.apply_chat_template(
            message, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        return {
            "prompt": prompt,
            "ground_truth": item['answer']
        }

