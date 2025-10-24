import torch
from typing import Dict, List
import logging
from src.data.base_dataset import MedicalDataset
from src.data.data_utils import DataLoader

logger = logging.getLogger(__name__)


class RLDataset(MedicalDataset):
    #Dataset for RL training on verifiable medical problems.
    
    def __init__(self, data, tokenizer, max_length=1000, debug=0):
        #Initialize RL dataset.
        class MinimalConfig:
            max_seq_len = max_length
        
        super().__init__(MinimalConfig(), tokenizer)
        self.max_length = max_length
        self.debug_count = debug
        
        # Filter data
        self.data = DataLoader.filter_rl_data(data)
        
        if not self.data:
            raise ValueError("No valid RL examples found after filtering")
    
    def setup_prompt(self, item):
        #Create prompt for RL training.
        message = [{"role": "user", "content": item['question']}]
        prompt = self.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        
        input_token = self.tokenizer(
            prompt,
            padding=False,
            truncation=False,
            add_special_tokens=False
        )
        
        item['input_ids'] = input_token["input_ids"]
        return item
    
    def collate_fn(self, batch):
        #Collate batch for RL training.
        
        data = [self.setup_prompt(item) for item in batch]
        input_ids = [item["input_ids"] for item in data]
        question = [item["question"] for item in data]
        answer = [item["answer"] for item in data]
        
        max_len = max(len(x) for x in input_ids)
        max_len = min(max_len, self.max_length)
        
        # Pad sequences (left padding for generation)
        input_ids = [
            [self.tokenizer.pad_token_id] * (max_len - len(item)) + item[:max_len]
            for item in input_ids
        ]
        
        if self.debug_count > 0:
            print('[input_ids]', self.tokenizer.decode(input_ids[-1]))
            print('[question]', question[-1])
            print('[answer]', answer[-1])
            self.debug_count -= 1
        
        return {
            "input_ids": torch.LongTensor(input_ids),
            "question": question,
            "answer": answer
        }

