import torch
from typing import Dict, List, Any
import logging
from src.data.base_dataset import MedicalDataset
from src.data.data_utils import DataLoader

logger = logging.getLogger(__name__)


class SFTDataset(MedicalDataset):
    #Dataset for supervised fine-tuning on medical reasoning.
    
    def __init__(self, config, tokenizer):
        # Initialize SFT dataset.
        super().__init__(config, tokenizer)
    
        # Load and filter data
        raw_data = DataLoader.load_json(config.data_path)
        self.data = DataLoader.filter_sft_data(raw_data)
        
        if not self.data:
            raise ValueError("No valid SFT examples found after filtering")
    
    def setup_response(self, item):
        #Get response template
        response_template = "## Thinking\n\n{}\n\n## Final Response\n\n{}"
        template = '## Thinking\n\n{}\n\n## Final Response\n\n{}'
        return template.format(item['Complex_CoT'], item['Response'])
    
    def setup_prompt(self, item):
        #Create training prompt with proper masking.
        #Get question and response
        q = item['Question']
        a = self.setup_response(item)
        
        assert q is not None and a is not None, f'q:{q} a:{a}'
        
        # Create full conversation
        input_text = self.template.render(
            messages=[
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ],
            bos_token=self.tokenizer.bos_token,
            add_generation_prompt=False
        )
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        
        # Create query only (for masking)
        query = self.template.render(
            messages=[{"role": "user", "content": q}],
            bos_token=self.tokenizer.bos_token,
            add_generation_prompt=True
        )
        query_ids = self.tokenizer.encode(query, add_special_tokens=False)
        
        # Create labels (mask query, keep response)
        labels = [-100] * len(query_ids) + input_ids[len(query_ids):]
        assert len(labels) == len(input_ids)
        
        # Truncate to max length
        return {
            "input_ids": input_ids[-self.max_seq_len:],
            "labels": labels[-self.max_seq_len:]
        }
    
    def collate_fn(self, batch):
       #Collate batch with proper padding.
        
        # Process each item
        data = [self.setup_prompt(item) for item in batch]
        
        # Extract sequences
        input_ids = [item["input_ids"] for item in data]
        labels = [item["labels"] for item in data]
        
        # Determine padding length
        max_len = max(len(x) for x in input_ids)
        max_len = min(max_len, self.max_seq_len)
        
        # Pad sequences
        input_ids = [
            item[:max_len] + [self.tokenizer.eos_token_id] * (max_len - len(item))
            for item in input_ids
        ]
        labels = [
            item[:max_len] + [-100] * (max_len - len(item))
            for item in labels
        ]
        
        # Debug logging (first 3 batches)
        if self.debug_count < 3:
            print('input_ids', self.tokenizer.decode(input_ids[-1]))
            print('labels', self.tokenizer.decode([0 if x == -100 else x for x in labels[-1]]))
            self.debug_count += 1
        
        return {
            "input_ids": torch.LongTensor(input_ids),
            "labels": torch.LongTensor(labels)
        }

