import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from jinja2 import Template
import logging

logger = logging.getLogger(__name__)


class MedicalDataset(Dataset):
    
    # Llama3-Instruct chat template
    LLAMA3_TEMPLATE = (
        "{% set loop_messages = messages %}"
        "{% for message in loop_messages %}"
        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'"
        "+ message['content'] | trim + '<|eot_id|>' %}"
        "{% if loop.index0 == 0 %}"
        "{% set content = bos_token + content %}"
        "{% endif %}"
        "{{ content }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
        "{% endif %}"
    )
    
    def __init__(self, config, tokenizer: PreTrainedTokenizer):
        
        self.config = config
        self.tokenizer = tokenizer
        self.max_seq_len = config.max_seq_len
        self.data = []
        self.debug_count = 0
        
        # Setup chat template
        self.setup_template()
    
    def setup_template(self):
        
        # If training from base LLMs, use llama3-instruct as template
        if not self.tokenizer.chat_template:
            self.tokenizer.chat_template = self.LLAMA3_TEMPLATE
            logger.info("Applied Llama3-Instruct chat template")
        
        self.template = Template(self.tokenizer.chat_template)
    
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

