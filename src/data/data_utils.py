import json
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoader:
    
    @staticmethod
    def load_json(file_path):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {file_path}: {e}")
        
        return data if isinstance(data, list) else []
    
    @staticmethod
    def filter_sft_data(data):

        required_fields = ['Question', 'Complex_CoT', 'Response']
        filtered = []
        
        for idx, item in enumerate(data):
            if DataLoader._validate_fields(item, required_fields):
                filtered.append(item)
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Skipping invalid SFT item at index {idx}")
        
        logger.info(f"SFT data filtering: {len(data)} → {len(filtered)} valid examples")
        return filtered
    
    @staticmethod
    def filter_rl_data(data):

        filtered = []
        
        for idx, item in enumerate(data):
            question = item.get('Open-ended Verifiable Question', '')
            answer = item.get('Ground-True Answer', '')
            
            if len(question) > 0 and len(answer) > 0:
                filtered.append({
                    'question': question,
                    'answer': answer
                })
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Skipping invalid RL item at index {idx}")
        
        logger.info(f"RL data filtering: {len(data)} → {len(filtered)} valid examples")
        return filtered
    
    @staticmethod
    def _validate_fields(item, required):
        #Check if item has all required fields with non-empty content.
        return all(
            field in item and 
            isinstance(item[field], str) and 
            len(item[field].strip()) > 0
            for field in required
        )

