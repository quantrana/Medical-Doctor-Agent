import re
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def format_reward(completions: List[str], verbose: bool = False) -> torch.Tensor:

    rewards = []
    
    for idx, completion in enumerate(completions):
        # Check for required structural tags
        has_thinking = "## Thinking" in completion
        has_final = "## Final Response" in completion
        
        # Count occurrences (uniqueness constraint)
        thinking_count = completion.count("## Thinking")
        final_count = completion.count("## Final Response")
        
        # Evaluate structural correctness
        if has_thinking and has_final and thinking_count == 1 and final_count == 1:
            # Check temporal ordering (thinking must precede final response)
            thinking_pos = completion.find("## Thinking")
            final_pos = completion.find("## Final Response")
            
            if thinking_pos < final_pos:
                # Perfect structure: full reward
                reward = 1.0
            else:
                # Incorrect temporal order: partial reward
                reward = 0.3
        elif has_thinking or has_final:
            # Partial structure: minimal reward for attempted formatting
            reward = 0.2
        else:
            # No structure: zero reward
            reward = 0.0
        
        rewards.append(reward)
        
        if verbose and idx < 3:  # Log first 3 samples
            logger.info(f"Format Reward {idx}: {reward:.2f}")
            logger.info(f"  Has Thinking: {has_thinking}, Has Final: {has_final}")
            logger.info(f"  Snippet: {completion[:100]}...")
    
    return torch.tensor(rewards, dtype=torch.float32)


def accuracy_reward(completions: List[str], ground_truth: List[str], reward_model, reward_tokenizer,
                    device: str = "cuda", max_length: int = 4000, threshold: float = 0.4, verbose: bool = False) -> torch.Tensor:
    # Template for verifier input (from original PPO implementation)
    verifier_template = """<Model Response>
{}
</Model Response>

<Reference Answer>
{}
</Reference Answer>

Your task is to evaluate the model response by comparing it to the reference answer. If the model response is correct and aligns with the reference answer, output "True". If it is incorrect or fails to select the correct option (if options are provided), output "False"."""
    
    # Extract final answers using regex pattern matching
    # Pattern: Capture text after "## Final Response\n\n"
    pattern = r"## Final Response\n\n(.*)"
    extracted_answers = []
    valid_format_mask = []
    
    for completion in completions:
        match = re.search(pattern, completion, re.DOTALL)
        if match:
            # Valid format: extract answer
            extracted_answers.append(match.group(1).strip())
            valid_format_mask.append(True)
        else:
            # Invalid format: use fallback response
            extracted_answers.append("I do not know the answer.")
            valid_format_mask.append(False)
    
    # Construct verifier inputs
    verifier_inputs = [
        verifier_template.format(answer, gt)
        for answer, gt in zip(extracted_answers, ground_truth)
    ]
    
    # Tokenize inputs for verifier
    inputs = reward_tokenizer(
        verifier_inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False
    ).to(device)
    
    # Get predictions from verifier (no gradient computation)
    with torch.no_grad():
        logits = reward_model(**inputs).logits
        probabilities = F.softmax(logits, dim=-1)
        
        # Extract correctness probability (assuming label 1 = correct)
        correctness_probs = probabilities[:, 1]
    
    # Convert probabilities to discrete rewards
    # Reward Engineering: Binary rewards with exploration bonus
    rewards = []
    for i, (prob, valid_format) in enumerate(zip(correctness_probs, valid_format_mask)):
        if not valid_format:
            # Format prerequisite failed: zero reward
            reward = 0.0
        elif prob.item() > threshold:
            # High confidence correct: full reward
            reward = 1.0
        else:
            # Low confidence or incorrect: small reward for exploration
            reward = 0.1
        
        rewards.append(reward)
        
        if verbose and i < 3:
            logger.info(f"Accuracy Reward {i}: {reward:.2f} (prob={prob.item():.3f})")
    
    return torch.tensor(rewards, dtype=torch.float32, device=device)


def combined_reward(format_rewards: torch.Tensor, accuracy_rewards: torch.Tensor, format_weight: float = 0.3,
                    accuracy_weight: float = 0.7, verbose: bool = False) -> torch.Tensor:

    # Format component: independent contribution
    format_component = format_weight * format_rewards
    
    # Accuracy component: gated by format (prerequisite structure)
    # Element-wise multiplication ensures accuracy only counts if format is good
    accuracy_component = accuracy_weight * (format_rewards * accuracy_rewards)
    
    # Total reward: sum of components
    combined = format_component + accuracy_component
    
    if verbose:
        logger.info("Reward Statistics:")
        logger.info(f"  Format (mean): {format_rewards.mean():.3f} ± {format_rewards.std():.3f}")
        logger.info(f"  Accuracy (mean): {accuracy_rewards.mean():.3f} ± {accuracy_rewards.std():.3f}")
        logger.info(f"  Combined (mean): {combined.mean():.3f} ± {combined.std():.3f}")
        logger.info(f"  Format contribution: {format_component.mean():.3f}")
        logger.info(f"  Accuracy contribution: {accuracy_component.mean():.3f}")
    
    return combined


# Utility functions for reward analysis and debugging

def analyze_reward_distribution(
    rewards: torch.Tensor,
    name: str = "Reward"
) -> dict:
    """
    Analyze statistical properties of reward distribution.
    
    Args:
        rewards: Reward tensor
        name: Name of reward for logging
        
    Returns:
        Dictionary with statistical metrics
    """
    stats = {
        "mean": rewards.mean().item(),
        "std": rewards.std().item(),
        "min": rewards.min().item(),
        "max": rewards.max().item(),
        "median": rewards.median().item(),
        "q25": torch.quantile(rewards, 0.25).item(),
        "q75": torch.quantile(rewards, 0.75).item(),
    }
    
    logger.info(f"{name} Distribution:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value:.3f}")
    
    return stats


def validate_reward_bounds(
    rewards: torch.Tensor,
    min_val: float = 0.0,
    max_val: float = 1.0,
    name: str = "Reward"
) -> bool:

    if torch.any(rewards < min_val) or torch.any(rewards > max_val):
        invalid_count = ((rewards < min_val) | (rewards > max_val)).sum().item()
        raise ValueError(
            f"{name} contains {invalid_count} values outside [{min_val}, {max_val}]. "
            f"Range: [{rewards.min():.3f}, {rewards.max():.3f}]"
        )
    return True
