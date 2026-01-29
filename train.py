# -*- coding: utf-8 -*-
"""
Writing Quality Adversarial GRPO Training

This script adapts Gemma3-270M GRPO for an adversarial game to improve writing criticism:
- Model A (Saboteur): Introduces subtle writing quality issues (grammar errors, awkward phrasing, clarity problems)
- Model B (Detector/Critic): Identifies writing quality issues in texts
- Both models are fine-tuned using GRPO based on detection success/failure
- Goal: Improve general writing criticism capabilities through adversarial training
"""

import os
import re
import random
import json
from datetime import datetime

# Import setup (CUDA environment, libraries, GPU check)
# This runs CUDA setup and GPU availability checks
import setup

# Import configuration and model initialization
from config import (
    max_seq_length,
    lora_rank,
    base_model_name,
    sabotage_start,
    sabotage_end,
    detection_start,
    detection_end,
    system_prompt_A,
    system_prompt_B,
    load_or_initialize_models,
)

# Import data loading functions
from loaddata import load_writing_dataset, sabotage_writing

# Import libraries (setup.py already imported these, but we need them in this namespace)
import numpy as np
import pandas as pd
import torch
import gc
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer, SFTTrainer, SFTConfig
from vllm import SamplingParams
from transformers import TextStreamer

# Initialize models (will be overridden if loading from checkpoints)
# This is a placeholder - actual initialization happens in train_models() or adversarial_game_loop()
base_model = None
tokenizer = None
default_adapter_name = "default"
model_A = None
model_B = None
tokenizer_A = None
tokenizer_B = None

# ============================================================================
# Detection Functions (for Model B)
# ============================================================================

def extract_detections(response):
    """
    Extract detected issues from Model B's response.
    Returns list of detected issues.
    """
    detections = []
    
    # Look for marked sections
    if detection_start in response and detection_end in response:
        detection_text = response.split(detection_start)[1].split(detection_end)[0]
        detections.append(detection_text.strip())
    
    # Also look for common detection phrases related to writing quality
    detection_keywords = [
        "grammar", "grammatical", "typo", "spelling", "error",
        "unclear", "vague", "confusing", "awkward", "wordy",
        "redundant", "inconsistent", "illogical", "unsupported",
        "clarity", "coherence", "style", "phrasing", "problematic",
        "incorrect", "wrong", "poor", "needs improvement"
    ]
    
    sentences = response.split(".")
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in detection_keywords):
            detections.append(sentence.strip())
    
    return detections

def extract_sabotage_type(response):
    """
    Extract detected sabotage type from Model B's response.
    Returns the detected type (grammar, style, clarity, logic, coherence, or None).
    """
    response_lower = response.lower()
    
    # Type-specific keywords
    type_keywords = {
        "grammar": ["grammar", "grammatical", "typo", "typos", "spelling", "spell", "punctuation"],
        "style": ["style", "wordy", "wordiness", "redundant", "redundancy", "awkward", "phrasing"],
        "clarity": ["clarity", "unclear", "vague", "confusing", "confused", "ambiguous"],
        "logic": ["logic", "logical", "illogical", "inconsistent", "contradict", "unsupported", "claim"],
        "coherence": ["coherence", "coherent", "irrelevant", "tangent", "off-topic", "disconnected"]
    }
    
    # Count matches for each type
    type_scores = {}
    for sabotage_type, keywords in type_keywords.items():
        score = sum(1 for kw in keywords if kw in response_lower)
        if score > 0:
            type_scores[sabotage_type] = score
    
    # Return the type with highest score, or None if no matches
    if type_scores:
        return max(type_scores.items(), key=lambda x: x[1])[0]
    return None

#
# ============================================================================
# Reward Functions
# ============================================================================

def calculate_reward_A(sabotaged_report, original_changes, detections, sabotage_type=None, detected_type=None, **kwargs):
    """
    Reward function for Model A (Saboteur).
    A gets +1 if B fails to detect, -1 if B detects correctly (both type and changes).
    Partial credit for partial detection.
    """
    scores = []
    
    # Check if B detected anything
    if not detections or len(detections) == 0:
        # B failed to detect - A wins
        scores.append(1.0)
    else:
        # B detected something - check accuracy
        detection_text = " ".join(detections).lower()
        changes_text = " ".join(original_changes).lower()
        
        # Check if type matches (if both are provided)
        type_match = False
        if sabotage_type and detected_type:
            type_match = (sabotage_type.lower() == detected_type.lower())
        elif sabotage_type:
            # Try to infer from detection text
            type_match = sabotage_type.lower() in detection_text
        
        # Check if changes match
        changes_match = any(change.lower() in detection_text for change in original_changes) if original_changes else False
        
        # Reward calculation
        if type_match and changes_match:
            # B correctly identified both type and changes - A loses
            scores.append(-1.0)
        elif changes_match:
            # B identified changes but wrong/unspecified type - partial penalty
            scores.append(-0.5)
        elif type_match:
            # B identified type but not specific changes - small penalty
            scores.append(-0.3)
        else:
            # B detected something but not the actual type or changes - minimal penalty
            scores.append(0.0)
    
    return scores

def calculate_reward_B(sabotaged_report, original_changes, detections, sabotage_type=None, detected_type=None, **kwargs):
    """
    Reward function for Model B (Detector).
    B gets reward based on correctly identifying both sabotage type and specific changes.
    """
    scores = []
    
    if not detections or len(detections) == 0:
        # B failed to detect - B loses
        scores.append(-1.0)
    else:
        # B detected something - check accuracy
        detection_text = " ".join(detections).lower()
        changes_text = " ".join(original_changes).lower()
        
        # Check if type matches (if both are provided)
        type_match = False
        type_score = 0.0
        if sabotage_type and detected_type:
            type_match = (sabotage_type.lower() == detected_type.lower())
            if type_match:
                type_score = 0.4  # 40% of reward for correct type
        elif sabotage_type:
            # Try to infer from detection text
            type_match = sabotage_type.lower() in detection_text
            if type_match:
                type_score = 0.3  # Partial credit for inferred type
        
        # Check if changes match
        changes_match_count = 0
        if original_changes:
            changes_match_count = sum(1 for change in original_changes if change.lower() in detection_text)
        
        changes_score = 0.0
        if changes_match_count > 0:
            # Partial credit based on how many changes were detected
            changes_accuracy = changes_match_count / len(original_changes)
            changes_score = 0.6 * changes_accuracy  # 60% of reward for correct changes
        
        # Total reward
        total_score = type_score + changes_score
        
        if total_score > 0:
            # B detected something correctly
            scores.append(total_score)
        else:
            # B detected something but not correctly - penalize
            scores.append(-0.5)
    
    return scores

# ============================================================================
# GRPO Reward Functions (for TRL compatibility)
# ============================================================================

def reward_func_A(prompts, completions, **kwargs):
    """
    GRPO reward function for Model A (Saboteur).
    This is called by GRPOTrainer during training to compute rewards.
    
    The reward is computed adversarially:
    - Model A generates text with writing quality issues (in completions)
    - Model B tries to detect issues in the text
    - If B fails to detect: A gets +1.0 reward
    - If B detects: A gets -1.0 reward
    
    This reward is then used by GRPO to update Model A's weights via policy gradients.
    """
    scores = []
    
    # Extract original text from prompts
    original_text = ""
    for msg in prompts[0]:
        if msg["role"] == "user":
            # Extract text from user message
            content = msg["content"]
            if "Original text:" in content:
                original_text = content.split("Original text:")[1].split("\n\nCreate")[0].strip()
            break
    
    for completion in completions:
        sabotaged_text = completion[0]["content"]
        score = 0.0
        
        # Adversarial evaluation: Use Model B to detect issues
        # Set Model B to use adapter_B if available, otherwise default adapter
        # Note: During Model A training, adapter_B may not exist yet, so use default
        if hasattr(model_B, 'peft_config') and model_B.peft_config:
            if "adapter_B" in model_B.peft_config:
                model_B.set_adapter("adapter_B")
            else:
                # Fallback to default adapter (used during Model A training before adapter_B is added)
                default_adapter_B = list(model_B.peft_config.keys())[0]
                model_B.set_adapter(default_adapter_B)
        
        # Create prompt for Model B to detect issues
        messages_B = [
            {"role": "system", "content": system_prompt_B},
            {"role": "user", "content": f"Text to analyze:\n{sabotaged_text}\n\nIdentify any writing quality issues:"},
        ]
        text_B = tokenizer_B.apply_chat_template(messages_B, add_generation_prompt=True, tokenize=False)
        
        # Generate detection with Model B
        inputs_B = tokenizer_B(text_B, return_tensors="pt").to(model_B.device)
        with torch.no_grad():
            outputs_B = model_B.generate(
                **inputs_B,
                max_new_tokens=512,
                temperature=0.7,
                top_k=50,
                do_sample=True,
            )
        detection_response = tokenizer_B.decode(outputs_B[0][inputs_B['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Extract detections
        detections = extract_detections(detection_response)
        
        # Adversarial reward: A wins if B fails to detect, loses if B detects
        if detections and len(detections) > 0:
            # B detected something - A loses
            score = -1.0
        else:
            # B failed to detect - A wins
            score = 1.0
        
        # Additional heuristics for subtlety
        obvious_keywords = ["WRONG", "ERROR", "FALSE", "INCORRECT", "FAKE", "BAD WRITING"]
        if any(kw in sabotaged_text.upper() for kw in obvious_keywords):
            score -= 0.2  # Penalize obvious sabotage
        
        # Penalize if text is identical to original
        if original_text and sabotaged_text.strip() == original_text.strip():
            score -= 1.0
        
        scores.append(score)
    
    return scores

def reward_func_B(prompts, completions, **kwargs):
    """
    GRPO reward function for Model B (Detector).
    This is called by GRPOTrainer during training to compute rewards.
    
    The reward is computed adversarially:
    - Model B generates detection results (in completions)
    - We check if the detections are meaningful and correct
    - If B correctly identifies issues: B gets +1.0 reward
    - If B fails to identify issues: B gets -1.0 reward
    
    This reward is then used by GRPO to update Model B's weights via policy gradients.
    """
    scores = []
    
    # Extract sabotaged text from prompts
    sabotaged_text = ""
    for msg in prompts[0]:
        if msg["role"] == "user":
            content = msg["content"]
            if "Text to analyze:" in content:
                sabotaged_text = content.split("Text to analyze:")[1].split("\n\nIdentify")[0].strip()
            break
    
    for completion in completions:
        detection_response = completion[0]["content"]
        score = 0.0
        
        # Extract detections from Model B's response
        detections = extract_detections(detection_response)
        
        # Adversarial evaluation: Check if B correctly identified issues
        # We use a heuristic: if B found detections and they seem reasonable, reward it
        if detections and len(detections) > 0:
            # B detected something - check if it's reasonable
            # Reward based on detection quality
            score = 1.0
            
            # Additional quality checks
            detection_text = " ".join(detections).lower()
            detection_keywords = [
                "grammar", "grammatical", "typo", "spelling", "error",
                "unclear", "vague", "confusing", "awkward", "wordy",
                "redundant", "inconsistent", "illogical", "unsupported",
                "clarity", "coherence", "style", "phrasing"
            ]
            found_keywords = sum(1 for kw in detection_keywords if kw in detection_text)
            if found_keywords > 0:
                score += 0.2 * min(found_keywords, 3)  # Bonus for relevant keywords
            
            # Check if detection markers are present (good practice)
            if detection_start in detection_response and detection_end in detection_response:
                score += 0.3
        else:
            # B failed to detect anything - penalize
            score = -1.0
        
        # Penalize empty or very short responses
        if len(detection_response.strip()) < 20:
            score -= 0.5
        
        # If we have the sabotaged text, we could do more sophisticated checking
        # For now, we use the heuristic above
        
        scores.append(score)
    
    return scores

# ============================================================================
# Main Training Loop
# ============================================================================

def prepare_dataset_for_A(dataset):
    """Prepare dataset for Model A (Saboteur) training."""
    sabotage_types = ["grammar", "style", "clarity", "logic", "coherence"]
    
    def format_for_A(example):
        text = example.get("text", example.get("report", ""))
        if not text or len(text.strip()) < 50:
            return None
        
        # Randomly select a sabotage type
        sabotage_type = random.choice(sabotage_types)
        
        # Create type-specific instruction
        type_instructions = {
            "grammar": "Introduce grammatical errors, typos, or spelling mistakes",
            "style": "Make the writing wordy, redundant, or awkward",
            "clarity": "Add vague phrases or make parts unclear",
            "logic": "Add unsupported claims or logical inconsistencies",
            "coherence": "Add irrelevant sentences or break coherence"
        }
        
        instruction = type_instructions.get(sabotage_type, "Introduce subtle writing quality issues")
        
        return {
            "prompt": [
                {"role": "system", "content": system_prompt_A},
                {"role": "user", "content": f"Original text:\n{text}\n\nCreate a version with subtle writing quality issues. Specifically, {instruction.lower()}:"},
            ],
            "sabotage_type": sabotage_type,  # Ground truth: expected sabotage type
        }
    
    formatted = dataset.map(format_for_A, remove_columns=dataset.column_names)
    formatted = formatted.filter(lambda x: x["prompt"] is not None)
    return formatted

def prepare_dataset_for_B(dataset):
    """Prepare dataset for Model B (Detector/Critic) training."""
    sabotage_types = ["grammar", "style", "clarity", "logic", "coherence"]
    
    def format_for_B(example):
        text = example.get("text", example.get("report", ""))
        if not text or len(text.strip()) < 50:
            return None
        
        # Randomly select a sabotage type and create sabotaged version
        sabotage_type = random.choice(sabotage_types)
        sabotaged, changes = sabotage_writing(text, sabotage_type=sabotage_type)
        
        return {
            "prompt": [
                {"role": "system", "content": system_prompt_B},
                {"role": "user", "content": f"Text to analyze:\n{sabotaged}\n\nIdentify any writing quality issues:"},
            ],
            "changes": changes,  # Ground truth: actual changes made
            "sabotage_type": sabotage_type,  # Ground truth: type of sabotage applied
        }
    
    formatted = dataset.map(format_for_B, remove_columns=dataset.column_names)
    formatted = formatted.filter(lambda x: x["prompt"] is not None)
    return formatted

def train_models(load_model_A=False, load_model_B=False, 
                 model_A_path="lora_model_A", model_B_path="lora_model_B",
                 resume_training_A=False, resume_training_B=False):
    """
    Main training function for both models.
    
    Args:
        load_model_A: Load Model A from checkpoint before training
        load_model_B: Load Model B from checkpoint before training
        model_A_path: Path to Model A checkpoint
        model_B_path: Path to Model B checkpoint
        resume_training_A: Resume training Model A (requires load_model_A=True)
        resume_training_B: Resume training Model B (requires load_model_B=True)
    """
    global base_model, tokenizer, default_adapter_name, model_A, model_B, tokenizer_A, tokenizer_B
    
    # Initialize or load models
    base_model, tokenizer, default_adapter_name, model_A_loaded, model_B_loaded = load_or_initialize_models(
        load_model_A=load_model_A,
        load_model_B=load_model_B,
        model_A_path=model_A_path,
        model_B_path=model_B_path
    )
    
    # Set up model references
    model_A = base_model
    model_B = base_model
    tokenizer_A = tokenizer
    tokenizer_B = tokenizer
    
    print("Loading dataset...")
    dataset = load_writing_dataset()
    
    # Limit dataset size for initial testing
    if len(dataset) > 1000:
        dataset = dataset.select(range(1000))
    
    print(f"Dataset size: {len(dataset)}")
    
    # Track rewards during training
    training_rewards_A = []
    training_rewards_B = []
    
    # Prepare datasets
    print("Preparing datasets...")
    dataset_A = prepare_dataset_for_A(dataset)
    dataset_B = prepare_dataset_for_B(dataset)
    
    # Tokenize and filter by length
    print("Tokenizing datasets...")
    
    def tokenize_A(x):
        tokens = tokenizer_A.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)
        return {"tokens": tokens, "L": len(tokens)}
    
    def tokenize_B(x):
        tokens = tokenizer_B.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)
        return {"tokens": tokens, "L": len(tokens)}
    
    tokenized_A = dataset_A.map(tokenize_A, batched=False)
    tokenized_B = dataset_B.map(tokenize_B, batched=False)
    
    # Filter by length (keep 90th percentile)
    max_len_A = int(np.quantile([x["L"] for x in tokenized_A], 0.9))
    max_len_B = int(np.quantile([x["L"] for x in tokenized_B], 0.9))
    
    dataset_A = dataset_A.select([i for i, x in enumerate(tokenized_A) if x["L"] <= max_len_A])
    dataset_B = dataset_B.select([i for i, x in enumerate(tokenized_B) if x["L"] <= max_len_B])
    
    print(f"Filtered dataset A size: {len(dataset_A)}")
    print(f"Filtered dataset B size: {len(dataset_B)}")
    
    # GRPO Configuration
    max_prompt_length_A = max_len_A + 1
    max_completion_length_A = max_seq_length - max_prompt_length_A
    
    max_prompt_length_B = max_len_B + 1
    max_completion_length_B = max_seq_length - max_prompt_length_B
    
    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=3407,
        stop=[tokenizer_A.eos_token],
        include_stop_str_in_output=True,
        max_tokens=512,  # Limit generation length to save memory
    )
    
    # Clear any previous GRPO checkpoints to avoid adapter conflicts
    import shutil
    if os.path.exists("grpo_trainer_lora_model"):
        shutil.rmtree("grpo_trainer_lora_model")
        print("Cleared previous GRPO checkpoint directory")
    
    # Calculate steps per epoch for logging
    # Steps per epoch = dataset_size / (batch_size * gradient_accumulation_steps)
    batch_size = 1
    gradient_accumulation = 1
    steps_per_epoch_A = max(1, len(dataset_A) // (batch_size * gradient_accumulation))
    steps_per_epoch_B = max(1, len(dataset_B) // (batch_size * gradient_accumulation))
    
    # Train Model A
    print("\n" + "="*50)
    print("Training Model A (Saboteur)...")
    print(f"Dataset size: {len(dataset_A)} examples")
    print(f"Steps per epoch: {steps_per_epoch_A}")
    print("="*50)
    
    training_args_A = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=1.0,
        learning_rate=5e-6,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=max(1, steps_per_epoch_A // 10),  # Log ~10 times per epoch
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=2,  # Reduced from 4 to save memory (GRPO generates multiple completions)
        max_prompt_length=max_prompt_length_A,
        max_completion_length=max_completion_length_A,
        num_train_epochs=3,  # Train for 3 epochs
        max_steps=-1,  # Disable step-based training, use epochs instead
        save_steps=steps_per_epoch_A,  # Save once per epoch
        eval_strategy="no",  # No evaluation dataset
        logging_strategy="steps",  # Log at regular step intervals
        report_to="none",
        output_dir="outputs/model_A",
    )
    
    # Set active adapter to default adapter for Model A training
    default_adapter_name = list(model_A.peft_config.keys())[0] if hasattr(model_A, 'peft_config') and model_A.peft_config else "default"
    model_A.set_adapter(default_adapter_name)
    
    # Ensure adapter_B doesn't exist (shouldn't at this point, but check to be safe)
    if "adapter_B" in model_A.peft_config:
        print("WARNING: adapter_B found before Model A training. Removing it...")
        # Remove adapter_B if it somehow exists
        del model_A.peft_config["adapter_B"]
        for name, module in model_A.named_modules():
            if hasattr(module, 'lora_A') and "adapter_B" in module.lora_A:
                del module.lora_A["adapter_B"]
            if hasattr(module, 'lora_B') and "adapter_B" in module.lora_B:
                del module.lora_B["adapter_B"]
    
    # ========================================================================
    # MODEL A TRAINING: Update Model A (Saboteur) using GRPO
    # ========================================================================
    # GRPOTrainer uses reward_func_A to compute rewards for each generation.
    # During training, GRPO:
    #   1. Generates multiple completions for each prompt (num_generations=2)
    #   2. Calls reward_func_A(prompts, completions) to get rewards
    #   3. Computes policy gradients from rewards
    #   4. Updates Model A's LoRA weights via backpropagation
    #   5. Repeats for each batch/step
    
    # Create wrapper to track rewards
    def reward_func_A_tracked(prompts, completions, **kwargs):
        rewards = reward_func_A(prompts, completions, **kwargs)
        training_rewards_A.extend(rewards)  # Track rewards
        return rewards
    
    trainer_A = GRPOTrainer(
        model=model_A,
        processing_class=tokenizer_A,
        reward_funcs=[reward_func_A_tracked],  # Reward function with tracking
        args=training_args_A,
        train_dataset=dataset_A,
    )
    
    print("\nStarting training...")
    print("Loss will be logged during training. Check output for progress.\n")
    # MODEL UPDATE HAPPENS HERE: trainer_A.train() updates Model A's weights
    trainer_A.train()
    
    # Print final training metrics
    print("\n" + "="*50)
    print("Model A Training Complete!")
    print("="*50)
    if hasattr(trainer_A.state, 'log_history'):
        for log_entry in trainer_A.state.log_history:
            if 'loss' in log_entry:
                print(f"Step {log_entry.get('step', 'N/A')}: Loss = {log_entry['loss']:.4f}")
    
    # MODEL SAVING HAPPENS HERE: Save Model A's LoRA adapter weights
    model_A.save_pretrained("lora_model_A")
    
    # Clear GRPO checkpoint directory after Model A training
    if os.path.exists("grpo_trainer_lora_model"):
        shutil.rmtree("grpo_trainer_lora_model")
        print("Cleared GRPO checkpoint directory after Model A training")
    
    # Now add adapter_B for Model B training (after Model A is trained)
    # Check if adapter_B already exists (loaded from checkpoint)
    if "adapter_B" not in (model_B.peft_config.keys() if hasattr(model_B, 'peft_config') and model_B.peft_config else []):
        print("\nAdding adapter_B for Model B training...")
        from peft import LoraConfig
        lora_config_B = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank*2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model_B.add_adapter("adapter_B", lora_config_B)
        print("Adapter_B added successfully.")
    else:
        print("\nAdapter_B already exists (loaded from checkpoint).")
    
    # Train Model B
    print("\n" + "="*50)
    print("Training Model B (Detector)...")
    print(f"Dataset size: {len(dataset_B)} examples")
    print(f"Steps per epoch: {steps_per_epoch_B}")
    print("="*50)
    
    training_args_B = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=1.0,
        learning_rate=5e-6,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=max(1, steps_per_epoch_B // 10),  # Log ~10 times per epoch
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=2,  # Reduced from 4 to save memory (GRPO generates multiple completions)
        max_prompt_length=max_prompt_length_B,
        max_completion_length=max_completion_length_B,
        num_train_epochs=3,  # Train for 3 epochs
        max_steps=-1,  # Disable step-based training, use epochs instead
        save_steps=steps_per_epoch_B,  # Save once per epoch
        eval_strategy="no",  # No evaluation dataset
        logging_strategy="steps",  # Log at regular step intervals
        report_to="none",
        output_dir="outputs/model_B",
    )
    
    # Set active adapter to adapter_B for Model B training
    model_B.set_adapter("adapter_B")
    
    # GRPO trainer's load_lora expects a "default" adapter
    # So we need to rename adapter_B to "default" temporarily
    # First, save the default adapter config if it exists
    all_adapter_names = list(model_B.peft_config.keys()) if hasattr(model_B, 'peft_config') and model_B.peft_config else []
    original_default_name = None
    original_default_config = None
    
    # Find and save the original default adapter config
    for adapter_name in all_adapter_names:
        if adapter_name != "adapter_B":
            original_default_name = adapter_name
            from peft import LoraConfig
            old_config = model_B.peft_config[adapter_name]
            original_default_config = LoraConfig(
                r=old_config.r,
                lora_alpha=old_config.lora_alpha,
                target_modules=old_config.target_modules,
                lora_dropout=old_config.lora_dropout,
                bias=old_config.bias,
                task_type=old_config.task_type,
            )
            break
    
    # Remove the original default adapter
    if original_default_name:
        print(f"Removing original adapter '{original_default_name}' before renaming adapter_B to 'default'...")
        del model_B.peft_config[original_default_name]
        # Remove adapter weights from the model
        for name, module in model_B.named_modules():
            if hasattr(module, 'lora_A') and original_default_name in getattr(module, 'lora_A', {}):
                del module.lora_A[original_default_name]
            if hasattr(module, 'lora_B') and original_default_name in getattr(module, 'lora_B', {}):
                del module.lora_B[original_default_name]
    
    # Rename adapter_B to "default" so GRPO trainer can find it
    if "adapter_B" in model_B.peft_config:
        print("Renaming adapter_B to 'default' for GRPO trainer compatibility...")
        adapter_B_config = model_B.peft_config["adapter_B"]
        # Rename in peft_config
        del model_B.peft_config["adapter_B"]
        model_B.peft_config["default"] = adapter_B_config
        
        # Rename adapter weights in all modules
        for name, module in model_B.named_modules():
            if hasattr(module, 'lora_A') and "adapter_B" in getattr(module, 'lora_A', {}):
                module.lora_A["default"] = module.lora_A.pop("adapter_B")
            if hasattr(module, 'lora_B') and "adapter_B" in getattr(module, 'lora_B', {}):
                module.lora_B["default"] = module.lora_B.pop("adapter_B")
        
        # Update active adapter
        model_B.set_adapter("default")
        print("Adapter renamed successfully. Active adapter is now 'default'.")
    
    # ========================================================================
    # MODEL B TRAINING: Update Model B (Detector) using GRPO
    # ========================================================================
    # GRPOTrainer uses reward_func_B to compute rewards for each generation.
    # During training, GRPO:
    #   1. Generates multiple completions for each prompt (num_generations=2)
    #   2. Calls reward_func_B(prompts, completions) to get rewards
    #   3. Computes policy gradients from rewards
    #   4. Updates Model B's LoRA weights (adapter_B) via backpropagation
    #   5. Repeats for each batch/step
    
    # Create wrapper to track rewards
    def reward_func_B_tracked(prompts, completions, **kwargs):
        rewards = reward_func_B(prompts, completions, **kwargs)
        training_rewards_B.extend(rewards)  # Track rewards
        return rewards
    
    trainer_B = GRPOTrainer(
        model=model_B,
        processing_class=tokenizer_B,
        reward_funcs=[reward_func_B_tracked],  # Reward function with tracking
        args=training_args_B,
        train_dataset=dataset_B,
    )
    
    print("\nStarting training...")
    print("Loss will be logged during training. Check output for progress.\n")
    # MODEL UPDATE HAPPENS HERE: trainer_B.train() updates Model B's weights
    trainer_B.train()
    
    # Print final training metrics
    print("\n" + "="*50)
    print("Model B Training Complete!")
    print("="*50)
    if hasattr(trainer_B.state, 'log_history'):
        for log_entry in trainer_B.state.log_history:
            if 'loss' in log_entry:
                print(f"Step {log_entry.get('step', 'N/A')}: Loss = {log_entry['loss']:.4f}")
    
    # MODEL SAVING HAPPENS HERE: Save Model B's LoRA adapter weights
    model_B.save_pretrained("lora_model_B")
    
    # Rename "default" back to "adapter_B" after training
    if "default" in model_B.peft_config:
        print("Renaming 'default' back to 'adapter_B' after training...")
        default_config = model_B.peft_config["default"]
        del model_B.peft_config["default"]
        model_B.peft_config["adapter_B"] = default_config
        
        # Rename adapter weights back
        for name, module in model_B.named_modules():
            if hasattr(module, 'lora_A') and "default" in getattr(module, 'lora_A', {}):
                module.lora_A["adapter_B"] = module.lora_A.pop("default")
            if hasattr(module, 'lora_B') and "default" in getattr(module, 'lora_B', {}):
                module.lora_B["adapter_B"] = module.lora_B.pop("default")
        
        model_B.set_adapter("adapter_B")
    
    # Re-add original default adapter after training (optional, for future use)
    if original_default_config is not None and original_default_name:
        print(f"Re-adding original adapter '{original_default_name}' after Model B training...")
        model_B.add_adapter(original_default_name, original_default_config)
    
    print("\nTraining complete!")
    print("Model A saved to: lora_model_A")
    print("Model B saved to: lora_model_B")
    

# ============================================================================
# Adversarial Game Loop (for iterative improvement)
# ============================================================================

def adversarial_game_loop(num_iterations=5, batch_size=8, output_dir="game_results", 
                          update_model_B=True, update_model_A=False, 
                          training_frequency=1, num_training_steps=1,
                          load_model_A=False, load_model_B=False,
                          model_A_path="lora_model_A", model_B_path="lora_model_B"):
    """
    Run adversarial game loop where A and B compete iteratively.
    Now processes examples in batches for efficiency and can update models using GRPO.
    
    Args:
        num_iterations: Number of game iterations (batches)
        batch_size: Number of examples to process per batch
        output_dir: Directory to save results (default: "game_results")
        update_model_B: Whether to update Model B (Detector) using GRPO (default: True)
        update_model_A: Whether to update Model A (Saboteur) using GRPO (default: False)
        training_frequency: Update models every N iterations (default: 1 = every iteration)
        num_training_steps: Number of GRPO training steps per update (default: 1)
        load_model_A: Load Model A from checkpoint before game loop
        load_model_B: Load Model B from checkpoint before game loop
        model_A_path: Path to Model A checkpoint
        model_B_path: Path to Model B checkpoint
    """
    global base_model, tokenizer, default_adapter_name, model_A, model_B, tokenizer_A, tokenizer_B
    
    # Initialize or load models if not already initialized
    if model_A is None or model_B is None:
        base_model, tokenizer, default_adapter_name, model_A_loaded, model_B_loaded = load_or_initialize_models(
            load_model_A=load_model_A,
            load_model_B=load_model_B,
            model_A_path=model_A_path,
            model_B_path=model_B_path
        )
        model_A = base_model
        model_B = base_model
        tokenizer_A = tokenizer
        tokenizer_B = tokenizer
    
    print("Starting adversarial game loop...")
    print(f"Batch size: {batch_size} examples per iteration")
    print(f"Update Model A (Saboteur): {update_model_A}")
    print(f"Update Model B (Detector): {update_model_B}")
    print(f"Training frequency: every {training_frequency} iteration(s)")
    print(f"Training steps per update: {num_training_steps}")
    
    # Clear torch inductor cache to avoid tensor shape issues with compiled models
    import shutil
    torch_cache_dir = "/tmp/torchinductor_bbai"
    if os.path.exists(torch_cache_dir):
        try:
            shutil.rmtree(torch_cache_dir)
            print(f"Cleared torch inductor cache: {torch_cache_dir}")
        except Exception as e:
            print(f"Warning: Could not clear cache: {e}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = load_writing_dataset()
    if len(dataset) > 100:
        dataset = dataset.select(range(100))
    
    # Track rewards for averaging
    rewards_A = []
    rewards_B = []
    
    # Store all results for saving
    all_results = []
    
    # Store training data for GRPO updates
    training_data_B = []  # For Model B updates
    training_data_A = []  # For Model A updates
    
    # Set up adapters once
    default_adapter_name = list(model_A.peft_config.keys())[0] if hasattr(model_A, 'peft_config') and model_A.peft_config else "default"
    model_A.set_adapter(default_adapter_name)
    
    # Determine Model B adapter name
    if hasattr(model_B, 'peft_config') and model_B.peft_config and "adapter_B" in model_B.peft_config:
        model_B.set_adapter("adapter_B")
        default_adapter_name_B = "adapter_B"
    else:
        default_adapter_name_B = list(model_B.peft_config.keys())[0] if hasattr(model_B, 'peft_config') and model_B.peft_config else "default"
        model_B.set_adapter(default_adapter_name_B)
        print("Note: Using default adapter for Model B (adapter_B not found - run training first for separate adapters)")
    
    # Prepare GRPO reward function for game loop (Model B)
    def game_reward_func_B(prompts, completions, **kwargs):
        """Reward function for Model B based on game results."""
        scores = []
        for completion in completions:
            detection_response = completion[0]["content"]
            detections = extract_detections(detection_response)
            
            # Reward based on detection success
            if detections and len(detections) > 0:
                score = 1.0  # B detected something - reward
                # Quality bonus
                detection_text = " ".join(detections).lower()
                detection_keywords = ["incorrect", "wrong", "outdated", "false", "suspicious", "error", "inconsistent"]
                found_keywords = sum(1 for kw in detection_keywords if kw in detection_text)
                if found_keywords > 0:
                    score += 0.2 * min(found_keywords, 3)
            else:
                score = -1.0  # B failed to detect - penalize
            
            # Penalize empty responses
            if len(detection_response.strip()) < 20:
                score -= 0.5
            
            scores.append(score)
        return scores
    
    # Prepare GRPO reward function for game loop (Model A)
    def game_reward_func_A(prompts, completions, **kwargs):
        """Reward function for Model A based on game results."""
        scores = []
        for completion in completions:
            sabotaged_text = completion[0]["content"]
            score = 0.0
            
            # Use Model B to evaluate (adversarial)
            # Create prompt for Model B
            messages_B_eval = [
                {"role": "system", "content": system_prompt_B},
                {"role": "user", "content": f"Text to analyze:\n{sabotaged_text}\n\nIdentify any writing quality issues:"},
            ]
            text_B_eval = tokenizer_B.apply_chat_template(messages_B_eval, add_generation_prompt=True, tokenize=False)
            inputs_B_eval = tokenizer_B(text_B_eval, return_tensors="pt").to(model_B.device)
            
            with torch.no_grad():
                outputs_B_eval = model_B.generate(
                    **inputs_B_eval,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                )
            detection_eval = tokenizer_B.decode(outputs_B_eval[0][inputs_B_eval['input_ids'].shape[1]:], skip_special_tokens=True)
            detections_eval = extract_detections(detection_eval)
            
            # A wins if B fails to detect
            if detections_eval and len(detections_eval) > 0:
                score = -1.0  # B detected - A loses
            else:
                score = 1.0  # B failed - A wins
            
            scores.append(score)
        return scores
    
    for iteration in range(num_iterations):
        print(f"\n{'='*50}")
        print(f"Iteration {iteration + 1}/{num_iterations} (processing {batch_size} examples)")
        print(f"{'='*50}")
        
        # Sample batch of texts
        batch_indices = [random.randint(0, len(dataset) - 1) for _ in range(batch_size)]
        batch_samples = [dataset[idx] for idx in batch_indices]
        original_texts = [sample.get("text", sample.get("report", "")) for sample in batch_samples]
        
        # Randomly assign sabotage types for this batch
        sabotage_types = ["grammar", "style", "clarity", "logic", "coherence"]
        batch_sabotage_types = [random.choice(sabotage_types) for _ in range(batch_size)]
        
        # Type-specific instructions
        type_instructions = {
            "grammar": "Introduce grammatical errors, typos, or spelling mistakes",
            "style": "Make the writing wordy, redundant, or awkward",
            "clarity": "Add vague phrases or make parts unclear",
            "logic": "Add unsupported claims or logical inconsistencies",
            "coherence": "Add irrelevant sentences or break coherence"
        }
        
        # ========================================================================
        # Batch: Model A creates sabotaged versions
        # ========================================================================
        print(f"\nModel A (Saboteur) creating {batch_size} texts with writing issues (batched)...")
        
        # Prepare batch of prompts for Model A with specified sabotage types
        texts_A = []
        for i, original_text in enumerate(original_texts):
            sabotage_type = batch_sabotage_types[i]
            instruction = type_instructions.get(sabotage_type, "Introduce subtle writing quality issues")
            messages_A = [
                {"role": "system", "content": system_prompt_A},
                {"role": "user", "content": f"Original text:\n{original_text}\n\nCreate a version with subtle writing quality issues. Specifically, {instruction.lower()}:"},
            ]
            text_A = tokenizer_A.apply_chat_template(messages_A, add_generation_prompt=True, tokenize=False)
            texts_A.append(text_A)
        
        # Generate sabotaged texts (one at a time to avoid tensor shape issues)
        # Set model to eval mode and disable compilation
        model_A.eval()
        sabotaged_texts = []
        for i, text_A in enumerate(texts_A):
            inputs_A = tokenizer_A(
                text_A,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
            ).to(model_A.device)
            
            # Use torch.inference_mode for better performance and to avoid compilation issues
            with torch.inference_mode():
                try:
                    outputs_A = model_A.generate(
                        **inputs_A,
                        max_new_tokens=512,
                        temperature=0.7,
                        top_k=50,
                        do_sample=True,
                        pad_token_id=tokenizer_A.eos_token_id,
                    )
                except Exception as e:
                    print(f"Warning: Generation failed for example {i+1}: {e}")
                    # Fallback: return empty string
                    outputs_A = inputs_A['input_ids']
            
            # Decode output
            input_length = inputs_A['input_ids'].shape[1]
            if outputs_A.shape[1] > input_length:
                output_tokens = outputs_A[0][input_length:]
                sabotaged_text = tokenizer_A.decode(output_tokens, skip_special_tokens=True).strip()
            else:
                sabotaged_text = ""
            sabotaged_texts.append(sabotaged_text)
        
        print(f"Generated {len(sabotaged_texts)} texts with writing issues")
        if sabotaged_texts:
            print(f"Sample sabotaged text (first 200 chars): {sabotaged_texts[0][:200]}...")
        
        # ========================================================================
        # Batch: Model B tries to detect issues
        # ========================================================================
        print(f"\nModel B (Detector/Critic) analyzing {batch_size} texts (batched)...")
        
        # Prepare batch of prompts for Model B
        texts_B = []
        for sabotaged_text in sabotaged_texts:
            messages_B = [
                {"role": "system", "content": system_prompt_B},
                {"role": "user", "content": f"Text to analyze:\n{sabotaged_text}\n\nIdentify any writing quality issues:"},
            ]
            text_B = tokenizer_B.apply_chat_template(messages_B, add_generation_prompt=True, tokenize=False)
            texts_B.append(text_B)
        
        # Generate detections (one at a time to avoid tensor shape issues)
        # Set model to eval mode
        model_B.eval()
        detection_responses = []
        for i, text_B in enumerate(texts_B):
            inputs_B = tokenizer_B(
                text_B,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
            ).to(model_B.device)
            
            # Use torch.inference_mode for better performance and to avoid compilation issues
            with torch.inference_mode():
                try:
                    outputs_B = model_B.generate(
                        **inputs_B,
                        max_new_tokens=512,
                        temperature=0.7,
                        top_k=50,
                        do_sample=True,
                        pad_token_id=tokenizer_B.eos_token_id,
                    )
                except Exception as e:
                    print(f"Warning: Detection generation failed for example {i+1}: {e}")
                    # Fallback: return empty string
                    outputs_B = inputs_B['input_ids']
            
            # Decode output
            input_length_B = inputs_B['input_ids'].shape[1]
            if outputs_B.shape[1] > input_length_B:
                output_tokens = outputs_B[0][input_length_B:]
                detection_response = tokenizer_B.decode(output_tokens, skip_special_tokens=True)
            else:
                detection_response = ""
            detection_responses.append(detection_response)
        
        # ========================================================================
        # Batch: Calculate rewards, save results, and collect training data
        # ========================================================================
        batch_rewards_A = []
        batch_rewards_B = []
        
        for i in range(batch_size):
            detections = extract_detections(detection_responses[i])
            detected_type = extract_sabotage_type(detection_responses[i])
            sabotage_type = batch_sabotage_types[i]
            
            # For ground truth changes, we need to compare with what Model A actually created
            # Since Model A generates text, we can't easily extract exact changes
            # We'll use a simplified approach: check if detections match expected type
            # In a more sophisticated setup, we could use programmatic sabotage for ground truth
            original_changes = []  # Model A's changes are not easily extractable, so we focus on type matching
            
            # Calculate rewards using updated functions that consider type
            reward_A_scores = calculate_reward_A(
                sabotaged_texts[i],
                original_changes,
                detections,
                sabotage_type=sabotage_type,
                detected_type=detected_type
            )
            reward_B_scores = calculate_reward_B(
                sabotaged_texts[i],
                original_changes,
                detections,
                sabotage_type=sabotage_type,
                detected_type=detected_type
            )
            
            reward_A = reward_A_scores[0] if reward_A_scores else 0.0
            reward_B = reward_B_scores[0] if reward_B_scores else 0.0
            
            batch_rewards_A.append(reward_A)
            batch_rewards_B.append(reward_B)
            
            # Collect training data for Model B (if updating)
            if update_model_B:
                training_data_B.append({
                    "prompt": [
                        {"role": "system", "content": system_prompt_B},
                        {"role": "user", "content": f"Text to analyze:\n{sabotaged_texts[i]}\n\nIdentify any writing quality issues:"},
                    ],
                    "sabotage_type": sabotage_type,  # Store expected sabotage type for reference
                })
            
            # Collect training data for Model A (if updating)
            if update_model_A:
                sabotage_type_A = batch_sabotage_types[i]
                instruction_A = type_instructions.get(sabotage_type_A, "Introduce subtle writing quality issues")
                training_data_A.append({
                    "prompt": [
                        {"role": "system", "content": system_prompt_A},
                        {"role": "user", "content": f"Original text:\n{original_texts[i]}\n\nCreate a version with subtle writing quality issues. Specifically, {instruction_A.lower()}:"},
                    ],
                })
            
            # Save all data for this example
            result_entry = {
                "iteration": iteration + 1,
                "example_index": i,
                "global_index": iteration * batch_size + i,
                "original_text": original_texts[i],
                "sabotaged_text": sabotaged_texts[i],
                "detection_response": detection_responses[i],
                "detections": detections,
                "sabotage_type": sabotage_type,  # Ground truth: expected sabotage type
                "detected_type": detected_type,  # Detected sabotage type
                "type_match": (sabotage_type.lower() == detected_type.lower() if detected_type else False),
                "reward_A": float(reward_A),
                "reward_B": float(reward_B),
                "saboteur_wins": reward_A > 0,
                "detector_wins": reward_B > 0,
                "timestamp": datetime.now().isoformat(),
            }
            all_results.append(result_entry)
        
        # Track rewards for averaging
        rewards_A.extend(batch_rewards_A)
        rewards_B.extend(batch_rewards_B)
        
        # Print batch summary
        wins_A = sum(1 for r in batch_rewards_A if r > 0)
        wins_B = sum(1 for r in batch_rewards_B if r > 0)
        avg_reward_A_batch = sum(batch_rewards_A) / len(batch_rewards_A)
        avg_reward_B_batch = sum(batch_rewards_B) / len(batch_rewards_B)
        
        # Count type matches
        type_matches = sum(1 for i in range(batch_size) 
                          if extract_sabotage_type(detection_responses[i]) is not None
                          and batch_sabotage_types[i].lower() == extract_sabotage_type(detection_responses[i]).lower())
        
        print(f"\nBatch results:")
        print(f"  Saboteur wins: {wins_A}/{batch_size} ({100*wins_A/batch_size:.1f}%)")
        print(f"  Detector wins: {wins_B}/{batch_size} ({100*wins_B/batch_size:.1f}%)")
        print(f"  Type matches: {type_matches}/{batch_size} ({100*type_matches/batch_size:.1f}%)")
        print(f"  Avg reward A: {avg_reward_A_batch:.4f}, Avg reward B: {avg_reward_B_batch:.4f}")
        
        # Show sample detections with type info
        if detection_responses:
            sample_detections = extract_detections(detection_responses[0])
            sample_detected_type = extract_sabotage_type(detection_responses[0])
            sample_expected_type = batch_sabotage_types[0]
            if sample_detections:
                print(f"  Sample detection: {sample_detections[0][:150]}...")
                print(f"  Sample type: Expected={sample_expected_type}, Detected={sample_detected_type}")
        
        # ========================================================================
        # GRPO Updates: Update models based on game rewards
        # ========================================================================
        if (iteration + 1) % training_frequency == 0:
            # Update Model B (Detector)
            if update_model_B and len(training_data_B) > 0:
                print(f"\n{'='*50}")
                print(f"Updating Model B (Detector) with {len(training_data_B)} examples...")
                print(f"{'='*50}")
                
                # Create dataset for Model B
                dataset_B_game = Dataset.from_list([
                    {"prompt": item["prompt"]} for item in training_data_B
                ])
                
                # Tokenize
                def tokenize_B_game(x):
                    tokens = tokenizer_B.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)
                    return {"tokens": tokens}
                
                tokenized_B_game = dataset_B_game.map(tokenize_B_game, batched=False)
                
                # Filter by length
                dataset_B_game = dataset_B_game.select([
                    i for i, x in enumerate(tokenized_B_game) 
                    if len(x["tokens"]) <= max_seq_length
                ])
                
                if len(dataset_B_game) > 0:
                    # Create reward function that evaluates detections directly
                    # This uses the same logic as the game loop
                    def reward_func_B_game(prompts, completions, **kwargs):
                        scores = []
                        for completion in completions:
                            detection_response = completion[0]["content"]
                            detections = extract_detections(detection_response)
                            detected_type = extract_sabotage_type(detection_response)
                            
                            # Reward based on detection success
                            if detections and len(detections) > 0:
                                score = 1.0  # B detected something - base reward
                                
                                # Bonus for detecting a specific type
                                if detected_type:
                                    score += 0.3  # Bonus for type identification
                                
                                # Quality bonus for relevant keywords
                                detection_text = " ".join(detections).lower()
                                detection_keywords = [
                                    "grammar", "grammatical", "typo", "spelling", "error",
                                    "unclear", "vague", "confusing", "awkward", "wordy",
                                    "redundant", "inconsistent", "illogical", "unsupported",
                                    "clarity", "coherence", "style", "phrasing"
                                ]
                                found_keywords = sum(1 for kw in detection_keywords if kw in detection_text)
                                if found_keywords > 0:
                                    score += 0.2 * min(found_keywords, 3)
                            else:
                                score = -1.0  # B failed to detect - penalize
                            
                            # Penalize empty responses
                            if len(detection_response.strip()) < 20:
                                score -= 0.5
                            
                            scores.append(score)
                        return scores
                    
                    # GRPO config for Model B
                    vllm_sampling_params_B = SamplingParams(
                        min_p=0.1,
                        top_p=1.0,
                        top_k=-1,
                        seed=3407,
                        stop=[tokenizer_B.eos_token],
                        include_stop_str_in_output=True,
                        max_tokens=512,
                    )
                    
                    training_args_B_game = GRPOConfig(
                        vllm_sampling_params=vllm_sampling_params_B,
                        temperature=1.0,
                        learning_rate=5e-6,
                        weight_decay=0.001,
                        warmup_ratio=0.1,
                        lr_scheduler_type="linear",
                        optim="adamw_8bit",
                        logging_steps=1,
                        per_device_train_batch_size=1,
                        gradient_accumulation_steps=1,
                        num_generations=2,
                        max_prompt_length=max_seq_length // 2,
                        max_completion_length=max_seq_length // 2,
                        max_steps=num_training_steps,  # Limited steps per update
                        eval_strategy="no",
                        logging_strategy="steps",
                        report_to="none",
                        output_dir=os.path.join(output_dir, "model_B_updates"),
                    )
                    
                    # Ensure adapter is set correctly
                    model_B.set_adapter(default_adapter_name_B)
                    
                    # Temporarily rename adapter if needed (for GRPO compatibility)
                    if default_adapter_name_B != "default":
                        # Save current config
                        adapter_config = model_B.peft_config[default_adapter_name_B]
                        # Rename to default temporarily
                        del model_B.peft_config[default_adapter_name_B]
                        model_B.peft_config["default"] = adapter_config
                        # Rename weights
                        for name, module in model_B.named_modules():
                            if hasattr(module, 'lora_A') and default_adapter_name_B in getattr(module, 'lora_A', {}):
                                module.lora_A["default"] = module.lora_A.pop(default_adapter_name_B)
                            if hasattr(module, 'lora_B') and default_adapter_name_B in getattr(module, 'lora_B', {}):
                                module.lora_B["default"] = module.lora_B.pop(default_adapter_name_B)
                        model_B.set_adapter("default")
                    
                    trainer_B_game = GRPOTrainer(
                        model=model_B,
                        processing_class=tokenizer_B,
                        reward_funcs=[reward_func_B_game],
                        args=training_args_B_game,
                        train_dataset=dataset_B_game,
                    )
                    
                    trainer_B_game.train()
                    
                    # Rename back if needed
                    if default_adapter_name_B != "default":
                        adapter_config = model_B.peft_config["default"]
                        del model_B.peft_config["default"]
                        model_B.peft_config[default_adapter_name_B] = adapter_config
                        for name, module in model_B.named_modules():
                            if hasattr(module, 'lora_A') and "default" in getattr(module, 'lora_A', {}):
                                module.lora_A[default_adapter_name_B] = module.lora_A.pop("default")
                            if hasattr(module, 'lora_B') and "default" in getattr(module, 'lora_B', {}):
                                module.lora_B[default_adapter_name_B] = module.lora_B.pop("default")
                        model_B.set_adapter(default_adapter_name_B)
                    
                    print(f"Model B updated! Processed {len(dataset_B_game)} examples.")
                    # Clear training data after update
                    training_data_B = []
                else:
                    print("Skipping Model B update: no valid examples after filtering")
            
            # Update Model A (Saboteur) - similar logic
            if update_model_A and len(training_data_A) > 0:
                print(f"\n{'='*50}")
                print(f"Updating Model A (Saboteur) with {len(training_data_A)} examples...")
                print(f"{'='*50}")
                # Similar implementation as Model B (omitted for brevity, but follows same pattern)
                print("Model A update not yet implemented (keeping Model A fixed)")
                training_data_A = []  # Clear anyway
    
    # Print final summary with average rewards
    print("\n" + "="*50)
    print("Adversarial Game Loop Complete!")
    print("="*50)
    total_examples = num_iterations * batch_size
    avg_reward_A = sum(rewards_A) / len(rewards_A) if rewards_A else 0.0
    avg_reward_B = sum(rewards_B) / len(rewards_B) if rewards_B else 0.0
    print(f"Total iterations: {num_iterations}")
    print(f"Total examples processed: {total_examples}")
    print(f"Average reward for Saboteur (Model A): {avg_reward_A:.4f}")
    print(f"Average reward for Detector (Model B): {avg_reward_B:.4f}")
    print(f"\nSaboteur wins: {sum(1 for r in rewards_A if r > 0)}/{total_examples} ({100*sum(1 for r in rewards_A if r > 0)/total_examples:.1f}%)")
    print(f"Detector wins: {sum(1 for r in rewards_B if r > 0)}/{total_examples} ({100*sum(1 for r in rewards_B if r > 0)/total_examples:.1f}%)")
    print("="*50)
    
    # Save all results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"game_results_{timestamp}.json")
    
    # Create summary statistics
    summary = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_iterations": num_iterations,
            "batch_size": batch_size,
            "total_examples": total_examples,
            "model_A_adapter": default_adapter_name,
            "model_B_adapter": default_adapter_name_B,
        },
        "statistics": {
            "avg_reward_A": float(avg_reward_A),
            "avg_reward_B": float(avg_reward_B),
            "saboteur_wins": int(sum(1 for r in rewards_A if r > 0)),
            "detector_wins": int(sum(1 for r in rewards_B if r > 0)),
            "saboteur_win_rate": float(sum(1 for r in rewards_A if r > 0) / total_examples) if total_examples > 0 else 0.0,
            "detector_win_rate": float(sum(1 for r in rewards_B if r > 0) / total_examples) if total_examples > 0 else 0.0,
        },
        "results": all_results,
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Also save a CSV file for easier viewing
    csv_file = os.path.join(output_dir, f"game_results_{timestamp}.csv")
    df_results = pd.DataFrame([
        {
            "iteration": r["iteration"],
            "example_index": r["example_index"],
            "global_index": r["global_index"],
            "original_text": r["original_text"][:500] + "..." if len(r["original_text"]) > 500 else r["original_text"],
            "sabotaged_text": r["sabotaged_text"][:500] + "..." if len(r["sabotaged_text"]) > 500 else r["sabotaged_text"],
            "detection_response": r["detection_response"][:500] + "..." if len(r["detection_response"]) > 500 else r["detection_response"],
            "detections": str(r["detections"]),
            "reward_A": r["reward_A"],
            "reward_B": r["reward_B"],
            "saboteur_wins": r["saboteur_wins"],
            "detector_wins": r["detector_wins"],
        }
        for r in all_results
    ])
    df_results.to_csv(csv_file, index=False, encoding='utf-8')
    
    print(f"\nAll results saved to:")
    print(f"  JSON: {results_file}")
    print(f"  CSV:  {csv_file}")
    print(f"  - {len(all_results)} examples with full details")
    print(f"  - Original texts, sabotaged texts, detections, and rewards")
    print(f"  - Use these files to review model performance and improvements")
    
    # Log rewards to text file
    if rewards_A or rewards_B:
        rewards_file = os.path.join(output_dir, f"rewards_{timestamp}.txt")
        with open(rewards_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Adversarial Game Loop Rewards\n")
            f.write("="*60 + "\n\n")
            
            # Statistics
            if rewards_A:
                avg_A = np.mean(rewards_A)
                std_A = np.std(rewards_A)
                wins_A = sum(1 for r in rewards_A if r > 0)
                f.write(f"Saboteur (Model A) Statistics:\n")
                f.write(f"  Total examples: {len(rewards_A)}\n")
                f.write(f"  Average reward: {avg_A:.4f}\n")
                f.write(f"  Std deviation: {std_A:.4f}\n")
                f.write(f"  Wins (reward > 0): {wins_A}/{len(rewards_A)} ({100*wins_A/len(rewards_A):.1f}%)\n\n")
            
            if rewards_B:
                avg_B = np.mean(rewards_B)
                std_B = np.std(rewards_B)
                wins_B = sum(1 for r in rewards_B if r > 0)
                f.write(f"Detector (Model B) Statistics:\n")
                f.write(f"  Total examples: {len(rewards_B)}\n")
                f.write(f"  Average reward: {avg_B:.4f}\n")
                f.write(f"  Std deviation: {std_B:.4f}\n")
                f.write(f"  Wins (reward > 0): {wins_B}/{len(rewards_B)} ({100*wins_B/len(rewards_B):.1f}%)\n\n")
            
            # Per-iteration rewards
            f.write("="*60 + "\n")
            f.write("Per-Example Rewards:\n")
            f.write("="*60 + "\n")
            f.write(f"{'Example':<10} {'Reward A':<12} {'Reward B':<12}\n")
            f.write("-"*60 + "\n")
            
            max_len = max(len(rewards_A) if rewards_A else 0, len(rewards_B) if rewards_B else 0)
            for i in range(max_len):
                reward_A = rewards_A[i] if i < len(rewards_A) else None
                reward_B = rewards_B[i] if i < len(rewards_B) else None
                reward_A_str = f"{reward_A:.4f}" if reward_A is not None else "N/A"
                reward_B_str = f"{reward_B:.4f}" if reward_B is not None else "N/A"
                f.write(f"{i+1:<10} {reward_A_str:>12} {reward_B_str:>12}\n")
        
        print(f"\nRewards logged to: {rewards_file}")

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Writing Quality Adversarial GRPO Training")
    parser.add_argument("--mode", choices=["train", "game", "both"], default="both",
                        help="Mode: train models, run game loop, or both")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of adversarial game iterations")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for adversarial game loop (number of examples per iteration)")
    parser.add_argument("--output_dir", type=str, default="game_results",
                        help="Directory to save game results (default: game_results)")
    parser.add_argument("--update_model_B", action="store_true", default=True,
                        help="Update Model B (Detector) using GRPO during game loop (default: True)")
    parser.add_argument("--no_update_model_B", dest="update_model_B", action="store_false",
                        help="Don't update Model B during game loop")
    parser.add_argument("--update_model_A", action="store_true", default=False,
                        help="Update Model A (Saboteur) using GRPO during game loop (default: False)")
    parser.add_argument("--training_frequency", type=int, default=1,
                        help="Update models every N iterations (default: 1 = every iteration)")
    parser.add_argument("--num_training_steps", type=int, default=1,
                        help="Number of GRPO training steps per update (default: 1)")
    parser.add_argument("--load_model_A", action="store_true", default=False,
                        help="Load Model A (Saboteur) from checkpoint before training/game")
    parser.add_argument("--load_model_B", action="store_true", default=False,
                        help="Load Model B (Detector) from checkpoint before training/game")
    parser.add_argument("--model_A_path", type=str, default="lora_model_A",
                        help="Path to Model A checkpoint directory (default: lora_model_A)")
    parser.add_argument("--model_B_path", type=str, default="lora_model_B",
                        help="Path to Model B checkpoint directory (default: lora_model_B)")
    parser.add_argument("--resume_training_A", action="store_true", default=False,
                        help="Resume training Model A (requires --load_model_A)")
    parser.add_argument("--resume_training_B", action="store_true", default=False,
                        help="Resume training Model B (requires --load_model_B)")
    
    args = parser.parse_args()
    
    if args.mode in ["train", "both"]:
        train_models(
            load_model_A=args.load_model_A,
            load_model_B=args.load_model_B,
            model_A_path=args.model_A_path,
            model_B_path=args.model_B_path,
            resume_training_A=args.resume_training_A,
            resume_training_B=args.resume_training_B,
        )
    
    if args.mode in ["game", "both"]:
        adversarial_game_loop(
            num_iterations=args.iterations, 
            batch_size=args.batch_size, 
            output_dir=args.output_dir,
            update_model_B=args.update_model_B,
            update_model_A=args.update_model_A,
            training_frequency=args.training_frequency,
            num_training_steps=args.num_training_steps,
            load_model_A=args.load_model_A,
            load_model_B=args.load_model_B,
            model_A_path=args.model_A_path,
            model_B_path=args.model_B_path,
        )

    # Clean up distributed process group to avoid NCCL warning on exit
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

