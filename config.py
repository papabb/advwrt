# -*- coding: utf-8 -*-
"""
Configuration and model initialization for Financial Reports Adversarial GRPO Training

This module contains:
- Configuration parameters
- Model initialization functions
- Chat templates setup
"""

import os
import json
from peft import PeftModel, LoraConfig
from unsloth import FastLanguageModel

# ============================================================================
# Configuration
# ============================================================================

max_seq_length = 1024  # Reduced from 2048 to save memory
lora_rank = 32
# Try Gemma3-270M first, fallback to Gemma-2-2B if not available
base_model_name = "google/gemma-3-270m-it"  # Smaller model: 270M parameters

# Special tokens for our adversarial setup
sabotage_start = "<SABOTAGE_START>"
sabotage_end = "<SABOTAGE_END>"
detection_start = "<DETECTION_START>"
detection_end = "<DETECTION_END>"

# ============================================================================
# Chat Templates
# ============================================================================

# System prompt for Model A (Saboteur)
system_prompt_A = """You are a writing modifier. Your task is to subtly introduce writing quality issues into texts by:
1. Adding grammatical errors, typos, or awkward phrasing
2. Introducing logical inconsistencies or unclear arguments
3. Making vague or unsupported claims
4. Reducing clarity and coherence
5. Adding redundant or wordy sections

Be subtle - the changes should degrade writing quality but remain somewhat readable. Output the modified text."""

# System prompt for Model B (Detector/Critic)
system_prompt_B = """You are a writing critic and editor. Your task is to identify writing quality issues in texts.

Analyze the writing carefully and identify:
1. Grammatical errors, typos, or awkward phrasing
2. Logical inconsistencies or unclear arguments
3. Vague or unsupported claims
4. Clarity and coherence problems
5. Redundancy or wordiness
6. Any other writing quality issues

Output your findings, marking the specific parts that need improvement and explaining why."""

# Chat template for A
chat_template_A = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_prompt_A}' + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"

chat_template_A = chat_template_A.replace("'{system_prompt_A}'", f"'{system_prompt_A}'")

# Chat template for B
chat_template_B = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_prompt_B}' + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ '{detection_start}' }}"\
    "{% endif %}"

chat_template_B = chat_template_B\
    .replace("'{system_prompt_B}'", f"'{system_prompt_B}'")\
    .replace("'{detection_start}'", f"'{detection_start}'")

def setup_chat_templates(tokenizer_A, tokenizer_B):
    """Setup chat templates for both tokenizers."""
    tokenizer_A.chat_template = chat_template_A
    tokenizer_B.chat_template = chat_template_B

# ============================================================================
# Model Initialization
# ============================================================================

def load_or_initialize_models(load_model_A=False, load_model_B=False, 
                               model_A_path="lora_model_A", model_B_path="lora_model_B"):
    """
    Load previously trained models or initialize new ones.
    
    Args:
        load_model_A: Whether to load Model A from saved checkpoint
        load_model_B: Whether to load Model B from saved checkpoint
        model_A_path: Path to saved Model A checkpoint
        model_B_path: Path to saved Model B checkpoint
    
    Returns:
        base_model, tokenizer, default_adapter_name, model_A_loaded, model_B_loaded
    """
    model_A_loaded = False
    model_B_loaded = False
    
    # Load base model once (shared between A and B)
    print("Loading base model...")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=False,  # Disabled due to vLLM 0.14.0 LoRA manager API incompatibility
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.4,  # Reduced to leave more room for GRPO generation
    )
    
    # Try to load Model A if requested and exists
    if load_model_A and os.path.exists(model_A_path):
        print(f"\nLoading Model A (Saboteur) from {model_A_path}...")
        try:
            # Load the saved LoRA adapters onto the base model
            # PeftModel.from_pretrained returns the model, not a tuple
            base_model = PeftModel.from_pretrained(base_model, model_A_path)
            model_A_loaded = True
            print(f"✓ Model A loaded successfully from {model_A_path}")
        except Exception as e:
            print(f"⚠ Warning: Failed to load Model A from {model_A_path}: {e}")
            print("  Initializing new Model A adapter...")
            model_A_loaded = False
    
    # If Model A not loaded, create new adapter
    if not model_A_loaded:
        print("\nInitializing new Model A (Saboteur) adapter...")
        base_model = FastLanguageModel.get_peft_model(
            base_model,
            r=lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_rank*2,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
    
    # Get the default adapter name (usually "default" or the first adapter)
    default_adapter_name = list(base_model.peft_config.keys())[0] if hasattr(base_model, 'peft_config') and base_model.peft_config else "default"
    
    # Try to load Model B if requested and exists
    if load_model_B and os.path.exists(model_B_path):
        print(f"\nLoading Model B (Detector) from {model_B_path}...")
        try:
            # Add adapter_B from saved checkpoint
            # Load adapter config
            adapter_config_path = os.path.join(model_B_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                with open(adapter_config_path, 'r') as f:
                    adapter_config_dict = json.load(f)
                
                adapter_B_config = LoraConfig(
                    r=adapter_config_dict.get("r", lora_rank),
                    lora_alpha=adapter_config_dict.get("lora_alpha", lora_rank*2),
                    target_modules=adapter_config_dict.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
                    lora_dropout=adapter_config_dict.get("lora_dropout", 0.0),
                    bias=adapter_config_dict.get("bias", "none"),
                    task_type=adapter_config_dict.get("task_type", "CAUSAL_LM"),
                )
                
                # Add adapter and load weights
                base_model.add_adapter("adapter_B", adapter_B_config)
                base_model.load_adapter(model_B_path, "adapter_B")
                model_B_loaded = True
                print(f"✓ Model B loaded successfully from {model_B_path}")
            else:
                print(f"⚠ Warning: adapter_config.json not found in {model_B_path}")
                model_B_loaded = False
        except Exception as e:
            print(f"⚠ Warning: Failed to load Model B from {model_B_path}: {e}")
            print("  Will add adapter_B during training if needed...")
            model_B_loaded = False
    
    # Print summary
    print("\n" + "="*50)
    print("Model Initialization Summary:")
    print("="*50)
    print(f"Base model: {base_model_name}")
    adapters = list(base_model.peft_config.keys()) if hasattr(base_model, 'peft_config') and base_model.peft_config else []
    print(f"Loaded adapters: {adapters}")
    print(f"Model A (Saboteur): {'✓ Loaded' if model_A_loaded else '✗ New'}")
    print(f"Model B (Detector): {'✓ Loaded' if model_B_loaded else '✗ New (will be added during training)'}")
    print("="*50 + "\n")
    
    # Setup chat templates
    setup_chat_templates(tokenizer, tokenizer)
    
    return base_model, tokenizer, default_adapter_name, model_A_loaded, model_B_loaded
