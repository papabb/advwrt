# -*- coding: utf-8 -*-
"""
Financial Reports Adversarial GRPO Training

This script adapts Gemma3-270M GRPO for an adversarial game (using smaller model with lora_rank=8):
- Model A (Saboteur): Takes financial reports and sabotages them (false info, outdated numbers, bad writing)
- Model B (Detector): Identifies sabotaged parts in reports
- Both models are fine-tuned using GRPO based on detection success/failure
"""

import os
import re
import random
import subprocess
import tempfile
import json
from datetime import datetime

# IMPORTANT: Set CUDA_HOME BEFORE importing torch/vLLM/unsloth
# FlashInfer reads CUDA_HOME when generating build files
try:
    nvcc_path = subprocess.check_output(["which", "nvcc"], stderr=subprocess.DEVNULL).decode().strip()
    if nvcc_path:
        # Create a CUDA directory structure that FlashInfer expects
        # FlashInfer looks for CUDA_HOME/bin/nvcc
        # Try multiple locations: home directory first (user-writable), then /usr/local, then temp
        home_cuda = os.path.expanduser("~/cuda-12.8")
        cuda_dirs = [home_cuda, "/usr/local/cuda-12.8"]
        cuda_dir = None
        
        for cuda_path in cuda_dirs:
            try:
                bin_dir = os.path.join(cuda_path, "bin")
                include_dir = os.path.join(cuda_path, "include")
                os.makedirs(bin_dir, exist_ok=True)
                os.makedirs(include_dir, exist_ok=True)
                
                # Create nvcc symlink
                nvcc_link = os.path.join(bin_dir, "nvcc")
                if not os.path.exists(nvcc_link):
                    os.symlink(nvcc_path, nvcc_link)
                
                # Create CUDA header symlinks from conda environment (full CUDA toolkit)
                # Priority: conda targets/include > nvidia packages > triton
                cuda_include_sources = []
                
                # Try conda targets first (most complete)
                conda_targets_include = os.path.join(os.environ.get("CONDA_PREFIX", ""), "targets", "x86_64-linux", "include")
                if os.path.exists(conda_targets_include):
                    cuda_include_sources.append(conda_targets_include)
                
                # Try nvidia curand package
                try:
                    import site
                    site_packages = site.getsitepackages()
                    for sp in site_packages:
                        nvidia_curand = os.path.join(sp, "nvidia", "curand", "include")
                        if os.path.exists(nvidia_curand):
                            cuda_include_sources.append(nvidia_curand)
                            break
                except:
                    pass
                
                # Fallback to triton
                try:
                    import triton
                    triton_path = os.path.dirname(triton.__file__)
                    triton_cuda_include = os.path.join(triton_path, "backends", "nvidia", "include")
                    if os.path.exists(triton_cuda_include):
                        cuda_include_sources.append(triton_cuda_include)
                except:
                    pass
                
                # Symlink all CUDA headers
                for source_include in cuda_include_sources:
                    try:
                        for item in os.listdir(source_include):
                            src = os.path.join(source_include, item)
                            dst = os.path.join(include_dir, item)
                            if not os.path.exists(dst):
                                if os.path.isdir(src):
                                    os.symlink(src, dst)
                                else:
                                    os.symlink(src, dst)
                    except Exception as e:
                        continue
                
                # Create cccL directory (needed for some builds)
                os.makedirs(os.path.join(include_dir, "cccl"), exist_ok=True)
                
                # Verify curand.h is available (critical for FlashInfer sampling)
                curand_h = os.path.join(include_dir, "curand.h")
                if not os.path.exists(curand_h):
                    # Try to find and symlink it directly
                    for source_include in cuda_include_sources:
                        potential_curand = os.path.join(source_include, "curand.h")
                        if os.path.exists(potential_curand):
                            os.symlink(potential_curand, curand_h)
                            break
                
                if os.path.exists(curand_h):
                    print(f"CUDA headers linked successfully (including curand.h)")
                else:
                    print(f"Warning: curand.h not found, FlashInfer sampling may fail")
                
                cuda_dir = cuda_path
                print(f"Created CUDA structure at {cuda_path}")
                break
            except (OSError, PermissionError) as e:
                continue
        
        if cuda_dir is None:
            # Fallback to temp directory
            temp_cuda_dir = tempfile.mkdtemp(prefix="cuda_")
            temp_bin_dir = os.path.join(temp_cuda_dir, "bin")
            os.makedirs(temp_bin_dir, exist_ok=True)
            os.symlink(nvcc_path, os.path.join(temp_bin_dir, "nvcc"))
            cuda_dir = temp_cuda_dir
            print(f"Created temp CUDA_HOME={temp_cuda_dir}")
        
        os.environ["CUDA_HOME"] = cuda_dir
        os.environ["CUDA_PATH"] = os.environ["CUDA_HOME"]
        
        # Add nvcc directory to PATH
        nvcc_dir = os.path.dirname(nvcc_path)
        if nvcc_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = nvcc_dir + os.pathsep + os.environ.get("PATH", "")
        
        # Add cicc (CUDA internal compiler) to PATH - required for nvcc compilation
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix:
            nvvm_bin = os.path.join(conda_prefix, "nvvm", "bin")
            if os.path.exists(nvvm_bin):
                if nvvm_bin not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = nvvm_bin + os.pathsep + os.environ.get("PATH", "")
                # Also symlink nvvm directory to CUDA_HOME (nvcc expects it there)
                nvvm_dir = os.path.join(conda_prefix, "nvvm")
                cuda_nvvm = os.path.join(cuda_dir, "nvvm")
                if not os.path.exists(cuda_nvvm) and os.path.exists(nvvm_dir):
                    try:
                        os.symlink(nvvm_dir, cuda_nvvm)
                        print(f"Linked nvvm directory to {cuda_nvvm}")
                    except Exception:
                        pass
        
        # Add CUDA libraries (lib64 directory) - required for linking
        if conda_prefix:
            cuda_lib64 = os.path.join(cuda_dir, "lib64")
            os.makedirs(cuda_lib64, exist_ok=True)
            
            # Link from targets/lib (main CUDA libraries)
            conda_lib = os.path.join(conda_prefix, "targets", "x86_64-linux", "lib")
            if os.path.exists(conda_lib):
                try:
                    for lib_file in os.listdir(conda_lib):
                        if lib_file.startswith(("libcuda", "libcudart", "libcurand", "libcublas")):
                            src = os.path.join(conda_lib, lib_file)
                            dst = os.path.join(cuda_lib64, lib_file)
                            if os.path.isfile(src) and not os.path.exists(dst):
                                os.symlink(src, dst)
                except Exception:
                    pass
            
            # Link from lib/stubs (libcuda stub) - critical for linking
            conda_stubs = os.path.join(conda_prefix, "lib", "stubs")
            cuda_stubs = os.path.join(cuda_lib64, "stubs")
            os.makedirs(cuda_stubs, exist_ok=True)
            if os.path.exists(conda_stubs):
                try:
                    for lib_file in os.listdir(conda_stubs):
                        if lib_file.startswith("libcuda"):
                            src = os.path.join(conda_stubs, lib_file)
                            dst = os.path.join(cuda_stubs, lib_file)
                            if os.path.isfile(src) and not os.path.exists(dst):
                                os.symlink(src, dst)
                except Exception:
                    pass
            
            print(f"Linked CUDA libraries to {cuda_lib64}")
        
        print(f"Set CUDA_HOME={os.environ['CUDA_HOME']}, nvcc at {nvcc_path}")
except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
    # If nvcc not found, use FLASH_ATTN backend which doesn't need nvcc
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    print(f"nvcc setup failed ({e}), using FLASH_ATTN backend (no CUDA compiler needed)")

import numpy as np
import pandas as pd
import torch
import gc
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from datasets import load_dataset, Dataset

# Check CUDA availability before importing unsloth
if not torch.cuda.is_available():
    print("="*60)
    print("ERROR: CUDA/GPU not available!")
    print("="*60)
    print("PyTorch cannot detect a CUDA-capable GPU.")
    print("\nPossible causes:")
    print("  1. No GPU installed")
    print("  2. NVIDIA driver not installed or outdated")
    print("  3. Driver/library version mismatch (try: sudo reboot)")
    print("  4. PyTorch not compiled with CUDA support")
    print("\nTo check:")
    print("  - Run: nvidia-smi")
    print("  - Run: python -c 'import torch; print(torch.cuda.is_available())'")
    print("  - If nvidia-smi shows 'Driver/library version mismatch', reboot your system")
    print("\nThis script requires a GPU for training. Exiting...")
    print("="*60)
    import sys
    sys.exit(1)

from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer, SFTTrainer, SFTConfig
from vllm import SamplingParams
from transformers import TextStreamer

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
            # Load the PEFT model with saved adapters
            base_model = FastLanguageModel.from_pretrained(
                model_name=base_model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=False,
                fast_inference=False,
                max_lora_rank=lora_rank,
                gpu_memory_utilization=0.4,
            )
            # Load the saved LoRA adapters
            from peft import PeftModel
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
            from peft import PeftConfig
            # Load adapter config
            adapter_config_path = os.path.join(model_B_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                import json
                with open(adapter_config_path, 'r') as f:
                    adapter_config_dict = json.load(f)
                
                from peft import LoraConfig
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
# Chat Templates
# ============================================================================

# System prompt for Model A (Saboteur)
system_prompt_A = """You are a financial report modifier. Your task is to subtly sabotage financial reports by:
1. Adding false information or outdated numbers
2. Introducing subtle writing style issues
3. Making misleading claims or recommendations
4. Changing key financial figures slightly

Be subtle - the changes should be believable but incorrect. Output the sabotaged report."""

# System prompt for Model B (Detector)
system_prompt_B = """You are a financial report auditor. Your task is to identify any sabotaged, incorrect, or suspicious parts in financial reports.

Analyze the report carefully and identify:
1. False or outdated numerical information
2. Writing style inconsistencies
3. Misleading claims or unsupported recommendations
4. Any other suspicious content

Output your findings, marking the specific parts that are problematic."""

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
# Dataset Loading and Preprocessing
# ============================================================================

def load_financial_dataset():
    """
    Load financial reports dataset.
    Try multiple sources, fallback to synthetic data if needed.
    """
    print("Loading financial dataset...")
    
    # Try to load from HuggingFace datasets
    try:
        # Try FinQA or similar datasets
        dataset = load_dataset("abhishek/FinQA", split="train[:1000]")  # Limit for testing
        print(f"Loaded FinQA dataset: {len(dataset)} examples")
        return dataset
    except:
        pass
    
    try:
        # Try other financial datasets
        dataset = load_dataset("lighteval/financial_qa", split="train[:1000]")
        print(f"Loaded financial_qa dataset: {len(dataset)} examples")
        return dataset
    except:
        pass
    
    # Fallback: Create synthetic financial reports
    print("Creating synthetic financial reports dataset...")
    synthetic_reports = []
    companies = ["TechCorp", "FinanceInc", "RetailGroup", "EnergyCo", "HealthSys"]
    
    for i in range(500):
        company = random.choice(companies)
        revenue = random.randint(1000000, 100000000)
        profit = random.randint(100000, revenue // 10)
        eps = round(profit / 1000000, 2)
        
        report = f"""
{company} Quarterly Financial Report

Revenue: ${revenue:,}
Net Profit: ${profit:,}
Earnings Per Share (EPS): ${eps}
Stock Price: ${random.randint(50, 200)}

Analysis:
The company shows strong performance this quarter. Revenue increased by {random.randint(5, 25)}% compared to last quarter.
We recommend a {'BUY' if eps > 0.5 else 'HOLD'} rating for this stock.

Key Metrics:
- P/E Ratio: {random.randint(10, 30)}
- Market Cap: ${revenue * random.randint(5, 15):,}
- Dividend Yield: {random.uniform(1.0, 5.0):.2f}%

Outlook:
The company expects continued growth in the next quarter. Management is optimistic about future prospects.
"""
        synthetic_reports.append({
            "report": report.strip(),
            "company": company,
            "revenue": revenue,
            "profit": profit,
            "eps": eps,
        })
    
    return Dataset.from_list(synthetic_reports)

# ============================================================================
# Sabotage Functions (for Model A)
# ============================================================================

def sabotage_report(report_text, sabotage_type="mixed"):
    """
    Apply various sabotage techniques to a financial report.
    Returns the sabotaged report and metadata about what was changed.
    """
    sabotaged = report_text
    changes = []
    
    if sabotage_type == "numeric" or sabotage_type == "mixed":
        # Change numbers slightly
        def modify_number(match):
            num_str = match.group(0)
            try:
                # Remove commas and $ signs
                clean_num = num_str.replace(",", "").replace("$", "").replace("%", "")
                if "." in clean_num:
                    num = float(clean_num)
                    # Modify by 5-20%
                    multiplier = random.uniform(0.8, 1.2)
                    new_num = num * multiplier
                    if "$" in num_str:
                        return f"${new_num:,.2f}"
                    elif "%" in num_str:
                        return f"{new_num:.2f}%"
                    else:
                        return f"{new_num:,.2f}"
                else:
                    num = int(clean_num)
                    multiplier = random.uniform(0.85, 1.15)
                    new_num = int(num * multiplier)
                    if "$" in num_str:
                        return f"${new_num:,}"
                    else:
                        return str(new_num)
            except:
                return num_str
        
        # Find and modify financial numbers
        pattern = r'\$[\d,]+(?:\.\d+)?|[\d,]+\.[\d]+%|[\d,]+%'
        matches = list(re.finditer(pattern, sabotaged))
        if matches:
            # Modify 1-3 random numbers
            num_to_modify = min(random.randint(1, 3), len(matches))
            indices = random.sample(range(len(matches)), num_to_modify)
            for idx in sorted(indices, reverse=True):
                match = matches[idx]
                old_val = match.group(0)
                new_val = modify_number(match)
                sabotaged = sabotaged[:match.start()] + new_val + sabotaged[match.end():]
                changes.append(f"Changed {old_val} to {new_val}")
    
    if sabotage_type == "temporal" or sabotage_type == "mixed":
        # Add outdated references
        outdated_phrases = [
            "as of Q1 2020",
            "based on 2019 data",
            "according to last year's figures",
        ]
        if random.random() < 0.3:
            phrase = random.choice(outdated_phrases)
            # Insert at a random position
            sentences = sabotaged.split(".")
            if len(sentences) > 2:
                insert_pos = random.randint(1, len(sentences) - 1)
                sentences[insert_pos] = sentences[insert_pos] + f" {phrase}."
                sabotaged = ".".join(sentences)
                changes.append(f"Added outdated reference: {phrase}")
    
    if sabotage_type == "style" or sabotage_type == "mixed":
        # Introduce writing style issues
        style_issues = [
            ("recommend", "recomend"),  # Typo
            ("analysis", "analisis"),   # Typo
            ("recommend a BUY", "strongly recommend a BUY"),  # Overstatement
        ]
        if random.random() < 0.2:
            old, new = random.choice(style_issues)
            if old in sabotaged:
                sabotaged = sabotaged.replace(old, new, 1)
                changes.append(f"Style change: {old} -> {new}")
    
    if sabotage_type == "logical" or sabotage_type == "mixed":
        # Add unsupported claims
        unsupported = [
            "This guarantees exceptional returns.",
            "The stock is certain to outperform.",
            "Risk is minimal and returns are guaranteed.",
        ]
        if random.random() < 0.2:
            claim = random.choice(unsupported)
            sentences = sabotaged.split(".")
            if len(sentences) > 1:
                insert_pos = random.randint(0, len(sentences) - 1)
                sentences[insert_pos] = sentences[insert_pos] + f" {claim}"
                sabotaged = ".".join(sentences)
                changes.append(f"Added unsupported claim: {claim}")
    
    return sabotaged, changes

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
    
    # Also look for common detection phrases
    detection_keywords = [
        "incorrect", "wrong", "outdated", "false", "suspicious",
        "misleading", "error", "inaccurate", "problematic"
    ]
    
    sentences = response.split(".")
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in detection_keywords):
            detections.append(sentence.strip())
    
    return detections

# ============================================================================
# Plotting Functions
# ============================================================================

def plot_rewards(rewards_A, rewards_B, output_dir="game_results", title="Rewards Over Time", 
                 save_name="rewards_plot.png", iteration_numbers=None):
    """
    Plot rewards for Saboteur (Model A) and Detector (Model B).
    
    Args:
        rewards_A: List of rewards for Model A (Saboteur)
        rewards_B: List of rewards for Model B (Detector)
        output_dir: Directory to save the plot
        title: Plot title
        save_name: Filename for saved plot
        iteration_numbers: Optional list of iteration numbers (for x-axis)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not rewards_A and not rewards_B:
        print("No reward data to plot")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Prepare x-axis
    if iteration_numbers is None:
        x_axis = list(range(len(rewards_A) if rewards_A else len(rewards_B)))
    else:
        x_axis = iteration_numbers
    
    # Plot 1: Individual rewards over time
    ax1 = axes[0]
    if rewards_A:
        ax1.plot(x_axis[:len(rewards_A)], rewards_A, 'r-', label='Saboteur (Model A)', alpha=0.7, linewidth=1.5)
        ax1.scatter(x_axis[:len(rewards_A)], rewards_A, c='red', s=20, alpha=0.5)
    if rewards_B:
        ax1.plot(x_axis[:len(rewards_B)], rewards_B, 'b-', label='Detector (Model B)', alpha=0.7, linewidth=1.5)
        ax1.scatter(x_axis[:len(rewards_B)], rewards_B, c='blue', s=20, alpha=0.5)
    
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax1.set_xlabel('Iteration/Example', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title(f'{title} - Individual Rewards', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Moving average rewards
    ax2 = axes[1]
    window_size = max(5, len(rewards_A) // 20) if rewards_A else max(5, len(rewards_B) // 20)
    window_size = min(window_size, 50)  # Cap at 50
    
    if rewards_A and len(rewards_A) >= window_size:
        moving_avg_A = pd.Series(rewards_A).rolling(window=window_size, min_periods=1).mean()
        ax2.plot(x_axis[:len(rewards_A)], moving_avg_A, 'r-', label=f'Saboteur (Model A) - {window_size}-point MA', 
                linewidth=2, alpha=0.8)
    
    if rewards_B and len(rewards_B) >= window_size:
        moving_avg_B = pd.Series(rewards_B).rolling(window=window_size, min_periods=1).mean()
        ax2.plot(x_axis[:len(rewards_B)], moving_avg_B, 'b-', label=f'Detector (Model B) - {window_size}-point MA', 
                linewidth=2, alpha=0.8)
    
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax2.set_xlabel('Iteration/Example', fontsize=12)
    ax2.set_ylabel('Moving Average Reward', fontsize=12)
    ax2.set_title(f'{title} - Moving Average ({window_size}-point window)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = []
    if rewards_A:
        avg_A = np.mean(rewards_A)
        std_A = np.std(rewards_A)
        wins_A = sum(1 for r in rewards_A if r > 0)
        stats_text.append(f'Saboteur: Avg={avg_A:.3f}, Std={std_A:.3f}, Wins={wins_A}/{len(rewards_A)}')
    if rewards_B:
        avg_B = np.mean(rewards_B)
        std_B = np.std(rewards_B)
        wins_B = sum(1 for r in rewards_B if r > 0)
        stats_text.append(f'Detector: Avg={avg_B:.3f}, Std={std_B:.3f}, Wins={wins_B}/{len(rewards_B)}')
    
    if stats_text:
        fig.text(0.5, 0.02, ' | '.join(stats_text), ha='center', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, save_name)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nReward plot saved to: {plot_path}")
    
    plt.close()

# ============================================================================
# Reward Functions
# ============================================================================

def calculate_reward_A(sabotaged_report, original_changes, detections, **kwargs):
    """
    Reward function for Model A (Saboteur).
    A gets +1 if B fails to detect, -1 if B detects correctly.
    Partial credit for subtle sabotage.
    """
    scores = []
    
    # Check if B detected anything
    if not detections or len(detections) == 0:
        # B failed to detect - A wins
        scores.append(1.0)
    else:
        # B detected something - check if it matches actual changes
        detection_text = " ".join(detections).lower()
        changes_text = " ".join(original_changes).lower()
        
        # Check overlap
        if any(change.lower() in detection_text for change in original_changes):
            # B correctly identified changes - A loses
            scores.append(-1.0)
        else:
            # B detected something but not the actual changes - partial credit
            scores.append(0.0)
    
    return scores

def calculate_reward_B(sabotaged_report, original_changes, detections, **kwargs):
    """
    Reward function for Model B (Detector).
    B gets +1 if correctly identifies sabotaged parts, -1 if fails.
    """
    scores = []
    
    if not detections or len(detections) == 0:
        # B failed to detect - B loses
        scores.append(-1.0)
    else:
        # B detected something - check accuracy
        detection_text = " ".join(detections).lower()
        changes_text = " ".join(original_changes).lower()
        
        # Check if detection matches actual changes
        matches = sum(1 for change in original_changes if change.lower() in detection_text)
        
        if matches > 0:
            # B correctly identified at least some changes - B wins
            accuracy = matches / len(original_changes) if original_changes else 0
            scores.append(1.0 * accuracy)  # Partial credit based on accuracy
        else:
            # B detected something but not the actual changes - B loses
            scores.append(-1.0)
    
    return scores

# ============================================================================
# GRPO Reward Functions (for TRL compatibility)
# ============================================================================

def reward_func_A(prompts, completions, **kwargs):
    """
    GRPO reward function for Model A (Saboteur).
    This is called by GRPOTrainer during training to compute rewards.
    
    The reward is computed adversarially:
    - Model A generates a sabotaged report (in completions)
    - Model B tries to detect issues in the sabotaged report
    - If B fails to detect: A gets +1.0 reward
    - If B detects: A gets -1.0 reward
    
    This reward is then used by GRPO to update Model A's weights via policy gradients.
    """
    scores = []
    
    # Extract original report from prompts
    original_report = ""
    for msg in prompts[0]:
        if msg["role"] == "user":
            # Extract report from user message
            content = msg["content"]
            if "Original report:" in content:
                original_report = content.split("Original report:")[1].split("\n\nCreate")[0].strip()
            break
    
    for completion in completions:
        sabotaged_report = completion[0]["content"]
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
            {"role": "user", "content": f"Financial report to analyze:\n{sabotaged_report}\n\nIdentify any issues:"},
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
        obvious_keywords = ["WRONG", "ERROR", "FALSE", "INCORRECT", "FAKE"]
        if any(kw in sabotaged_report.upper() for kw in obvious_keywords):
            score -= 0.2  # Penalize obvious sabotage
        
        # Penalize if report is identical to original
        if original_report and sabotaged_report.strip() == original_report.strip():
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
    
    # Extract sabotaged report from prompts
    sabotaged_report = ""
    for msg in prompts[0]:
        if msg["role"] == "user":
            content = msg["content"]
            if "Financial report to analyze:" in content:
                sabotaged_report = content.split("Financial report to analyze:")[1].split("\n\nIdentify")[0].strip()
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
            detection_keywords = ["incorrect", "wrong", "outdated", "false", "suspicious", "error", "inconsistent"]
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
        
        # If we have the sabotaged report, we could do more sophisticated checking
        # For now, we use the heuristic above
        
        scores.append(score)
    
    return scores

# ============================================================================
# Main Training Loop
# ============================================================================

def prepare_dataset_for_A(dataset):
    """Prepare dataset for Model A (Saboteur) training."""
    def format_for_A(example):
        report = example.get("report", example.get("text", ""))
        if not report:
            return None
        
        return {
            "prompt": [
                {"role": "system", "content": system_prompt_A},
                {"role": "user", "content": f"Original report:\n{report}\n\nCreate a subtly sabotaged version:"},
            ],
        }
    
    formatted = dataset.map(format_for_A, remove_columns=dataset.column_names)
    formatted = formatted.filter(lambda x: x["prompt"] is not None)
    return formatted

def prepare_dataset_for_B(dataset):
    """Prepare dataset for Model B (Detector) training."""
    def format_for_B(example):
        report = example.get("report", example.get("text", ""))
        if not report:
            return None
        
        # Create a sabotaged version for training
        sabotaged, changes = sabotage_report(report)
        
        return {
            "prompt": [
                {"role": "system", "content": system_prompt_B},
                {"role": "user", "content": f"Financial report to analyze:\n{sabotaged}\n\nIdentify any issues:"},
            ],
            "changes": changes,  # Ground truth for evaluation
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
    dataset = load_financial_dataset()
    
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
    
    # Plot training rewards
    if training_rewards_A or training_rewards_B:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_rewards(
            training_rewards_A,
            training_rewards_B,
            output_dir="outputs",
            title="GRPO Training Rewards",
            save_name=f"training_rewards_{timestamp}.png"
        )

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
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = load_financial_dataset()
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
            sabotaged_report = completion[0]["content"]
            score = 0.0
            
            # Use Model B to evaluate (adversarial)
            # Create prompt for Model B
            messages_B_eval = [
                {"role": "system", "content": system_prompt_B},
                {"role": "user", "content": f"Financial report to analyze:\n{sabotaged_report}\n\nIdentify any issues:"},
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
        
        # Sample batch of reports
        batch_indices = [random.randint(0, len(dataset) - 1) for _ in range(batch_size)]
        batch_samples = [dataset[idx] for idx in batch_indices]
        original_reports = [sample.get("report", sample.get("text", "")) for sample in batch_samples]
        
        # ========================================================================
        # Batch: Model A creates sabotaged versions
        # ========================================================================
        print(f"\nModel A (Saboteur) creating {batch_size} sabotaged reports (batched)...")
        
        # Prepare batch of prompts for Model A
        texts_A = []
        for original_report in original_reports:
            messages_A = [
                {"role": "system", "content": system_prompt_A},
                {"role": "user", "content": f"Original report:\n{original_report}\n\nCreate a subtly sabotaged version:"},
            ]
            text_A = tokenizer_A.apply_chat_template(messages_A, add_generation_prompt=True, tokenize=False)
            texts_A.append(text_A)
        
        # Batch tokenize
        inputs_A = tokenizer_A(
            texts_A,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        ).to(model_A.device)
        
        # Batch generate sabotaged reports
        with torch.no_grad():
            outputs_A = model_A.generate(
                **inputs_A,
                max_new_tokens=512,  # Reduced for batch efficiency
                temperature=0.7,
                top_k=50,
                do_sample=True,
                pad_token_id=tokenizer_A.eos_token_id,  # Important for batching
            )
        
        # Decode batch outputs
        sabotaged_reports = []
        input_lengths = inputs_A['input_ids'].shape[1]
        for i in range(batch_size):
            output_tokens = outputs_A[i][input_lengths:]
            sabotaged_report = tokenizer_A.decode(output_tokens, skip_special_tokens=True).strip()
            sabotaged_reports.append(sabotaged_report)
        
        print(f"Generated {len(sabotaged_reports)} sabotaged reports")
        if sabotaged_reports:
            print(f"Sample sabotaged report (first 200 chars): {sabotaged_reports[0][:200]}...")
        
        # ========================================================================
        # Batch: Model B tries to detect issues
        # ========================================================================
        print(f"\nModel B (Detector) analyzing {batch_size} reports (batched)...")
        
        # Prepare batch of prompts for Model B
        texts_B = []
        for sabotaged_report in sabotaged_reports:
            messages_B = [
                {"role": "system", "content": system_prompt_B},
                {"role": "user", "content": f"Financial report to analyze:\n{sabotaged_report}\n\nIdentify any issues:"},
            ]
            text_B = tokenizer_B.apply_chat_template(messages_B, add_generation_prompt=True, tokenize=False)
            texts_B.append(text_B)
        
        # Batch tokenize
        inputs_B = tokenizer_B(
            texts_B,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        ).to(model_B.device)
        
        # Batch generate detections
        with torch.no_grad():
            outputs_B = model_B.generate(
                **inputs_B,
                max_new_tokens=512,  # Reduced for batch efficiency
                temperature=0.7,
                top_k=50,
                do_sample=True,
                pad_token_id=tokenizer_B.eos_token_id,  # Important for batching
            )
        
        # Decode batch outputs
        detection_responses = []
        input_lengths_B = inputs_B['input_ids'].shape[1]
        for i in range(batch_size):
            output_tokens = outputs_B[i][input_lengths_B:]
            detection_response = tokenizer_B.decode(output_tokens, skip_special_tokens=True)
            detection_responses.append(detection_response)
        
        # ========================================================================
        # Batch: Calculate rewards, save results, and collect training data
        # ========================================================================
        batch_rewards_A = []
        batch_rewards_B = []
        
        for i in range(batch_size):
            detections = extract_detections(detection_responses[i])
            
            # Calculate rewards (simplified)
            if detections:
                reward_A = -1.0
                reward_B = 1.0
            else:
                reward_A = 1.0
                reward_B = -1.0
            
            batch_rewards_A.append(reward_A)
            batch_rewards_B.append(reward_B)
            
            # Collect training data for Model B (if updating)
            if update_model_B:
                training_data_B.append({
                    "prompt": [
                        {"role": "system", "content": system_prompt_B},
                        {"role": "user", "content": f"Financial report to analyze:\n{sabotaged_reports[i]}\n\nIdentify any issues:"},
                    ],
                })
            
            # Collect training data for Model A (if updating)
            if update_model_A:
                training_data_A.append({
                    "prompt": [
                        {"role": "system", "content": system_prompt_A},
                        {"role": "user", "content": f"Original report:\n{original_reports[i]}\n\nCreate a subtly sabotaged version:"},
                    ],
                })
            
            # Save all data for this example
            result_entry = {
                "iteration": iteration + 1,
                "example_index": i,
                "global_index": iteration * batch_size + i,
                "original_report": original_reports[i],
                "sabotaged_report": sabotaged_reports[i],
                "detection_response": detection_responses[i],
                "detections": detections,
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
        
        print(f"\nBatch results:")
        print(f"  Saboteur wins: {wins_A}/{batch_size} ({100*wins_A/batch_size:.1f}%)")
        print(f"  Detector wins: {wins_B}/{batch_size} ({100*wins_B/batch_size:.1f}%)")
        print(f"  Avg reward A: {avg_reward_A_batch:.4f}, Avg reward B: {avg_reward_B_batch:.4f}")
        
        # Show sample detections
        if detection_responses:
            sample_detections = extract_detections(detection_responses[0])
            if sample_detections:
                print(f"  Sample detection: {sample_detections[0][:150]}...")
        
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
            "original_report": r["original_report"][:500] + "..." if len(r["original_report"]) > 500 else r["original_report"],
            "sabotaged_report": r["sabotaged_report"][:500] + "..." if len(r["sabotaged_report"]) > 500 else r["sabotaged_report"],
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
    print(f"  - Original reports, sabotaged reports, detections, and rewards")
    print(f"  - Use these files to review model performance and improvements")
    
    # Plot rewards
    if rewards_A or rewards_B:
        plot_rewards(
            rewards_A, 
            rewards_B, 
            output_dir=output_dir,
            title="Adversarial Game Loop Rewards",
            save_name=f"rewards_plot_{timestamp}.png"
        )

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Financial Adversarial GRPO Training")
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

