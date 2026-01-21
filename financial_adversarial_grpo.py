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

from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer, SFTTrainer, SFTConfig
from vllm import SamplingParams
from transformers import TextStreamer

# ============================================================================
# Configuration
# ============================================================================

max_seq_length = 1024  # Reduced from 2048 to save memory
lora_rank = 8
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

print("Initializing shared model with two LoRA adapters...")

# Load base model once (shared between A and B)
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.4,  # Reduced to leave more room for GRPO generation
)

# Add first LoRA adapter for Model A (Saboteur)
# We'll add adapter_B later after training Model A to avoid vLLM LoRA loading issues
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

# Set up model references
# Model A uses default adapter, Model B will use adapter_B (added after training A)
model_A = base_model
model_B = base_model  # Will be the same model, but adapter_B will be added later
tokenizer_A = tokenizer
tokenizer_B = tokenizer

print("Shared model initialized:")
print(f"  - Adapter '{default_adapter_name}': for Saboteur (Model A)")
print(f"  - Adapter 'adapter_B': will be added after training Model A")

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
tokenizer_A.chat_template = chat_template_A

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
    GRPO reward function for Model A.
    This will be called during GRPO training.
    """
    scores = []
    for completion in completions:
        sabotaged_report = completion[0]["content"]
        # For now, use a simple heuristic
        # In full implementation, this would interact with Model B
        score = 0.0
        
        # Reward subtle sabotage (not too obvious)
        obvious_keywords = ["WRONG", "ERROR", "FALSE", "INCORRECT"]
        if not any(kw in sabotaged_report.upper() for kw in obvious_keywords):
            score += 0.5
        
        # Penalize if report is identical to original
        if sabotaged_report == prompts[0][-1]["content"]:
            score -= 1.0
        
        scores.append(score)
    
    return scores

def reward_func_B(prompts, completions, **kwargs):
    """
    GRPO reward function for Model B.
    This will be called during GRPO training.
    """
    scores = []
    for completion in completions:
        detection_response = completion[0]["content"]
        score = 0.0
        
        # Reward if detection markers are present
        if detection_start in detection_response and detection_end in detection_response:
            score += 1.0
        
        # Reward if detection keywords are present
        detection_keywords = ["incorrect", "wrong", "outdated", "false", "suspicious"]
        found_keywords = sum(1 for kw in detection_keywords if kw in detection_response.lower())
        score += found_keywords * 0.2
        
        # Penalize empty or very short responses
        if len(detection_response.strip()) < 20:
            score -= 1.0
        
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

def train_models():
    """Main training function for both models."""
    print("Loading dataset...")
    dataset = load_financial_dataset()
    
    # Limit dataset size for initial testing
    if len(dataset) > 1000:
        dataset = dataset.select(range(1000))
    
    print(f"Dataset size: {len(dataset)}")
    
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
    
    trainer_A = GRPOTrainer(
        model=model_A,
        processing_class=tokenizer_A,
        reward_funcs=[reward_func_A],
        args=training_args_A,
        train_dataset=dataset_A,
    )
    
    print("\nStarting training...")
    print("Loss will be logged during training. Check output for progress.\n")
    trainer_A.train()
    
    # Print final training metrics
    print("\n" + "="*50)
    print("Model A Training Complete!")
    print("="*50)
    if hasattr(trainer_A.state, 'log_history'):
        for log_entry in trainer_A.state.log_history:
            if 'loss' in log_entry:
                print(f"Step {log_entry.get('step', 'N/A')}: Loss = {log_entry['loss']:.4f}")
    
    model_A.save_pretrained("lora_model_A")
    
    # Clear GRPO checkpoint directory after Model A training
    if os.path.exists("grpo_trainer_lora_model"):
        shutil.rmtree("grpo_trainer_lora_model")
        print("Cleared GRPO checkpoint directory after Model A training")
    
    # Now add adapter_B for Model B training (after Model A is trained)
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
    
    trainer_B = GRPOTrainer(
        model=model_B,
        processing_class=tokenizer_B,
        reward_funcs=[reward_func_B],
        args=training_args_B,
        train_dataset=dataset_B,
    )
    
    print("\nStarting training...")
    print("Loss will be logged during training. Check output for progress.\n")
    trainer_B.train()
    
    # Print final training metrics
    print("\n" + "="*50)
    print("Model B Training Complete!")
    print("="*50)
    if hasattr(trainer_B.state, 'log_history'):
        for log_entry in trainer_B.state.log_history:
            if 'loss' in log_entry:
                print(f"Step {log_entry.get('step', 'N/A')}: Loss = {log_entry['loss']:.4f}")
    
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

def adversarial_game_loop(num_iterations=5):
    """
    Run adversarial game loop where A and B compete iteratively.
    """
    print("Starting adversarial game loop...")
    
    dataset = load_financial_dataset()
    if len(dataset) > 100:
        dataset = dataset.select(range(100))
    
    for iteration in range(num_iterations):
        print(f"\n{'='*50}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*50}")
        
        # Sample a report
        sample = dataset[random.randint(0, len(dataset) - 1)]
        original_report = sample.get("report", sample.get("text", ""))
        
        # Model A creates sabotaged version
        print("\nModel A (Saboteur) creating sabotaged report...")
        messages_A = [
            {"role": "system", "content": system_prompt_A},
            {"role": "user", "content": f"Original report:\n{original_report}\n\nCreate a subtly sabotaged version:"},
        ]
        
        text_A = tokenizer_A.apply_chat_template(messages_A, add_generation_prompt=True, tokenize=False)
        sampling_params = SamplingParams(temperature=0.7, top_k=50, max_tokens=1024)
        
        # Use default adapter for Model A
        default_adapter_name = list(model_A.peft_config.keys())[0] if hasattr(model_A, 'peft_config') and model_A.peft_config else "default"
        model_A.set_adapter(default_adapter_name)
        output_A = model_A.fast_generate(
            text_A,
            sampling_params=sampling_params,
        )[0].outputs[0].text
        
        sabotaged_report = output_A.strip()
        print(f"Sabotaged report (first 200 chars): {sabotaged_report[:200]}...")
        
        # Model B tries to detect issues
        print("\nModel B (Detector) analyzing report...")
        messages_B = [
            {"role": "system", "content": system_prompt_B},
            {"role": "user", "content": f"Financial report to analyze:\n{sabotaged_report}\n\nIdentify any issues:"},
        ]
        
        text_B = tokenizer_B.apply_chat_template(messages_B, add_generation_prompt=True, tokenize=False)
        
        # Use adapter_B for Model B
        model_B.set_adapter("adapter_B")
        output_B = model_B.fast_generate(
            text_B,
            sampling_params=sampling_params,
        )[0].outputs[0].text
        
        detections = extract_detections(output_B)
        print(f"Model B detections: {detections}")
        
        # Calculate rewards (simplified)
        if detections:
            reward_A = -1.0
            reward_B = 1.0
            print("Result: Model B detected issues - B wins, A loses")
        else:
            reward_A = 1.0
            reward_B = -1.0
            print("Result: Model B failed to detect - A wins, B loses")
        
        print(f"Reward A: {reward_A}, Reward B: {reward_B}")

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
    
    args = parser.parse_args()
    
    if args.mode in ["train", "both"]:
        train_models()
    
    if args.mode in ["game", "both"]:
        adversarial_game_loop(num_iterations=args.iterations)

    # Clean up distributed process group to avoid NCCL warning on exit
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

