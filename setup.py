# -*- coding: utf-8 -*-
"""
Setup and initialization for Financial Reports Adversarial GRPO Training

This module handles:
- CUDA environment setup
- Library imports
- GPU availability checks
"""

import os
import subprocess
import tempfile
import sys

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
    sys.exit(1)

from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer, SFTTrainer, SFTConfig
from vllm import SamplingParams
from transformers import TextStreamer
