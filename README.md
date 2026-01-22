# Financial Reports Adversarial GRPO Training

This project adapts the Qwen3-4B GRPO (Group Relative Policy Optimization) framework for an adversarial game involving financial report analysis.

## Overview

The system consists of two competing models:

- **Model A (Saboteur)**: Takes financial reports and subtly sabotages them by:
  - Adding false information or outdated numbers
  - Introducing subtle writing style issues
  - Making misleading claims or recommendations
  - Changing key financial figures slightly

- **Model B (Detector)**: Analyzes financial reports and identifies:
  - False or outdated numerical information
  - Writing style inconsistencies
  - Misleading claims or unsupported recommendations
  - Any other suspicious content

## Adversarial Game Mechanics

1. Model A receives a financial report and creates a sabotaged version
2. Model B analyzes the sabotaged report and tries to identify issues
3. **Scoring System**:
   - If B correctly identifies sabotaged parts → B gets +1 point, A gets -1 point
   - If B fails to identify → A gets +1 point, B gets -1 point
4. Both models are fine-tuned using GRPO with these rewards
5. This creates a competitive environment where both models improve iteratively

## Files

- `financial_adversarial_grpo.py`: Standalone Python script version
- `Financial_Adversarial_GRPO.ipynb`: Jupyter notebook version
- `qwen3_(4b)_grpo.py`: Original GRPO implementation (reference)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Conda environment (optional but recommended)

### Setup

1. Activate your conda environment (if using):
```bash
conda activate Adversarial
```

2. Install dependencies:
```bash
pip install unsloth vllm transformers==4.56.2
pip install --no-deps trl==0.22.2
pip install datasets pandas numpy torch
pip install pytz six idna certifi pyyaml markupsafe
```

Or using conda:
```bash
conda install -y pytz six idna certifi pyyaml markupsafe
```

Or if using the notebook, the installation cell will handle this automatically.

## Usage

### Python Script

```bash
# Train both models
python financial_adversarial_grpo.py --mode train

# Run adversarial game loop
python financial_adversarial_grpo.py --mode game --iterations 10

# Do both
python financial_adversarial_grpo.py --mode both --iterations 5
```

### Jupyter Notebook

1. Open `Financial_Adversarial_GRPO.ipynb`
2. Run all cells sequentially
3. The notebook includes:
   - Model initialization
   - Dataset loading
   - Training for both models
   - Adversarial game loop demonstration

## Dataset

The code attempts to load financial datasets from HuggingFace in this order:
1. `abhishek/FinQA` - Financial QA dataset
2. `lighteval/financial_qa` - Financial QA dataset

If these are unavailable, it automatically generates synthetic financial reports with:
- Company names
- Revenue, profit, EPS figures
- Stock recommendations
- Financial metrics (P/E ratio, market cap, dividend yield)

## Sabotage Types

Model A can apply different types of sabotage:

1. **Numeric**: Modifies financial numbers by 5-20%
2. **Temporal**: Adds outdated date references
3. **Style**: Introduces typos or style inconsistencies
4. **Logical**: Adds unsupported claims or guarantees
5. **Mixed**: Combines multiple sabotage types (default)

## Training Configuration

- **Model**: Qwen3-4B-Base
- **LoRA Rank**: 32
- **Max Sequence Length**: 2048
- **Learning Rate**: 5e-6
- **Batch Size**: 1 (with gradient accumulation)
- **Generations per Step**: 4

Adjust these parameters in the configuration section as needed.

## Output

After training, the models are saved as:
- `lora_model_A/`: Model A (Saboteur) LoRA weights
- `lora_model_B/`: Model B (Detector) LoRA weights
- `outputs/model_A/`: Model A training outputs
- `outputs/model_B/`: Model B training outputs

## Adversarial Game Loop

The game loop demonstrates the interaction:
1. Sample a financial report
2. Model A creates sabotaged version
3. Model B analyzes and detects issues
4. Calculate rewards based on detection success
5. Repeat for multiple iterations

## Customization

### Modify Sabotage Strategies

Edit the `sabotage_report()` function in the code to add new sabotage techniques.

### Adjust Reward Functions

Modify `reward_func_A()` and `reward_func_B()` to change how models are rewarded during training.

### Change Dataset

Modify `load_financial_dataset()` to use your own financial reports dataset.

## Notes

- GPU memory is split between the two models (45% each)
- The code includes fallback to synthetic data if real datasets aren't available
- Training steps are set to 50 by default for testing - increase for full training
- Both models use the same base architecture but different random seeds

## Future Improvements

- Implement true iterative adversarial training where models compete in real-time
- Add more sophisticated reward functions that consider detection accuracy
- Support for longer financial reports (10-K filings, etc.)
- Integration with real financial datasets (SEC filings, etc.)
- Evaluation metrics for measuring model improvement over iterations

## License

This code is adapted from the Unsloth Qwen3 GRPO notebook, licensed under LGPL-3.0.

## References

- Original GRPO implementation: [Unsloth Notebooks](https://github.com/unslothai/notebooks)
- Qwen3 Model: [HuggingFace](https://huggingface.co/unsloth/Qwen3-4B-Base)
- GRPO Paper: [Group Relative Policy Optimization](https://unsloth.ai/docs/new/grpo-long-context)

