# Runtime-Optimized Synaptic Input Consolidation (SIC) for Efficient CNN Deployment

This repository contains the official implementation of **Runtime-Optimized Synaptic Input Consolidation (SIC)**, a post-training compression method that clusters and consolidates synaptic weights in neural networks while maintaining accuracy. This implementation focuses on making SIC practically usable by reducing preprocessing runtime by 70-90% compared to baseline SIC, making it suitable for offline compression workflows in edge deployment pipelines.

## Overview

**Synaptic Input Consolidation (SIC)** is a training-free, architecture-preserving compression technique that:
- Clusters similar incoming weights per neuron using Jenks natural breaks
- Pre-sums inputs within each cluster, reducing multiply count and unique parameters
- Maintains exact accuracy on an acceptance subset (ASC) of training/validation data
- Preserves network topology without requiring retraining

**Runtime-Optimized SIC** extends the baseline algorithm with two key optimizations:
1. **Filtered Validation Subset**: Replaces expensive full-dataset validation with a small, class-balanced subset that acts as a proxy for full validation
2. **Tiered Acceptance**: Evaluates candidate configurations in stages (fast → medium → full), avoiding expensive validation passes on clearly suboptimal configurations

Our experiments show that runtime-optimized SIC reduces preprocessing time by **70-90%** while keeping test accuracy within **~0.2 percentage points** of baseline SIC, making it practical for real-world deployment workflows.

## Requirements

### Dependencies

The following dependencies are required (with tested versions):

```
torch: 2.2.1+cu118
torchvision: 0.17.1+cu118
numpy: 1.26.2
pandas: 2.2.1
psutil: 5.9.8
jenkspy: 0.4.1
matplotlib: 3.8.3
dill: 0.3.8
PyYAML: 6.0.1
```

### Installation

To install requirements:

```bash
cd SIC
pip install -r requirements.txt
```

> **Note**: For CUDA support, ensure you have compatible CUDA drivers installed. The `+cu118` suffix indicates CUDA 11.8 support. Adjust PyTorch installation if using a different CUDA version.

## Dataset Setup

The repository supports the following datasets:
- **MNIST**: Handwritten digits (28×28 grayscale, 10 classes, 60K train / 10K test)
- **Fashion-MNIST**: Fashion items (28×28 grayscale, 10 classes, 60K train / 10K test)
- **CIFAR-10**: Natural images (32×32 RGB, 10 classes, 50K train / 10K test)
- **SVHN**: Street View House Numbers (32×32 RGB, 10 classes, ~73K train / 26K test)

Datasets are automatically downloaded to `./data/` on first use. No manual download is required.

## Training

### Training Base Models

Before applying SIC, you need to train a base model. The repository expects pre-trained model weights in the `saved_models/` directory. 

Example training workflow (not included in this repository):
1. Train your model using standard PyTorch training
2. Save the model weights to `saved_models/{MODEL}_{DATASET}.pth`
3. Apply SIC optimization using the instructions below

### Available Models

The following model architectures are supported (tested in our experiments):
- `SmallMNISTCNN`: Simple 2-layer fully connected network for MNIST
- `LeNet5Tanh`: LeNet-5 with Tanh activations (tested on MNIST, Fashion-MNIST)
- `LeNet5ELU`: LeNet-5 with ELU activations (tested on MNIST)
- `ResNet16`: ResNet with 16 layers (tested on CIFAR-10, SVHN)

Additional models available:
- `LeNet5Seed`: LeNet-5 feature extractor (no classifier)
- `SeedCNN`: Seed CNN architecture
- `ResNet8`: ResNet with 8 layers
- `MojoCIFARCNN`: CNN optimized for CIFAR-10

## Running SIC

### Basic Usage

To run runtime-optimized SIC on a pre-trained model:

```bash
cd SIC
python -m scripts.run_sic --config config/sic.yaml
```

### Methodology: Runtime-Optimized SIC

Our implementation includes two key optimizations that dramatically reduce preprocessing time:

#### 1. Filtered Validation Subset

Instead of evaluating every SIC candidate on the full train+validation set, we construct a small, class-balanced acceptance subset `D_filter` that acts as a proxy for full validation:

- **Class Balance**: Allocates budget evenly across classes to ensure rare classes are represented
- **Hard-Example Topping**: Fills remaining slots with hard examples (low prediction margin) to capture edge cases
- **Size Budget**: Configurable maximum size (typically a few thousand examples) that caps evaluation cost

This filtered subset completely replaces full validation in the inner SIC loop, cutting wall-clock time per candidate by an order of magnitude while providing stable accuracy signals.

#### 2. Tiered Adaptive Acceptance

To avoid wasting computation on obviously bad configurations, we use a **tiered All Samples Correct (ASC)** procedure that evaluates candidates in stages:

- **Fast Tier**: Evaluates on a small prefix (e.g., 1024 examples). If any misclassification is found, the candidate is immediately rejected.
- **Medium Tier**: If fast tier passes, expands to an intermediate slice (e.g., 2048 examples) with the same all-correct requirement.
- **Full Tier**: Only candidates passing both fast and medium tiers are evaluated on the remaining acceptance set.

Failed candidates trigger promotion of the first failing index to the front of the permutation, so future candidates encounter problematic examples early. This staged evaluation provides an additional 60-89% reduction in runtime relative to filtered-only SIC.

### Configuration

The main configuration file is `config/sic.yaml`. Key settings for runtime-optimized SIC:

**Top-level settings:**
- `device`: `auto` (auto-detect), `cpu`, or `cuda`
- `dataset`: `MNIST`, `FashionMNIST`, `CIFAR10`, or `SVHN`
- `model`: Model architecture name (e.g., `LeNet5Tanh`)

**SIC-specific settings:**
- `sic.mode`: `hybrid` (GPU-accelerated) or `classic` (CPU-only)
- `sic.rounding_decimals`: Precision for weight clustering (default: 32)
- `sic.max_passes`: Maximum number of SIC passes (default: 1)
- `sic.acceptance_from`: Source for acceptance subset (`"train+val"` recommended)
- `sic.eval_max`: Set to `null` to use full filtered acceptance set (recommended for runtime-optimized SIC)
- `sic.tiered_asc.enable`: Enable tiered acceptance (set to `true` for runtime optimization)
- `sic.tiered_asc.fast`: Fast tier size (default: 1024)
- `sic.tiered_asc.medium`: Medium tier size (default: 2048)
- `sic.enable_product_of_sums`: Enable PoS materialization for acceleration

**Pruning settings:**
- `prune.enabled`: Set to `false` for runtime-optimized SIC (focuses on speed rather than additional compression)

### Recommended Configuration for Runtime-Optimized SIC

```yaml
sic:
  acceptance_from: "train+val"
  eval_max: null  # Use full filtered acceptance set
  tiered_asc:
    enable: true
    fast: 1024
    medium: 2048

prune:
  enabled: false  # Disable pruning for runtime-optimized workflow
```

### Command-Line Arguments

You can override config file settings via command-line:

```bash
python -m scripts.run_sic \
  --config config/sic.yaml \
  --device cuda \
  --model LeNet5Tanh \
  --dataset MNIST
```

## Evaluation

SIC automatically evaluates the model at multiple stages:
1. **Baseline accuracy**: Before SIC optimization
2. **Pre-prune accuracy**: After SIC, before pruning (if enabled)
3. **Final accuracy**: After all optimizations

The evaluation results are printed to the console during execution.

### Manual Evaluation

To evaluate a SIC-optimized model:

```bash
python -m scripts.run_sic --config config/sic.yaml
```

The script will:
- Load the base model from `saved_models/{MODEL}_{DATASET}.pth`
- Apply SIC optimization
- Save the optimized model to `saved_models/SIC_{MODEL}_{DATASET}.pth`
- Print accuracy metrics at each stage

## Pre-trained Models

Pre-trained models should be placed in the `saved_models/` directory with the naming convention:
```
{MODEL}_{DATASET}.pth
```

Example: `LeNet5Tanh_MNIST.pth`

The repository includes example pre-trained models:
- `LeNet5Tanh_MNIST.pth`
- `LeNet5ELU_MNIST.pth`
- `SmallMNISTCNN_MNIST.pth`

SIC-optimized models are saved with the prefix `SIC_`:
- `SIC_LeNet5Tanh_MNIST.pth`

## Results

### Experimental Results

Our experiments demonstrate that runtime-optimized SIC achieves substantial speedups while maintaining accuracy and compression quality:

#### Runtime Reduction

- **Filtered-only SIC**: 78-93% reduction in preprocessing time compared to baseline SIC (5×-14× speedups)
- **Filtered + Tiered SIC**: Additional 60-89% reduction compared to filtered-only (overall 70-90% reduction from baseline)

Example speedups on small CNNs:
- SmallMNISTCNN / MNIST: 78% time reduction (158.67s → 34.77s)
- LeNet5-ELU / MNIST: 92% time reduction (659.92s → 53.41s)
- LeNet5-Tanh / MNIST: 93% time reduction (1396.51s → 96.27s)
- LeNet5-Tanh / Fashion-MNIST: 91% time reduction (6753.97s → 635.68s)

With tiered acceptance:
- LeNet5-ELU / MNIST: 89% additional reduction (568.40s → 64.31s)
- ResNet-16 / CIFAR-10: 60% additional reduction (2969.39s → 1198.05s)
- ResNet-16 / SVHN: 72% additional reduction (1667.94s → 461.41s)

#### Accuracy Preservation

- Test accuracy changes are within **~0.2 percentage points** of baseline SIC
- Most configurations show effectively identical accuracy (within 0.01-0.03 pp)
- Some settings show slight improvements under the tiered variant

#### Compression Quality

- Unique weight reductions remain close to baseline SIC (typically within ±15%)
- Layer-wise compression behavior is preserved: layers with many parameters still see the largest reductions
- Relative ranking of easy vs. hard to consolidate layers remains intact

### Output Files

SIC optimization produces several outputs:

**Model Weights:**
- Optimized model weights saved to `saved_models/SIC_{MODEL}_{DATASET}.pth`

**Profiling Reports:**
- **JSON Statistics**: Detailed profiling data saved to `profiling/{MODEL}_detailed_stats.json`
  - Layer-wise statistics
  - Clustering metrics
  - Memory usage
  - Timing breakdown
  - Unique weight counts before/after SIC
- **Excel Reports**: Optional Excel files with global statistics (if `io.write_full_weights_to_excel: true`)

**Console Output:**
The script prints:
- Initial model analysis (weight statistics)
- Baseline accuracy (before SIC)
- SIC progress and convergence
- Pre-prune accuracy (if pruning enabled)
- Final model analysis
- Final accuracy metrics

### Example Output Structure

```
INITIAL MODEL ANALYSIS
[Verification statistics...]

BASELINE ACCURACY
[EVAL] Test (before SIC): 98.50%
[EVAL] Val  (before SIC): 98.20%

RUNNING SIC HYBRID
[SIC clustering progress with tiered acceptance...]

PRE-PRUNE ACCURACY
[EVAL] Test (pre-prune): 98.50%
[EVAL] Val  (pre-prune): 98.20%

FINAL MODEL ANALYSIS
[Final verification statistics...]

FINAL ACCURACY
[EVAL] Test (final): 98.50%
[EVAL] Val  (final): 98.20%
```

## Configuration Reference

For detailed configuration options, see `config/info.txt` which contains comprehensive documentation of all available settings including:

- **Data & Training**: Batch size, validation split, data loading settings
- **SIC Parameters**: Clustering controls, acceptance subset configuration, GPU acceleration thresholds
- **Pruning Options**: Strategy selection, acceptance criteria
- **I/O Settings**: Model paths, profiling directories, logging options

## Project Structure

```
SIC/
├── config/
│   ├── sic.yaml          # Main configuration file
│   └── info.txt          # Configuration reference
├── data/                 # Dataset storage (auto-created)
│   └── MNIST/           # Downloaded datasets
├── saved_models/        # Pre-trained and SIC-optimized models
├── profiling/           # Profiling reports and statistics
├── scripts/
│   └── run_sic.py       # Main SIC execution script
└── src/
    └── ANG_SIC/
        ├── core/        # Core utilities (config, data, numeric)
        ├── models/      # Model architectures
        └── SIC/         # SIC algorithm implementation
```
