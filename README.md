This repository contains the code for training and evaluating a biologically inspired recurrent neural network that learns scale‐invariant curve tracing and object grouping. The model replicates key findings from human psychophysics and monkey visual cortex: it dynamically selects spatial scale and propagates enhanced activity through feedback and horizontal connections, enabling generalization from short curves to long contours and 2D shapes. The network is trained using a biologically plausible reinforcement learning rule and captures reaction time patterns seen in perceptual grouping tasks.

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd multiscale_tracing
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Option 1: Run Complete Workflow 

Run all three stages automatically:

```bash
cd scripts
python run_workflow.py --all
```

This will:
1. Generate synthetic training data
2. Train feedforward networks for blob and curve detection
3. Train the recurrent network with reinforcement learning

### Option 2: Run Individual Stages

```bash
# Stage 1: Generate training data 
python scripts/train/01_generate_data.py

# Stage 2: Train feedforward networks 
python scripts/train/02_train_feedforward.py

# Stage 3: Train recurrent network
python scripts/train/03_train_recurrent.py
```

## Configuration

### Workflow Configuration

Edit `config/workflow_config.py` to customize:

```python
# Data generation
DataConfig.num_scales = 4          # Number of spatial scales
DataConfig.num_samples = 50000     # Training samples

# Feedforward training
FeedforwardConfig.epochs = 80      # Training epochs
FeedforwardConfig.learning_rate = 0.001
FeedforwardConfig.batch_size = 256

# Recurrent training
RecurrentConfig.grid_size = 36     # Spatial grid size
RecurrentConfig.trials = 50000     # Training trials
RecurrentConfig.total_length = 8   # Maximum curve length
```

### Model Configuration

Edit `config/model_config.py` for command-line arguments:

```bash
python scripts/train/02_train_feedforward.py --num_networks 5
python scripts/train/03_train_recurrent.py --num_scales 4 --total_length 8
```

## Training Pipeline

### Stage 1: Data Generation

Generates synthetic training data with:
- Curves at multiple spatial scales
- Blob/object shapes using Bezier curves
- Multi-scale labels for supervised learning

**Output**: `data/feedforward/feedforward_dataset.pkl`

### Stage 2: Feedforward Network Training

Trains convolutional networks for:
- **Blob detection**: Scale selection for 2D objects
- **Curve detection**: Scale selection for curves

**Output**: 
- `models/feedforward/blob/FF_blob_0.pt`
- `models/feedforward/curve/FF_curve_0.pt`

### Stage 3: Recurrent Network Training

Trains hierarchical recurrent network with:
- Pretrained feedforward features
- Reinforcement learning for attention
- VIP/SOM interneuron modulation
- Multi-scale recurrent dynamics

**Output**: `models/recurrent/final/recurrent_network_0_TIMESTAMP.pt`

## Usage Examples

### Training Multiple Networks

```bash
# Train 5 feedforward networks
python scripts/train/02_train_feedforward.py --num_networks 5

# Train 5 recurrent networks
python scripts/train/03_train_recurrent.py --num_networks 5
```

### Running Specific Stages

```bash
cd scripts

# Run only data generation
python run_workflow.py --stage 1

# Run only feedforward training
python run_workflow.py --stage 2

# Run only recurrent training
python run_workflow.py --stage 3
```


## Citation

If you use this code in your research, please cite:

```bibtex
@article {Mollard2024.06.17.599272,
	author = {Mollard, Sami and Bohte, Sander M. and Roelfsema, Pieter R.},
	title = {How the visual brain can learn to parse images using a multiscale, incremental grouping process},
	elocation-id = {2024.06.17.599272},
	year = {2025},
	doi = {10.1101/2024.06.17.599272},
	publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2025/04/23/2024.06.17.599272},
	eprint = {https://www.biorxiv.org/content/early/2025/04/23/2024.06.17.599272.full.pdf},
	journal = {bioRxiv}
}

```