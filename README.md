# Multiscale Tracing

A hierarchical recurrent neural network for visual attention and curve tracing tasks. This project implements a three-stage training pipeline with multi-scale feedforward and recurrent architectures for learning complex visual attention patterns.

## Features

- **Multi-scale Architecture**: Hierarchical processing at multiple spatial scales
- **Feedforward Networks**: Convolutional networks for blob and curve detection
- **Recurrent Networks**: Biologically-inspired recurrent dynamics with VIP/SOM interneurons
- **Reinforcement Learning**: Reward-based learning for attention and tracing tasks
- **Professional Workflow**: Automated pipeline with configuration management

## Project Structure

```
multiscale_tracing/
├── config/                          # Configuration files
│   ├── workflow_config.py          # Workflow paths and settings
│   └── model_config.py             # Model hyperparameters
│
├── src/                            # Source code (reusable library)
│   ├── models/                     # Neural network models
│   │   ├── layers.py              # Custom layer implementations
│   │   ├── feedforward_network.py # Feedforward architecture
│   │   └── recurrent_network.py   # Recurrent architecture
│   ├── data/                       # Data generation
│   │   └── feedforward_data.py    # Synthetic dataset generation
│   ├── tasks/                      # Task definitions
│   │   └── tasks.py               # Curve tracing tasks
│   └── utils/                      # Utilities
│       └── helper_functions.py    # Helper functions
│
├── scripts/                        # Executable scripts
│   ├── train/                      # Training scripts
│   │   ├── 01_generate_data.py    # Stage 1: Data generation
│   │   ├── 02_train_feedforward.py # Stage 2: Feedforward training
│   │   └── 03_train_recurrent.py  # Stage 3: Recurrent training
│   └── run_workflow.py            # Complete workflow orchestrator
│
├── data/                           # Generated datasets (not in git)
│   └── feedforward/
│
├── models/                         # Saved models (not in git)
│   ├── feedforward/
│   │   ├── blob/                  # Blob detection models
│   │   └── curve/                 # Curve detection models
│   └── recurrent/
│       ├── checkpoints/           # Training checkpoints
│       └── final/                 # Final trained models
│
├── results/                        # Experiment results (not in git)
│   └── experiments/
│       └── logs/
│
├── notebooks/                      # Jupyter notebooks for analysis
├── tests/                          # Unit tests
├── docs/                           # Documentation
│   ├── WORKFLOW_SUMMARY.md
│   ├── INLINE_COMMENTS_SUMMARY.md
│   └── DOCUMENTATION_SUMMARY.md
│
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

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

### Option 1: Run Complete Workflow (Recommended)

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
# Stage 1: Generate training data (~15 min)
python scripts/train/01_generate_data.py

# Stage 2: Train feedforward networks (~1-2 hours)
python scripts/train/02_train_feedforward.py

# Stage 3: Train recurrent network (~4-8 hours)
python scripts/train/03_train_recurrent.py
```

### Check Workflow Status

```bash
cd scripts
python run_workflow.py --status
```

Output:
```
============================================================
MULTISCALE TRACING WORKFLOW STATUS
============================================================
✓ 1. Data Generation: COMPLETED
✓ 2. Feedforward Training: COMPLETED
✗ 3. Recurrent Training: PENDING
============================================================
Next step: Run 'python train/03_train_recurrent.py'
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
- **Blob detection**: Recognizing object shapes
- **Curve detection**: Identifying curve segments at multiple scales

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

### Custom Configuration

```bash
# Different number of scales
python scripts/train/03_train_recurrent.py --num_scales 3

# Different curve length
python scripts/train/03_train_recurrent.py --total_length 10
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
@article{multiscale_tracing,
  title={Multiscale Hierarchical Recurrent Networks for Visual Attention},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: [your-email@domain.com]

## Acknowledgments

This project implements hierarchical recurrent networks for visual attention and curve tracing tasks, inspired by biological visual processing mechanisms.
