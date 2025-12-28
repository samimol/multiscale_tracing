# Multiscale Tracing Workflow Guide

## Overview

This project implements a three-stage training pipeline for a hierarchical recurrent neural network that performs visual attention and curve tracing tasks.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STAGE 1: DATA GENERATION                  │
│                                                               │
│  Script: main_data_feedforward.py                            │
│  Output: results/feedforward_data/feedforward_dataset.pkl    │
│                                                               │
│  Generates synthetic training data:                          │
│  - Curves at multiple scales                                 │
│  - Blob/object shapes                                        │
│  - Multi-scale labels                                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              STAGE 2: FEEDFORWARD NETWORK TRAINING           │
│                                                               │
│  Script: main_feedforward.py                                 │
│  Input:  results/feedforward_data/feedforward_dataset.pkl    │
│  Output: results/feedforward_networks/                       │
│          - FF_blob_0.pt                                      │
│          - FF_curve_0.pt                                     │
│                                                               │
│  Trains convolutional networks for:                          │
│  - Blob/object detection                                     │
│  - Curve detection                                           │
│  - Multi-scale feature extraction                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│             STAGE 3: RECURRENT NETWORK TRAINING              │
│                                                               │
│  Script: main.py                                             │
│  Input:  results/feedforward_networks/                       │
│          - FF_blob_0.pt                                      │
│          - FF_curve_0.pt                                     │
│  Output: results/recurrent_networks/                         │
│          - recurrent_network_0_TIMESTAMP.pt                  │
│                                                               │
│  Trains hierarchical recurrent network with:                 │
│  - Pretrained feedforward features                           │
│  - Reinforcement learning                                    │
│  - VIP/SOM interneuron modulation                            │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Option 1: Run Complete Workflow

```bash
# Run all three stages automatically
python run_workflow.py --all
```

### Option 2: Run Stages Individually

```bash
# Stage 1: Generate data
python main_data_feedforward.py

# Stage 2: Train feedforward networks
python main_feedforward.py

# Stage 3: Train recurrent network
python main.py
```

### Option 3: Check Status

```bash
# Check which stages are completed
python run_workflow.py --status
```

## Detailed Usage

### Stage 1: Data Generation

**Purpose:** Generate synthetic training data for feedforward networks.

```bash
python main_data_feedforward.py
```

**Configuration:** Edit `workflow_config.py` → `DataConfig`
- `num_scales`: Number of spatial scales (default: 4)
- `num_samples`: Number of training samples (default: 50,000)
- `grid_size_base`: Base grid size for curves (default: 18)

**Output:**
- `results/feedforward_data/feedforward_dataset.pkl`

---

### Stage 2: Feedforward Network Training

**Purpose:** Train convolutional networks for multi-scale feature detection.

```bash
python main_feedforward.py [--num_networks N]
```

**Arguments:**
- `--num_networks N`: Number of networks to train (default: 1)

**Configuration:** Edit `workflow_config.py` → `FeedforwardConfig`
- `num_scales`: Number of scales (default: 4)
- `learning_rate`: Learning rate (default: 0.001)
- `epochs`: Training epochs (default: 80)
- `batch_size`: Batch size (default: 256)

**Output:**
- `results/feedforward_networks/FF_blob_0.pt`
- `results/feedforward_networks/FF_curve_0.pt`

---

### Stage 3: Recurrent Network Training

**Purpose:** Train hierarchical recurrent network with reinforcement learning.

```bash
python main.py [--num_networks N] [--total_length L] [--num_scales S]
```

**Arguments:**
- `--num_networks N`: Number of networks to train (default: 1)
- `--total_length L`: Maximum curve length (default: 8)
- `--num_scales S`: Number of hierarchical scales (default: 4)

**Configuration:** Edit `workflow_config.py` → `RecurrentConfig`
- `grid_size`: Spatial grid size (default: 36)
- `trials`: Number of training trials (default: 50,000)

**Output:**
- `results/recurrent_networks/recurrent_network_0_TIMESTAMP.pt`

---

## Directory Structure

```
multiscale_tracing/
├── workflow_config.py          # Central configuration
├── run_workflow.py             # Workflow orchestrator
├── main_data_feedforward.py    # Stage 1 script
├── main_feedforward.py         # Stage 2 script
├── main.py                     # Stage 3 script
│
├── results/                    # All outputs
│   ├── feedforward_data/       # Generated datasets
│   ├── feedforward_networks/   # Trained feedforward models
│   ├── recurrent_networks/     # Trained recurrent models
│   └── logs/                   # Training logs
│
├── layers.py                   # Neural network layers
├── feedforward_network.py      # Feedforward architecture
├── recurrent_network.py        # Recurrent architecture
├── tasks.py                    # Task definitions
├── feedforward_data.py         # Data generation
└── helper_functions.py         # Utilities
```

## Configuration Management

All paths and parameters are centralized in `workflow_config.py`:

```python
from workflow_config import DataConfig, FeedforwardConfig, RecurrentConfig

# Access configuration
data_path = DataConfig.get_output_path()
ff_blob_path, ff_curve_path = FeedforwardConfig.get_output_paths(0)
rec_path = RecurrentConfig.get_output_path(0)

# Check status
data_exists = DataConfig.get_output_path().exists()
blob_exists, curve_exists = RecurrentConfig.check_feedforward_exists(0)
```

## Workflow Runner Commands

```bash
# Check workflow status
python run_workflow.py --status

# Run complete workflow
python run_workflow.py --all

# Run specific stage
python run_workflow.py --stage 1  # Data generation
python run_workflow.py --stage 2  # Feedforward training
python run_workflow.py --stage 3  # Recurrent training

# Setup directories only
python run_workflow.py --setup

# Get help
python run_workflow.py --help
```

## Citation

If you use this code, please cite:

```bibtex
@article{multiscale_tracing,
  title={Multiscale Hierarchical Recurrent Networks for Visual Attention},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

