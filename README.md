# Multiscale Tracing

A hierarchical recurrent neural network for visual attention and curve tracing tasks.

## Project Structure

```
multiscale_tracing/
├── config/                     # Configuration files
│   ├── workflow_config.py     # Workflow paths and settings
│   └── model_config.py        # Model hyperparameters
│
├── src/                       # Source code
│   ├── models/               # Neural network models
│   ├── data/                 # Data generation
│   ├── tasks/                # Task definitions
│   └── utils/                # Utilities
│
├── scripts/                   # Executable scripts
│   ├── train/                # Training scripts
│   │   ├── 01_generate_data.py
│   │   ├── 02_train_feedforward.py
│   │   └── 03_train_recurrent.py
│   └── run_workflow.py       # Complete workflow
│
├── data/                      # Generated datasets
├── models/                    # Saved models
├── results/                   # Experiment results
└── docs/                      # Documentation
```

## Quick Start

### Run Complete Workflow

```bash
cd scripts
python run_workflow.py --all
```

### Run Individual Stages

```bash
# Stage 1: Generate data
python scripts/train/01_generate_data.py

# Stage 2: Train feedforward networks
python scripts/train/02_train_feedforward.py

# Stage 3: Train recurrent network
python scripts/train/03_train_recurrent.py
```

### Check Status

```bash
cd scripts
python run_workflow.py --status
```

## Configuration

Edit `config/workflow_config.py` to customize:
- Data generation parameters
- Training hyperparameters
- File paths and directories

## Documentation

See `docs/` folder for detailed documentation:
- `WORKFLOW_SUMMARY.md` - Complete workflow guide
- `INLINE_COMMENTS_SUMMARY.md` - Code documentation
- `DOCUMENTATION_SUMMARY.md` - API documentation

## Requirements

```bash
pip install torch numpy scipy scikit-image
```

## Citation

If you use this code, please cite our work.
