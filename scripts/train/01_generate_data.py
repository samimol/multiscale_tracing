# -*- coding: utf-8 -*-
"""
Stage 1: Feedforward Data Generation

This script generates synthetic training data for feedforward networks.
The data includes curves and blobs at multiple spatial scales.

Usage:
    python main_data_feedforward.py

Outputs:
    - Saves dataset to: results/feedforward_data/feedforward_dataset.pkl

@author: Sami
"""

import torch
import numpy as np
import random
import os
import datetime
import pickle
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from src.data.feedforward_data import make_data_feedforward
from config.workflow_config import DataConfig, create_directory_structure, print_workflow_status


def main():
    """Generate feedforward training data."""
    
    # Setup
    print("\n" + "="*60)
    print("STAGE 1: FEEDFORWARD DATA GENERATION")
    print("="*60 + "\n")
    
    # Create directory structure
    create_directory_structure()
    
    # Set device
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Get batch ID (for parallel processing on clusters)
    if os.name == 'nt':
        batch_id = 0
    else:
        batch_id = int(os.environ.get("SLURM_PROCID", 0))
    
    print(f"Batch ID: {batch_id}")
    
    # Set random seed for reproducibility
    seed = int(batch_id) + datetime.datetime.now().microsecond
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Random seed: {seed}\n")
    
    # Generate data
    print(f"Generating data with {DataConfig.num_scales} scales...")
    print(f"  - Curve samples: {DataConfig.num_samples}")
    print(f"  - Blob samples: {DataConfig.num_samples}")
    
    data = make_data_feedforward(device, DataConfig.num_scales)
    
    print("Data generation complete\n")
    
    # Save data
    output_path = DataConfig.get_output_path(batch_id)
    print(f"Saving dataset to: {output_path}")
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    
    print("✓ Dataset saved successfully\n")
    
    # Print next steps
    print("="*60)
    print("NEXT STEP: Train feedforward networks")
    print("Run: python main_feedforward.py")
    print("="*60 + "\n")
    
    # Update workflow status
    print_workflow_status()


if __name__ == '__main__':
    main()

