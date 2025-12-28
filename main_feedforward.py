# -*- coding: utf-8 -*-
"""
Stage 2: Feedforward Network Training

This script trains feedforward networks for blob and curve detection
at multiple spatial scales using the generated data.

Usage:
    python main_feedforward.py [--num_networks N]

Inputs:
    - Requires: results/feedforward_data/feedforward_dataset.pkl

Outputs:
    - Saves models to: results/feedforward_networks/
      - FF_blob_0.pt, FF_blob_1.pt, ...
      - FF_curve_0.pt, FF_curve_1.pt, ...

@author: Sami
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
import datetime
import pickle
import sys
from pathlib import Path

# Import project modules
from config import parser
from feedforward_network import train_feedforward_blob, train_feedforward_curve
from workflow_config import (
    FeedforwardConfig, 
    create_directory_structure, 
    print_workflow_status
)


def load_dataset():
    """Load the feedforward dataset.
    
    Returns:
        dict: Dataset containing blob and curve data.
    """
    input_path = FeedforwardConfig.get_input_path()
    
    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {input_path}\n"
            f"Please run 'python main_data_feedforward.py' first."
        )
    
    print(f"Loading dataset from: {input_path}")
    with open(input_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print("✓ Dataset loaded successfully\n")
    return dataset


def prepare_data(feedforward_dataset, num_scales):
    """Prepare blob and curve data for training.
    
    Args:
        feedforward_dataset: Raw dataset.
        num_scales (int): Number of spatial scales.
        
    Returns:
        tuple: (input_blob, labels_blob, input_curve, labels_curve)
    """
    # Extract blob data
    input_blob = feedforward_dataset[1][0][0]
    labels_blob = feedforward_dataset[1][0][1]
    
    # Extract curve data from multiple scales
    input_curve = []
    labels_curve = [[] for i in range(num_scales-1)]
    
    for index_1, index_2 in enumerate([0, 2, 3]):
        inputt = feedforward_dataset[index_2][0][0]
        labels_curve_interm = feedforward_dataset[index_2][0][1]
        
        for p in range(len(labels_curve_interm)):
            labels_curve[p].append(labels_curve_interm[p])
        input_curve.append(inputt)
    
    return input_blob, labels_blob, input_curve, labels_curve


def main():
    """Train feedforward networks for blob and curve detection."""
    
    # Setup
    print("\n" + "="*60)
    print("STAGE 2: FEEDFORWARD NETWORK TRAINING")
    print("="*60 + "\n")
    
    # Create directory structure
    create_directory_structure()
    
    # Check if data exists
    if not FeedforwardConfig.check_data_exists():
        print("❌ ERROR: Dataset not found!")
        print("Please run 'python main_data_feedforward.py' first.\n")
        sys.exit(1)
    
    # Parse arguments
    args = parser.parse_args()
    num_networks = args.num_networks if hasattr(args, 'num_networks') else FeedforwardConfig.num_networks
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    seed = datetime.datetime.now().microsecond
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Random seed: {seed}\n")
    
    # Load and prepare data
    feedforward_dataset = load_dataset()
    input_blob, labels_blob, input_curve, labels_curve = prepare_data(
        feedforward_dataset, 
        FeedforwardConfig.num_scales
    )
    
    # Free memory
    del feedforward_dataset
    
    print(f"Training {num_networks} network(s)...\n")
    
    # Train networks
    for i in range(num_networks):
        print(f"Training network {i+1}/{num_networks}")
        print("-" * 40)
        
        # Train blob detection network
        print("Training blob detection network...")
        feedforward_blob = train_feedforward_blob(
            FeedforwardConfig.num_scales,
            input_blob,
            labels_blob,
            device
        )
        
        # Train curve detection network
        print("Training curve detection network...")
        feedforward_curve = train_feedforward_curve(
            FeedforwardConfig.num_scales,
            input_curve,
            labels_curve,
            device
        )
        
        # Save models
        blob_path, curve_path = FeedforwardConfig.get_output_paths(i)
        
        print(f"Saving blob model to: {blob_path}")
        torch.save(feedforward_blob, blob_path)
        
        print(f"Saving curve model to: {curve_path}")
        torch.save(feedforward_curve, curve_path)
        
        print(f"✓ Network {i+1} trained and saved\n")
    
    # Print next steps
    print("="*60)
    print("NEXT STEP: Train recurrent network")
    print("Run: python main.py")
    print("="*60 + "\n")
    
    # Update workflow status
    print_workflow_status()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
