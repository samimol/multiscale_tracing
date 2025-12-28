# -*- coding: utf-8 -*-
"""
Workflow Configuration for Multiscale Tracing Pipeline

This module defines paths and configuration for the three-stage training pipeline:
1. Data generation (main_data_feedforward.py)
2. Feedforward network training (main_feedforward.py)
3. Recurrent network training (main.py)

Author: Sami
"""

import os
from pathlib import Path
from datetime import datetime

# ============================================================================
# Base Paths - Hierarchical Structure
# ============================================================================

# Project root directory (go up one level from config/)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data directory structure
DATA_DIR = PROJECT_ROOT / 'data' / 'feedforward'

# Models directory structure
MODELS_DIR = PROJECT_ROOT / 'models'
FEEDFORWARD_BLOB_DIR = MODELS_DIR / 'feedforward' / 'blob'
FEEDFORWARD_CURVE_DIR = MODELS_DIR / 'feedforward' / 'curve'
RECURRENT_FINAL_DIR = MODELS_DIR / 'recurrent' / 'final'
RECURRENT_CHECKPOINT_DIR = MODELS_DIR / 'recurrent' / 'checkpoints'

# Results directory structure
RESULTS_DIR = PROJECT_ROOT / 'results' / 'experiments'
LOGS_DIR = RESULTS_DIR / 'logs'

# ============================================================================
# Data Generation Configuration
# ============================================================================

class DataConfig:
    """Configuration for feedforward data generation."""
    
    # Output paths
    output_dir = DATA_DIR
    dataset_filename = 'feedforward_dataset.pkl'
    
    # Data generation parameters
    num_scales = 4
    grid_size_base = 18  # Base grid size for curves
    blob_grid_size = 72  # Grid size for blobs
    num_samples = 50000  # Number of training samples
    
    @classmethod
    def get_output_path(cls, batch_id=0):
        """Get output path for generated data.
        
        Args:
            batch_id (int): Batch identifier for parallel generation.
            
        Returns:
            Path: Full path to output file.
        """
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        if batch_id == 0:
            return cls.output_dir / cls.dataset_filename
        else:
            return cls.output_dir / f'data_batch_{batch_id}.pkl'

# ============================================================================
# Feedforward Network Training Configuration
# ============================================================================

class FeedforwardConfig:
    """Configuration for feedforward network training."""
    
    # Input/Output paths
    input_dir = DATA_DIR
    blob_output_dir = FEEDFORWARD_BLOB_DIR
    curve_output_dir = FEEDFORWARD_CURVE_DIR
    dataset_filename = 'feedforward_dataset.pkl'
    
    # Training parameters
    num_networks = 1  # Number of networks to train
    num_scales = 4
    learning_rate = 0.001
    epochs = 2
    batch_size = 256
    
    # Model naming
    blob_model_prefix = 'FF_blob'
    curve_model_prefix = 'FF_curve'
    
    @classmethod
    def get_input_path(cls):
        """Get path to input dataset.
        
        Returns:
            Path: Full path to dataset file.
        """
        return cls.input_dir / cls.dataset_filename
    
    @classmethod
    def get_output_paths(cls, network_id=0):
        """Get output paths for trained models.
        
        Args:
            network_id (int): Network identifier.
            
        Returns:
            tuple: (blob_model_path, curve_model_path)
        """
        cls.blob_output_dir.mkdir(parents=True, exist_ok=True)
        cls.curve_output_dir.mkdir(parents=True, exist_ok=True)
        blob_path = cls.blob_output_dir / f'{cls.blob_model_prefix}_{network_id}.pt'
        curve_path = cls.curve_output_dir / f'{cls.curve_model_prefix}_{network_id}.pt'
        return blob_path, curve_path
    
    @classmethod
    def check_data_exists(cls):
        """Check if required data exists.
        
        Returns:
            bool: True if data exists, False otherwise.
        """
        return cls.get_input_path().exists()

# ============================================================================
# Recurrent Network Training Configuration
# ============================================================================

class RecurrentConfig:
    """Configuration for recurrent network training."""
    
    # Input/Output paths
    feedforward_blob_dir = FEEDFORWARD_BLOB_DIR
    feedforward_curve_dir = FEEDFORWARD_CURVE_DIR
    output_dir = RECURRENT_FINAL_DIR
    checkpoint_dir = RECURRENT_CHECKPOINT_DIR
    
    # Training parameters
    num_networks = 1
    num_scales = 4
    grid_size = (3 ** (num_scales - 1)) * 4
    total_length = 9  # Maximum curve length
    trials = 55000
    
    # Model naming
    model_prefix = 'recurrent_network'
    checkpoint_prefix = 'checkpoint'
    
    @classmethod
    def get_feedforward_paths(cls, network_id=0):
        """Get paths to pretrained feedforward networks.
        
        Args:
            network_id (int): Network identifier.
            
        Returns:
            tuple: (blob_model_path, curve_model_path)
        """
        blob_path = cls.feedforward_blob_dir / f'{FeedforwardConfig.blob_model_prefix}_{network_id}.pt'
        curve_path = cls.feedforward_curve_dir / f'{FeedforwardConfig.curve_model_prefix}_{network_id}.pt'
        return blob_path, curve_path
    
    @classmethod
    def get_output_path(cls, network_id=0, timestamp=None):
        """Get output path for trained recurrent network.
        
        Args:
            network_id (int): Network identifier.
            timestamp (str, optional): Timestamp for unique naming.
            
        Returns:
            Path: Full path to output file.
        """
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return cls.output_dir / f'{cls.model_prefix}_{network_id}_{timestamp}.pt'
    
    @classmethod
    def get_checkpoint_path(cls, network_id=0, epoch=0):
        """Get checkpoint path for intermediate saving.
        
        Args:
            network_id (int): Network identifier.
            epoch (int): Training epoch.
            
        Returns:
            Path: Full path to checkpoint file.
        """
        cls.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return cls.checkpoint_dir / f'{cls.checkpoint_prefix}_{network_id}_epoch_{epoch}.pt'
    
    @classmethod
    def check_feedforward_exists(cls, network_id=0):
        """Check if required feedforward networks exist.
        
        Args:
            network_id (int): Network identifier.
            
        Returns:
            tuple: (blob_exists, curve_exists)
        """
        blob_path, curve_path = cls.get_feedforward_paths(network_id)
        return blob_path.exists(), curve_path.exists()

# ============================================================================
# Logging Configuration
# ============================================================================

class LogConfig:
    """Configuration for logging."""
    
    output_dir = LOGS_DIR
    
    @classmethod
    def get_log_path(cls, stage, timestamp=None):
        """Get log file path for a training stage.
        
        Args:
            stage (str): Training stage ('data', 'feedforward', 'recurrent').
            timestamp (str, optional): Timestamp for unique naming.
            
        Returns:
            Path: Full path to log file.
        """
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return cls.output_dir / f'{stage}_{timestamp}.log'

# ============================================================================
# Utility Functions
# ============================================================================

def create_directory_structure():
    """Create all necessary directories for the workflow."""
    directories = [
        DATA_DIR,
        FEEDFORWARD_BLOB_DIR,
        FEEDFORWARD_CURVE_DIR,
        RECURRENT_FINAL_DIR,
        RECURRENT_CHECKPOINT_DIR,
        RESULTS_DIR,
        LOGS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created/verified directory: {directory}")

def check_workflow_status():
    """Check the status of the entire workflow.
    
    Returns:
        dict: Status of each stage.
    """
    status = {
        'data_generated': DataConfig.get_output_path().exists(),
        'feedforward_trained': False,
        'recurrent_trained': False
    }
    
    # Check if feedforward networks exist
    blob_path, curve_path = FeedforwardConfig.get_output_paths(0)
    status['feedforward_trained'] = blob_path.exists() and curve_path.exists()
    
    # Check if any recurrent networks exist
    if RecurrentConfig.output_dir.exists():
        recurrent_files = list(RecurrentConfig.output_dir.glob(f'{RecurrentConfig.model_prefix}_*.pt'))
        status['recurrent_trained'] = len(recurrent_files) > 0
    
    return status

def print_workflow_status():
    """Print a formatted status report of the workflow."""
    status = check_workflow_status()
    
    print("\n" + "="*60)
    print("MULTISCALE TRACING WORKFLOW STATUS")
    print("="*60)
    
    stages = [
        ("1. Data Generation", status['data_generated']),
        ("2. Feedforward Training", status['feedforward_trained']),
        ("3. Recurrent Training", status['recurrent_trained'])
    ]
    
    for stage_name, completed in stages:
        status_symbol = "✓" if completed else "✗"
        status_text = "COMPLETED" if completed else "PENDING"
        print(f"{status_symbol} {stage_name}: {status_text}")
    
    print("="*60)
    
    # Print next step
    if not status['data_generated']:
        print("\nNext step: Run 'python main_data_feedforward.py'")
    elif not status['feedforward_trained']:
        print("\nNext step: Run 'python main_feedforward.py'")
    elif not status['recurrent_trained']:
        print("\nNext step: Run 'python main.py'")
    else:
        print("\n✓ All stages completed!")
    
    print()

if __name__ == '__main__':
    # Create directory structure
    create_directory_structure()
    
    # Print workflow status
    print_workflow_status()
