#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete Workflow Runner for Multiscale Tracing

This script orchestrates the entire training pipeline:
1. Data generation
2. Feedforward network training
3. Recurrent network training

Usage:
    # Run entire workflow
    python run_workflow.py --all
    
    # Run specific stage
    python run_workflow.py --stage 1  # Data generation only
    python run_workflow.py --stage 2  # Feedforward training only
    python run_workflow.py --stage 3  # Recurrent training only
    
    # Check workflow status
    python run_workflow.py --status

@author: Sami
"""

import argparse
import subprocess
import sys
from pathlib import Path

from workflow_config import (
    check_workflow_status,
    print_workflow_status,
    create_directory_structure,
    DataConfig,
    FeedforwardConfig,
    RecurrentConfig
)


def run_stage_1():
    """Run Stage 1: Data Generation."""
    print("\n" + "="*70)
    print("RUNNING STAGE 1: DATA GENERATION")
    print("="*70)
    
    result = subprocess.run(
        [sys.executable, "main_data_feedforward.py"],
        capture_output=False
    )
    
    if result.returncode != 0:
        print("\n❌ Stage 1 failed!")
        return False
    
    print("\n✓ Stage 1 completed successfully")
    return True


def run_stage_2():
    """Run Stage 2: Feedforward Network Training."""
    print("\n" + "="*70)
    print("RUNNING STAGE 2: FEEDFORWARD NETWORK TRAINING")
    print("="*70)
    
    # Check prerequisites
    if not FeedforwardConfig.check_data_exists():
        print("\n❌ Cannot run Stage 2: Data not found!")
        print("Please run Stage 1 first: python run_workflow.py --stage 1")
        return False
    
    result = subprocess.run(
        [sys.executable, "main_feedforward.py"],
        capture_output=False
    )
    
    if result.returncode != 0:
        print("\n❌ Stage 2 failed!")
        return False
    
    print("\n✓ Stage 2 completed successfully")
    return True


def run_stage_3():
    """Run Stage 3: Recurrent Network Training."""
    print("\n" + "="*70)
    print("RUNNING STAGE 3: RECURRENT NETWORK TRAINING")
    print("="*70)
    
    # Check prerequisites
    blob_exists, curve_exists = RecurrentConfig.check_feedforward_exists(0)
    if not (blob_exists and curve_exists):
        print("\n❌ Cannot run Stage 3: Feedforward networks not found!")
        print("Please run Stage 2 first: python run_workflow.py --stage 2")
        return False
    
    result = subprocess.run(
        [sys.executable, "main.py"],
        capture_output=False
    )
    
    if result.returncode != 0:
        print("\n❌ Stage 3 failed!")
        return False
    
    print("\n✓ Stage 3 completed successfully")
    return True


def run_all_stages():
    """Run all three stages sequentially."""
    print("\n" + "="*70)
    print("RUNNING COMPLETE WORKFLOW")
    print("="*70)
    
    # Create directory structure
    create_directory_structure()
    
    # Check current status
    status = check_workflow_status()
    
    # Stage 1: Data Generation
    if not status['data_generated']:
        print("\n→ Stage 1 needs to run")
        if not run_stage_1():
            return False
    else:
        print("\n✓ Stage 1 already completed (skipping)")
    
    # Stage 2: Feedforward Training
    if not status['feedforward_trained']:
        print("\n→ Stage 2 needs to run")
        if not run_stage_2():
            return False
    else:
        print("\n✓ Stage 2 already completed (skipping)")
    
    # Stage 3: Recurrent Training
    if not status['recurrent_trained']:
        print("\n→ Stage 3 needs to run")
        if not run_stage_3():
            return False
    else:
        print("\n✓ Stage 3 already completed (skipping)")
    
    print("\n" + "="*70)
    print("✓ COMPLETE WORKFLOW FINISHED SUCCESSFULLY")
    print("="*70)
    
    return True


def main():
    """Main entry point for workflow runner."""
    parser = argparse.ArgumentParser(
        description="Multiscale Tracing Workflow Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run entire workflow
  python run_workflow.py --all
  
  # Run specific stage
  python run_workflow.py --stage 1
  python run_workflow.py --stage 2
  python run_workflow.py --stage 3
  
  # Check status
  python run_workflow.py --status
        """
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all stages of the workflow'
    )
    
    parser.add_argument(
        '--stage',
        type=int,
        choices=[1, 2, 3],
        help='Run a specific stage (1=data, 2=feedforward, 3=recurrent)'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Check workflow status'
    )
    
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Create directory structure only'
    )
    
    args = parser.parse_args()
    
    # If no arguments, show status
    if not any([args.all, args.stage, args.status, args.setup]):
        print_workflow_status()
        print("\nUse --help for usage information")
        return
    
    # Handle setup
    if args.setup:
        print("\nCreating directory structure...")
        create_directory_structure()
        print("\n✓ Directory structure created")
        return
    
    # Handle status check
    if args.status:
        print_workflow_status()
        return
    
    # Handle stage execution
    success = True
    
    if args.all:
        success = run_all_stages()
    elif args.stage == 1:
        success = run_stage_1()
    elif args.stage == 2:
        success = run_stage_2()
    elif args.stage == 3:
        success = run_stage_3()
    
    # Print final status
    print("\n")
    print_workflow_status()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
