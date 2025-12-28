# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 16:05:47 2023

@author: Sami
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
import random
from src.utils.helper_functions import make_curves, make_blob



def make_dataset_curve(grid_size, big_pixel_size, other_big_pixel_size, num_trial, device):
    """Generate synthetic dataset of curves for feedforward network training.
    
    Creates a dataset of random curves at multiple spatial scales. Each sample contains
    curves with start/end markers (different features) and intermediate points. The
    function generates labels indicating whether each spatial region contains a valid
    straight curve segment.
    
    Args:
        grid_size (int): Size of the spatial grid (e.g., 72).
        big_pixel_size (int): Size of target scale pixels (e.g., 3, 9, 27).
        other_big_pixel_size (list): Sizes of other scales for multi-scale labels.
        num_trial (int): Number of training examples to generate.
        device (torch.device): Device to place tensors on (CPU or CUDA).
        
    Returns:
        tuple: (input_list, labels, labels_other)
            - input_list: Tensor of shape (num_trial, 3, grid_size, grid_size)
                         3 features: [start_marker, curve_body, end_marker]
            - labels: Binary labels for target scale indicating straight curves
            - labels_other: List of labels for other scales
    """
    # Calculate grid dimensions at different scales
    grid_size = grid_size
    big_pixel_size = big_pixel_size
    big_grid_size = grid_size // big_pixel_size  # Number of big pixels in each dimension
    other_big_grid_size = [grid_size // other_big_pixel_size[i] for i in range(len(other_big_pixel_size))]
    num_trial = num_trial
    
    # Feature channels: 0=start marker, 1=curve body, 2=end marker (or vice versa)
    feature_number = 3
    
    # Initialize storage tensors
    input_list = torch.zeros((num_trial,feature_number,grid_size,grid_size),device=device)
    labels = torch.zeros((num_trial,1,big_grid_size,big_grid_size),device=device)
    labels_other = [torch.zeros((num_trial,1,other_big_grid_size[i],other_big_grid_size[i]),device=device) for i in range(len(other_big_pixel_size))]
    # Generate each training sample
    for k in range(num_trial):
          # Progress indicator
          if k%100 == 0:
            print(k)
          
          # Initialize input and label for this sample
          input = torch.zeros((feature_number,grid_size,grid_size),device=device)
          label = torch.zeros((1,big_grid_size,big_grid_size),device=device)
          
          # Iterate over each spatial region (big pixel) in the grid
          for i_init in range(0,grid_size,big_pixel_size):
            for j_init in range(0,grid_size,big_pixel_size):
              twocurves = False  # Flag to track if two curves are generated in this region
              
              # Kernel represents the current spatial region
              kernel = torch.zeros((1,feature_number,big_pixel_size,big_pixel_size))
              
              # Randomly choose curve length (0 means no curve, or 3 to 3*big_pixel_size-2)
              curvelength = np.random.choice([0]+list(range(3,3*big_pixel_size-2)))
              prob = np.random.rand()
              direction = np.random.choice([0,1])  # 0={up,down}, 1={left,right}  
              # Generate curve(s) if length is non-zero
              if curvelength != 0:
                # Retry loop to handle potential IndexErrors from curve generation
                while True:
                  try:
                    # Case 1: Short curves with constrained direction (50% probability)
                    if prob < 0.5 and curvelength in list(range(2,big_pixel_size+1)):
                        mask = np.zeros((big_pixel_size, big_pixel_size))
                        # Generate first curve with specified direction constraint
                        curve1, mask1 = make_curves([], mask,curvelength,grid_size=big_pixel_size,direction=direction)
                        # 50% chance to add a second curve with same direction
                        if np.random.rand() < 0.5 and big_pixel_size > 3:
                            curve2, mask2 = make_curves([], mask1,curvelength,grid_size=big_pixel_size,direction=direction)
                            twocurves = True                            
                    # Case 2: Longer curves without direction constraint
                    else:
                        mask = np.zeros((big_pixel_size, big_pixel_size))
                        # Generate first curve without direction constraint
                        curve1, mask1 = make_curves([], mask,curvelength,grid_size=big_pixel_size)
                        # 50% chance to add a second curve
                        if np.random.rand() < 0.5: 
                            if curvelength > 2 and curvelength < big_pixel_size*2 and big_pixel_size > 3:
                                curve2, mask2 = make_curves([], mask1,curvelength,grid_size=big_pixel_size)
                                twocurves = True
                            # Special case for small grid: remove one point from curve
                            elif  curvelength > 2 and curvelength < big_pixel_size*2 and big_pixel_size == 3:
                                curve1.pop(random.randrange(len(curve1))) 
                    # Place first curve into kernel
                    # Intermediate points go into feature channel 1 (curve body)
                    for i in range(1,len(curve1)-1):
                      kernel[0,1,curve1[i]%big_pixel_size,curve1[i]//big_pixel_size] = 1
                    
                    # Start point: randomly assign to feature 0, 1, or 2
                    feat1 = np.random.choice([0,1,-1])
                    kernel[0,feat1,curve1[0]%big_pixel_size,curve1[0]//big_pixel_size] = 1
                    
                    # End point: assign to a different feature than start
                    feat2 = [0,1,-1]
                    feat2.remove(feat1)
                    kernel[0,np.random.choice(feat2),curve1[-1]%big_pixel_size,curve1[-1]//big_pixel_size] = 1
                    
                    # If second curve exists, add it to the kernel
                    if twocurves:
                        # Intermediate points of second curve
                        for i in range(1,len(curve2)-1):
                          kernel[0,1,curve2[i]%big_pixel_size,curve2[i]//big_pixel_size] = 1
                        # Start point of second curve
                        feat1 = np.random.choice([0,1,-1])
                        kernel[0,feat1,curve2[0]%big_pixel_size,curve2[0]//big_pixel_size] = 1
                        # End point of second curve
                        feat2 = [0,1,-1]
                        feat2.remove(feat1)
                        kernel[0,np.random.choice(feat2),curve2[-1]%big_pixel_size,curve2[-1]//big_pixel_size] = 1    
                    break  # Successfully generated curve(s)
                  except IndexError:
                      pass  # Retry if curve generation failed
                # Generate label: 1 if single straight curve, 0 otherwise
                if curvelength > 0:
                    if twocurves == False:
                        # Check if curve is straight (all differences are 1 horizontally or vertically)
                        # np.diff(curve1)==1: horizontal straight line
                        # np.diff(curve1)==big_pixel_size: vertical straight line
                        if (np.all(np.abs(np.diff(curve1))==1) or np.all(np.abs(np.diff(curve1))==big_pixel_size)):
                          label[0,i_init//big_pixel_size,j_init//big_pixel_size] = 1  # Straight curve
                        else:
                          label[0,i_init//big_pixel_size,j_init//big_pixel_size] = 0.  # Non-straight curve
                    # Place kernel into the appropriate region of the input
                    input[:,i_init:i_init+big_pixel_size,j_init:j_init+big_pixel_size] = kernel
          # Collapse features to get all curve points (sum across feature dimension)
          input_collapsed = torch.sum(input,dim=0)
          
          # Generate labels for other scales
          for p in range(len(other_big_pixel_size)):
              label_other = torch.zeros((1,other_big_grid_size[p],other_big_grid_size[p]),device=device)
              # Iterate over regions at this scale
              for i_init_other in range(0,grid_size,other_big_pixel_size[p]):
                for j_init_other in range(0,grid_size,other_big_pixel_size[p]):
                    # Extract region from collapsed input
                    kernel = input_collapsed[i_init_other:i_init_other+other_big_pixel_size[p],j_init_other:j_init_other+other_big_pixel_size[p]]
                    non_zero_kernel = torch.nonzero(kernel)
                    
                    # Check if region contains a straight line
                    if len(non_zero_kernel) != 0:
                        # Horizontal line: all points have same row, consecutive columns
                        horizontal_line = (torch.all(non_zero_kernel[:,0] == non_zero_kernel[:,0][0]) and 
                                         torch.all(torch.diff(non_zero_kernel[:,1]) == 1))
                        # Vertical line: all points have same column, consecutive rows
                        vertical_line = (torch.all(non_zero_kernel[:,1] == non_zero_kernel[:,1][0]) and 
                                       torch.all(torch.diff(non_zero_kernel[:,0]) == 1))
                        
                        if horizontal_line or vertical_line:
                            label_other[0,i_init_other//other_big_pixel_size[p],j_init_other//other_big_pixel_size[p]] = 1
              
              # Store labels for this scale
              labels_other[p][k,:,:,:] = label_other
          
          input_list[k,:,:,:] = input
          labels[k,:,:,:] = label
    return(input_list,labels,labels_other)


def make_dataset_blob(grid_size, big_pixel_size, other_big_pixel_size, num_trial, device):
    """Generate synthetic dataset of blobs/objects for feedforward network training.
    
    Creates a dataset of blob-like objects with start and target markers. Each sample
    contains a blob shape (generated using Bezier curves) with a green start marker
    and optionally a blue target marker. Labels indicate regions where the blob
    occupies exactly one pixel per location (object detection task).
    
    Args:
        grid_size (int): Size of the spatial grid (e.g., 72).
        big_pixel_size (int): Size of target scale pixels (e.g., 3, 9, 27).
        other_big_pixel_size (list): Sizes of other scales for multi-scale labels.
        num_trial (int): Number of training examples to generate.
        device (torch.device): Device to place tensors on (CPU or CUDA).
        
    Returns:
        tuple: (input_list, labels, labels_other)
            - input_list: Tensor of shape (num_trial, 3, grid_size, grid_size)
                         3 features: [blue_target, red_blob, green_start]
            - labels: Binary labels indicating blob-occupied regions
            - labels_other: List of labels for other scales
    """
    # Calculate grid dimensions at different scales
    grid_size = grid_size
    big_pixel_size = big_pixel_size
    big_grid_size = grid_size // big_pixel_size  # Number of big pixels in each dimension
    other_big_grid_size = [grid_size // other_big_pixel_size[i] for i in range(len(other_big_pixel_size))]
    num_trial = num_trial
    
    # Feature channels: 0=blue (target), 1=red (blob body), 2=green (start)
    feature_number = 3
    
    # Initialize storage tensors
    input_list = torch.zeros((num_trial,feature_number,grid_size,grid_size),device=device)
    labels = torch.zeros((num_trial,1,big_grid_size,big_grid_size),device=device)
    labels_other = [torch.zeros((num_trial,1,other_big_grid_size[i],other_big_grid_size[i]),device=device) for i in range(len(other_big_pixel_size))]
    # Generate each training sample
    for k in range(num_trial):
          # Initialize input and label for this sample
          input = torch.zeros((feature_number,grid_size,grid_size),device=device)
          label = torch.zeros((1,big_grid_size,big_grid_size),device=device)
          
          # Random position for blob (not used in current implementation)
          i_init = np.random.choice(list(range(0,grid_size,big_pixel_size)))
          j_init = np.random.choice(list(range(0,grid_size,big_pixel_size)))
          
          # Retry loop to handle potential IndexErrors from blob generation
          while True:
            try:
              # Generate blob shape centered at (grid_size//2, grid_size//2)
              x_blob,y_blob = make_blob(grid_size,grid_size//2,grid_size//2)
              
              # Randomly select two points on the blob: start and target
              ind = np.random.randint(len(x_blob),size=2)
              x0 = x_blob[ind[0]]      # Start point x-coordinate
              y0 = y_blob[ind[0]]      # Start point y-coordinate
              x_target = x_blob[ind[1]]  # Target point x-coordinate
              y_target = y_blob[ind[1]]  # Target point y-coordinate
              
              # Place blob in red channel (feature 1)
              input[1,x_blob,y_blob] = 1
              
              # Remove start point from blob and mark it in green channel (feature 2)
              input[1,x0,y0] = 0
              input[2,x0,y0] = 1
              
              # 50% chance to add a blue target marker
              if np.random.rand() < 0.5:
                input[0,x_target,y_target] = 1  # Blue target marker
                input[1,x_target,y_target] = 0  # Remove from blob
              
              break  # Successfully generated blob
            except IndexError:
                      pass  # Retry if blob generation failed
          # Generate labels for target scale
          # Label is 1 if region contains exactly one pixel per location (blob occupies region)
          for i_init in range(0,grid_size,big_pixel_size):
            for j_init in range(0,grid_size,big_pixel_size):
                # Sum across feature channels to get total occupancy per pixel
                # Label=1 if all pixels in region have exactly 1 feature active
                if torch.all(torch.sum(input[:,i_init:i_init+big_pixel_size,j_init:j_init+big_pixel_size],dim=0)==1):
                    label[0,i_init//big_pixel_size,j_init//big_pixel_size] = 1.  # Blob region
                else:
                    label[0,i_init//big_pixel_size,j_init//big_pixel_size] = 0.  # Non-blob region
          # Generate labels for other scales
          for p in range(len(other_big_pixel_size)):
              label_other = torch.zeros((1,other_big_grid_size[p],other_big_grid_size[p]),device=device)
              # Iterate over regions at this scale
              for i_init in range(0,grid_size,other_big_pixel_size[p]):
                for j_init in range(0,grid_size,other_big_pixel_size[p]):
                    # Check if region contains blob (all pixels have exactly 1 feature)
                    if torch.all(torch.sum(input[:,i_init:i_init+other_big_pixel_size[p],j_init:j_init+other_big_pixel_size[p]],dim=0)==1):
                        label_other[0,i_init//other_big_pixel_size[p],j_init//other_big_pixel_size[p]] = 1.  # Blob region
                    else:
                        label_other[0,i_init//other_big_pixel_size[p],j_init//other_big_pixel_size[p]] = 0.  # Non-blob region
              
              # Store labels for this scale
              labels_other[p][k,:,:,:] = label_other
        
          # Store input and labels for this sample
          input_list[k,:,:,:] = input
          labels[k,:,:,:] = label
    return(input_list,labels,labels_other)


def make_data_feedforward(device, num_scales,num_trials):
    """Generate complete feedforward training dataset for all scales.
    
    This is the main entry point for generating the entire feedforward dataset.
    It creates curve datasets at multiple scales and a blob dataset, organizing
    them for multi-scale supervised learning.
    
    Args:
        device (torch.device): Device to place tensors on (CPU or CUDA).
        num_scales (int): Number of spatial scales (typically 4).
        
    Returns:
        tuple: (input_curve, labels_curve, input_blob, labels_other_blob)
            - input_curve: List of curve inputs for each scale
            - labels_curve: List of lists containing labels for all scales
            - input_blob: Blob inputs
            - labels_other_blob: Blob labels for all scales
    """
    # Calculate receptive field sizes for each scale
    # For num_scales=4: RF_size = [3, 9, 27] (3^1, 3^2, 3^3)
    RF_size = [3**i for i in range(1,num_scales)]
    
    # Initialize storage for curve data
    input_curve = []  # Will store inputs for each scale
    labels_curve = [[] for i in range(num_scales-1)]  # Labels organized by scale
    
    # Generate curve datasets for each scale
    for i in range(num_scales):
        # Generate curves at scale i
        # Grid size: RF_size[-1]*4 (e.g., 27*4=108 for 4 scales)
        # Target scale: RF_size[i]
        # Other scales: all scales except current one
        # Number of samples: 50,000
        (inputt,label,label_other) = make_dataset_curve(
            RF_size[-1]*4,                    # Grid size
            RF_size[i],                       # Target scale
            np.setdiff1d(RF_size,RF_size[i]), # Other scales
            num_trials,                            # Number of samples
            device
        )
        
        # Store input for this scale
        input_curve.append(inputt)
        
        # Insert the target scale label into the appropriate position
        label_other.insert(i,label)
        
        # Organize labels by scale (transpose structure)
        # labels_curve[scale][dataset] instead of labels_curve[dataset][scale]
        for p in range(len(label_other)):
            labels_curve[p].append(label_other[p])
    
    # Generate blob dataset at the highest scale
    # Grid size: RF_size[-1]*8 (e.g., 27*8=216 for 4 scales, larger than curves)
    # Target scale: RF_size[-1] (highest scale, e.g., 27)
    # Number of samples: 20,000 (fewer than curves)
    input_blob,highest_label_blob,labels_other_blob = make_dataset_blob(
        RF_size[-1]*8,                    # Grid size (larger for blobs)
        RF_size[-1],                      # Target scale (highest)
        np.setdiff1d(RF_size,RF_size[-1]), # Other scales
        20000,                            # Number of samples
        device
    )
    
    # Add the highest scale label to the blob labels
    labels_other_blob.insert(len(labels_other_blob),highest_label_blob)
    
    return(input_curve,labels_curve,input_blob,labels_other_blob)
