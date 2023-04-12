# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 16:05:47 2023

@author: Sami
"""
import torch
import numpy as np
import random
from helper_functions import make_curves, make_blob

def make_dataset_curve(grid_size,big_pixel_size,other_big_pixel_size,num_trial,device):
    grid_size = grid_size
    big_pixel_size = big_pixel_size
    big_grid_size = grid_size // big_pixel_size
    other_big_grid_size = grid_size // other_big_pixel_size
    num_trial = num_trial
    feature_number = 3
    input_list = torch.zeros((num_trial,feature_number,grid_size,grid_size),device=device)
    labels = torch.zeros((num_trial,1,big_grid_size,big_grid_size),device=device)
    labels_other = torch.zeros((num_trial,1,other_big_grid_size,other_big_grid_size),device=device)
    for k in range(num_trial):
          input = torch.zeros((feature_number,grid_size,grid_size),device=device)
          label = torch.zeros((1,big_grid_size,big_grid_size),device=device)
          label_other = torch.zeros((1,other_big_grid_size,other_big_grid_size),device=device)
          for i_init in range(0,grid_size,big_pixel_size):
            for j_init in range(0,grid_size,big_pixel_size):
              twocurves = False
              kernel = torch.zeros((1,feature_number,big_pixel_size,big_pixel_size))
              curvelength = np.random.choice([0]+list(range(3,3*big_pixel_size-2)))#,p=[0.05,0.75,0.05,0.05,0.05,0.05])
              prob = np.random.rand()
              direction = np.random.choice([0,1]) #{up,down},{left,right}  
              if curvelength != 0:
                while True:
                  try:
                    if prob < 0.5 and curvelength in list(range(2,big_pixel_size+1)):
                        mask = np.zeros((big_pixel_size, big_pixel_size))
                        curve1, mask1 = make_curves([], mask,curvelength,grid_size=big_pixel_size,direction=direction)
                        if np.random.rand() < 0.5 and big_pixel_size > 3:
                            curve2, mask2 = make_curves([], mask1,curvelength,grid_size=big_pixel_size,direction=direction)
                            twocurves = True                            
                    else:
                        mask = np.zeros((big_pixel_size, big_pixel_size))
                        curve1, mask1 = make_curves([], mask,curvelength,grid_size=big_pixel_size)
                        if np.random.rand() < 0.5: 
                            if curvelength > 2 and curvelength < big_pixel_size*2 and big_pixel_size > 3:
                                curve2, mask2 = make_curves([], mask1,curvelength,grid_size=big_pixel_size)
                                twocurves = True
                            elif  curvelength > 2 and curvelength < big_pixel_size*2 and big_pixel_size == 3:
                                curve1.pop(random.randrange(len(curve1))) 
                    for i in range(1,len(curve1)-1):
                      kernel[0,1,curve1[i]%big_pixel_size,curve1[i]//big_pixel_size] = 1
                    feat1 = np.random.choice([0,1,-1])
                    kernel[0,feat1,curve1[0]%big_pixel_size,curve1[0]//big_pixel_size] = 1
                    feat2 = [0,1,-1]
                    feat2.remove(feat1)
                    kernel[0,np.random.choice(feat2),curve1[-1]%big_pixel_size,curve1[-1]//big_pixel_size] = 1
                    if twocurves:
                        for i in range(1,len(curve2)-1):
                          kernel[0,1,curve2[i]%big_pixel_size,curve2[i]//big_pixel_size] = 1
                        feat1 = np.random.choice([0,1,-1])
                        kernel[0,feat1,curve2[0]%big_pixel_size,curve2[0]//big_pixel_size] = 1
                        feat2 = [0,1,-1]
                        feat2.remove(feat1)
                        kernel[0,np.random.choice(feat2),curve2[-1]%big_pixel_size,curve2[-1]//big_pixel_size] = 1    
                    break
                  except IndexError:
                      pass
                if curvelength > 0:
                    if twocurves == False:
                        if (np.all(np.abs(np.diff(curve1))==1) or np.all(np.abs(np.diff(curve1))==big_pixel_size)):
                          label[0,i_init//big_pixel_size,j_init//big_pixel_size] = 1#label_true
                        else:
                          label[0,i_init//big_pixel_size,j_init//big_pixel_size] = 0.#label_false
                    input[:,i_init:i_init+big_pixel_size,j_init:j_init+big_pixel_size] = kernel
          input_collapsed = torch.sum(input,dim=0)
          for i_init_other in range(0,grid_size,other_big_pixel_size):
            for j_init_other in range(0,grid_size,other_big_pixel_size):
                kernel = input_collapsed[i_init_other:i_init_other+other_big_pixel_size,j_init_other:j_init_other+other_big_pixel_size]
                non_zero_kernel = torch.nonzero(kernel)
                if len(non_zero_kernel) != 0:
                    if (torch.all(non_zero_kernel[:,0] == non_zero_kernel[:,0][0]) and torch.all(torch.diff(non_zero_kernel[:,1]) == 1)) or (torch.all(non_zero_kernel[:,1] == non_zero_kernel[:,1][0])and torch.all(torch.diff(non_zero_kernel[:,0]) == 1)):
                        label_other[0,i_init_other//other_big_pixel_size,j_init_other//other_big_pixel_size] = 1
          input_list[k,:,:,:] = input
          labels[k,:,:,:] = label
          labels_other[k,:,:,:] = label_other
    return(input_list,labels,labels_other)


def make_dataset_blob(grid_size,big_pixel_size,other_big_pixel_size,num_trial,device):
    grid_size = grid_size
    big_pixel_size = big_pixel_size
    big_grid_size = grid_size // big_pixel_size
    other_big_grid_size = grid_size // other_big_pixel_size
    num_trial = num_trial
    feature_number = 3
    input_list = torch.zeros((num_trial,feature_number,grid_size,grid_size),device=device)
    labels = torch.zeros((num_trial,1,big_grid_size,big_grid_size),device=device)
    labels_other = torch.zeros((num_trial,1,other_big_grid_size,other_big_grid_size),device=device)
    for k in range(num_trial):
          input = torch.zeros((feature_number,grid_size,grid_size),device=device)
          label = torch.zeros((1,big_grid_size,big_grid_size),device=device)
          label_other = torch.zeros((1,other_big_grid_size,other_big_grid_size),device=device)
          i_init = np.random.choice(list(range(0,grid_size,big_pixel_size)))
          j_init = np.random.choice(list(range(0,grid_size,big_pixel_size)))
          while True:
            try:
              x_blob,y_blob = make_blob(grid_size)
              ind = np.random.randint(len(x_blob),size=2)
              x0 = x_blob[ind[0]]
              y0 = y_blob[ind[0]]
              x_target = x_blob[ind[1]]
              y_target = y_blob[ind[1]]
              input[1,x_blob,y_blob] = 1
              input[1,x0,y0] = 0
              input[2,x0,y0] = 1
              if np.random.rand() < 0.5: #we put a blue pixel
                input[0,x_target,y_target] = 1
                input[1,x_target,y_target] = 0
              break
            except IndexError:
                      pass
          for i_init in range(0,grid_size,big_pixel_size):
            for j_init in range(0,grid_size,big_pixel_size):
                if torch.all(torch.sum(input[:,i_init:i_init+big_pixel_size,j_init:j_init+big_pixel_size],dim=0)==1):
                    label[0,i_init//big_pixel_size,j_init//big_pixel_size] = 1.
                else:
                    label[0,i_init//big_pixel_size,j_init//big_pixel_size] = 0.
          for i_init in range(0,grid_size,other_big_pixel_size):
            for j_init in range(0,grid_size,other_big_pixel_size):
                if torch.all(torch.sum(input[:,i_init:i_init+other_big_pixel_size,j_init:j_init+other_big_pixel_size],dim=0)==1):
                    label_other[0,i_init//other_big_pixel_size,j_init//other_big_pixel_size] = 1.
                else:
                    label_other[0,i_init//other_big_pixel_size,j_init//other_big_pixel_size] = 0.
          input_list[k,:,:,:] = input
          labels[k,:,:,:] = label
          labels_other[k,:,:,:] = label_other
    return(input_list,labels,labels_other)


def make_data_feedforward(device):

    (input_3_blob,labels_3_blob,labels_3_other_blob) = make_dataset_blob(72,3,9,10000,device)
    (input_9_blob,labels_9_blob,labels_9_other_blob) = make_dataset_blob(72,9,3,10000,device)

    (input_3_curve,labels_3_curve,labels_3_other_curve) = make_dataset_curve(18,3,9,50000,device)
    (input_9_curve,labels_9_curve,labels_9_other_curve) = make_dataset_curve(18,9,3,50000,device)
    return(input_3_blob,labels_3_blob,labels_3_other_blob,input_9_blob,labels_9_blob,labels_9_other_blob,input_3_curve,labels_3_curve,labels_3_other_curve,input_9_curve,labels_9_curve,labels_9_other_curve)
