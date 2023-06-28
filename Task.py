# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 13:17:30 2023

@author: Sami
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:13:17 2021

@author: Sami
"""

import numpy as np
import torch
from helper_functions import make_curves, get_random_points, get_bezier_curve
from skimage.draw import polygon

class Task():

    def __init__(self, n_hidden_features,device,grid_size,object_1=[],object_2=[]):

        self.grid_size = grid_size

        self.state = 0
        self.trial_ended = False


        self.state = 'intertrial'
        
        self.current_reward = 0
        self.final_reward = 1.50
        
        self.object_1 = object_1
        self.object_2 = object_2

        self.device = device
        
        self.euclidean_distance = 0

        
        self.n_hidden_features = n_hidden_features

        self.flowcontrol = {'intertrial': self.do_intertrial,
                            'go': self.do_go}

    def state_reset(self):
        self.trial_ended = True
        self.state = 'intertrial'
        self.input = self.display

    def do_step(self, action):
        self.trial_ended = False
        self.flowcontrol[self.state](action)
        reward = self.current_reward
        input = self.input
        trial_ended = self.trial_ended
        self.current_reward = 0
        return(input, reward, trial_ended)

    def do_intertrial(self, action):
        object_1,object_2 = self.pick_object()
        self.draw_stimulus(object_1,object_2)
        self.input = self.display
        self.state = 'go'


    def do_go(self, action):
        pixel_chosen = torch.where(action == 1)[-1]
        if pixel_chosen == self.target_curve[-1]:
            self.current_reward = self.current_reward + self.final_reward * 0.8
            self.state_reset()
        else:
            self.state_reset()
       
class TraceCurves(Task):
    
    def __init__(self,n_hidden_features,device,grid_size,num_scales,object_1=[],object_2=[]):
        super().__init__(n_hidden_features,device,grid_size,object_1,object_2)
        
        self.num_scales = num_scales
        self.RF_size_list =[3**i for i in range(1,self.num_scales)]
        
        self.grid_size_lists = [self.grid_size // self.RF_size_list[i] for i in range(len(self.RF_size_list))]
        
    def check_not_adjacent(self,curve1,curve2,grid_size,curve_1_2):
        next_to_each_other = False
        for i in range(len(curve1)):
            for j in range(len(curve2)):
                if curve_1_2 == False and np.abs(i-j) > 1: # Checking if the target curve loops on itself
                    if np.abs(curve1[i]-curve2[j]) == grid_size or np.abs(curve1[i]-curve2[j]) == 1:
                        next_to_each_other = True
                elif curve_1_2:
                    if np.array_equal(curve1,curve2):
                        next_to_each_other = True
                    else: # Checking if the target curve and distractor curve are next to each other
                        if np.abs(curve1[i]-curve2[j]) == grid_size or np.abs(curve1[i]-curve2[j]) == 1:
                            next_to_each_other = True
        return(next_to_each_other)
    
    def pick_object(self):
        if len(self.object_1) == 0:
          if self.only_blue:
              object_1 = [np.random.randint(self.grid_size ** 2)]
              object_2 = []
          else:
              proba = np.random.rand()
              while True:
                  try:
                    next_to_each_other = True
                    while next_to_each_other:
                      if proba < 0.5:
                        mask = np.zeros((self.grid_size, self.grid_size))
                        object_1, mask1 = make_curves([], mask,self.curve_length,grid_size=self.grid_size)
                        object_2, mask2 = make_curves([], mask1,self.curve_length,grid_size=self.grid_size)
      
                      else:
                        direction = np.random.choice([0,1]) #{up,down},{left,right}    
                        mask = np.zeros((self.grid_size, self.grid_size))
                        object_1, mask1 = make_curves([], mask,self.curve_length,grid_size=self.grid_size,direction=direction)
                        object_2, mask2 = make_curves([], mask1,self.curve_length,grid_size=self.grid_size,direction=direction)
                      
                      next_to_each_other = False
      
                      next_to_each_other_list = []
                      for scale in range(len(self.RF_size_list)):
                          curve1_temp = np.unique([((object_1[i] // self.grid_size) // self.RF_size_list[scale])*self.grid_size_lists[scale] + (object_1[i] % self.grid_size) // self.RF_size_list[scale] for i in range(len(object_1))])
                          curve2_temp = np.unique([((object_2[i] // self.grid_size) // self.RF_size_list[scale])*self.grid_size_lists[scale] + (object_2[i] % self.grid_size) // self.RF_size_list[scale] for i in range(len(object_2))])
                          next_to_each_other_list.append(self.check_not_adjacent(curve1_temp,curve1_temp,self.grid_size_lists[scale],False))
                          next_to_each_other_list.append(self.check_not_adjacent(curve1_temp,curve2_temp,self.grid_size_lists[scale],True))
                      
                      next_to_each_other = any(next_to_each_other_list)
                      
                    break
                  except IndexError:
                      pass
        else:
              object_1 = self.object_1
              object_2 = self.object_2
        return (object_1, object_2)
        
    def draw_stimulus(self,object_1, object_2):
        display = torch.zeros((1, self.n_hidden_features, self.grid_size, self.grid_size),device=self.device)
        if self.only_blue:
            display[:, 2, object_1[0] % self.grid_size, object_1[0]//self.grid_size] = 1
        else:
            for i in range(len(object_1)):
                if i == 0:  # red
                    display[:, 0, object_1[0] % self.grid_size, object_1[0]//self.grid_size] = 1
                elif i == len(object_1)-1:  # blue
                    display[:, 2, object_1[i] % self.grid_size, object_1[i]//self.grid_size] = 1
                else:  # green
                    display[:, 1, object_1[i] % self.grid_size, object_1[i]//self.grid_size] = 1
            for i in range(len(object_2)):
                if i == 0:
                    display[:, 1, object_2[0] % self.grid_size, object_2[0]//self.grid_size] = 1
                elif i == len(object_2)-1:
                    display[:, 2, object_2[i] % self.grid_size, object_2[i]//self.grid_size] = 1
                else:
                    display[:, 1, object_2[i] % self.grid_size, object_2[i]//self.grid_size] = 1

        self.target_curve = object_1
        self.distractor_curve = object_2
        self.display = display
        

        
class TraceObjects(Task):
    
    def __init__(self,n_hidden_features,device,grid_size,object_1=[],object_2=[]):
        super().__init__(n_hidden_features,device,grid_size,object_1,object_2)

    def pick_object(self):
        if len(self.object_1) == 0:
            rad = 0.9999
            edgy = 0.9999
            n = 7
            if np.random.rand() < 0.5: #first coordinates is where the target object will be
                c = [[0,0],[self.grid_size//2,self.grid_size//2]]
            else:
                c = [[self.grid_size//2,self.grid_size//2],[0,0]]
            dist = -5
            distances = [10,20,30,40,50,60]
            goal_dist = np.random.choice(distances)
            while not ((dist<goal_dist) and (dist>goal_dist-10)):
                intersect = True
                while intersect:
                  blob = []
                  for  i in range(len(c)):
                    a = get_random_points(n=n, scale=90) + c[i]
                    x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)
                    blob.append([x,y])     
                  x_blob_1,y_blob_1 = polygon(blob[0][0], blob[0][1], (self.grid_size,self.grid_size)) #filling in
                  x_blob_2,y_blob_2 = polygon(blob[1][0], blob[1][1], (self.grid_size,self.grid_size))
                  if len(np.intersect1d(x_blob_1,x_blob_2)) == 0 or len(np.intersect1d(y_blob_1,y_blob_2)) == 0:
                    intersect = False
                object_1 = np.array(x_blob_1) + np.array(y_blob_1) * self.grid_size
                object_2 = np.array(x_blob_2) + np.array(y_blob_2) * self.grid_size
                np.random.shuffle(object_1)
                xred = object_1[0] % self.grid_size
                yred = object_1[0] // self.grid_size
                xblue = object_1[-1] % self.grid_size
                yblue = object_1[-1] // self.grid_size  
                dist = np.sqrt((xred-xblue)**2+(yred-yblue)**2)
                np.random.shuffle(object_2)
            self.euclidean_distance = dist
        else:
            object_1 = self.object_1
            object_2 = self.object_2           
        return (object_1, object_2)
        
    def draw_stimulus(self,object_1, object_2):
        display = torch.zeros((1, self.n_hidden_features, self.grid_size, self.grid_size),device=self.device)

        #green  
        ind = np.unravel_index(object_1[1:-1], display[0,1,:].shape, 'F')
        display[:,1,ind[0],ind[1]] = 1
        ind = np.unravel_index(object_2[:-1], display[0,1,:].shape, 'F')
        display[:,1,ind[0],ind[1]] = 1
        
        #blue
        display[:, 2, object_1[-1] % self.grid_size, object_1[-1]//self.grid_size] = 1
        display[:, 2, object_2[-1] % self.grid_size, object_2[-1]//self.grid_size] = 1
        
        #red
        display[:, 0, object_1[0] % self.grid_size, object_1[0]//self.grid_size] = 1
          
        self.target_curve = object_1
        self.distractor_curve = object_2
        self.display = display
        
        
        
    
