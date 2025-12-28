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
    """Base class for visual tasks with reward-based learning.
    
    This class defines the interface for tasks that present visual stimuli
    and provide rewards based on agent actions.
    """

    def __init__(self, n_hidden_features, device, grid_size, object_1=[], object_2=[]):
        """Initialize task.
        
        Args:
            n_hidden_features (int): Number of hidden features.
            device (torch.device): Device for computation.
            grid_size (int): Size of spatial grid.
            object_1 (list): Predefined first object (empty for random generation).
            object_2 (list): Predefined second object (empty for random generation).
        """

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

        self.flowcontrol = {'intertrial': self.handle_intertrial,
                            'go': self.handle_go}

    def state_reset(self):
        """Reset task state at end of trial."""
        self.trial_ended = True
        self.state = 'intertrial'
        self.input = self.display

    def step(self, action):
        """Execute one step of the task.
        
        Args:
            action (torch.Tensor): Action taken by the agent.
            
        Returns:
            tuple: (input, reward, trial_ended)
        """
        self.trial_ended = False
        self.flowcontrol[self.state](action)
        reward = self.current_reward
        input = self.input
        trial_ended = self.trial_ended
        self.current_reward = 0
        return(input, reward, trial_ended)

    def handle_intertrial(self, action):
        """Handle intertrial state - generate new stimulus.
        
        Args:
            action (torch.Tensor): Action (ignored during intertrial).
        """
        object_1,object_2 = self.pick_object()
        self.draw_stimulus(object_1,object_2)
        self.input = self.display
        self.state = 'go'


    def handle_go(self, action):
        """Handle go state - evaluate action and provide reward.
        
        Args:
            action (torch.Tensor): Action taken by agent.
        """
        pixel_chosen = torch.where(action == 1)[-1]
        if pixel_chosen == self.target_curve[-1]:
            self.current_reward = self.current_reward + self.final_reward * 0.8
            self.state_reset()
        else:
            self.state_reset()
       
class TraceCurves(Task):
    """Curve tracing task requiring selective attention.
    
    The agent must trace a target curve from a cued starting point to its endpoint,
    ignoring distractor curves. This requires multi-scale processing and sustained attention.
    """
    
    def __init__(self, n_hidden_features, device, grid_size, num_scales, object_1=[], object_2=[]):
        """Initialize curve tracing task.
        
        Args:
            n_hidden_features (int): Number of hidden features.
            device (torch.device): Device for computation.
            grid_size (int): Size of spatial grid.
            num_scales (int): Number of hierarchical scales.
            object_1 (list): Predefined target curve.
            object_2 (list): Predefined distractor curve.
        """
        super().__init__(n_hidden_features,device,grid_size,object_1,object_2)
        
        self.num_scales = num_scales
        self.RF_size_list =[3**i for i in range(1,self.num_scales)]
        
        self.grid_size_lists = [self.grid_size // self.RF_size_list[i] for i in range(len(self.RF_size_list))]
        
    def check_not_adjacent(self, curve1, curve2, grid_size, curve_1_2):
        """Check if two curves are adjacent or overlapping.
        
        Args:
            curve1 (np.ndarray): First curve coordinates.
            curve2 (np.ndarray): Second curve coordinates.
            grid_size (int): Size of grid.
            curve_1_2 (bool): Whether comparing different curves.
            
        Returns:
            bool: True if curves are adjacent or overlapping.
        """
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
        # Use predefined objects if provided, otherwise generate new ones
        if len(self.object_1) == 0:
          if self.only_blue:
              # Simple task: single blue pixel
              object_1 = [np.random.randint(self.grid_size ** 2)]
              object_2 = []
          else:
              # Complex task: generate two curves that don't overlap or touch
              proba = np.random.rand()
              while True:
                  try:
                    next_to_each_other = True
                    # Keep generating until curves are properly separated
                    while next_to_each_other:
                      if proba < 0.5:
                        # Random orientation curves
                        mask = np.zeros((self.grid_size, self.grid_size))
                        object_1, mask1 = make_curves([], mask,self.curve_length,grid_size=self.grid_size)
                        object_2, mask2 = make_curves([], mask1,self.curve_length,grid_size=self.grid_size)
      
                      else:
                        # Aligned curves (both vertical or both horizontal)
                        direction = np.random.choice([0,1])  # 0=vertical, 1=horizontal
                        mask = np.zeros((self.grid_size, self.grid_size))
                        object_1, mask1 = make_curves([], mask,self.curve_length,grid_size=self.grid_size,direction=direction)
                        object_2, mask2 = make_curves([], mask1,self.curve_length,grid_size=self.grid_size,direction=direction)
                      
                      next_to_each_other = False
      
                      # Check separation at all scales (important for multi-scale processing)
                      next_to_each_other_list = []
                      for scale in range(len(self.RF_size_list)):
                          # Project curves to coarser scale
                          curve1_temp = np.unique([((object_1[i] // self.grid_size) // self.RF_size_list[scale])*self.grid_size_lists[scale] + (object_1[i] % self.grid_size) // self.RF_size_list[scale] for i in range(len(object_1))])
                          curve2_temp = np.unique([((object_2[i] // self.grid_size) // self.RF_size_list[scale])*self.grid_size_lists[scale] + (object_2[i] % self.grid_size) // self.RF_size_list[scale] for i in range(len(object_2))])
                          # Check if target curve loops on itself
                          next_to_each_other_list.append(self.check_not_adjacent(curve1_temp,curve1_temp,self.grid_size_lists[scale],False))
                          # Check if target and distractor are adjacent
                          next_to_each_other_list.append(self.check_not_adjacent(curve1_temp,curve2_temp,self.grid_size_lists[scale],True))
                      
                      # Reject if any scale has touching curves
                      next_to_each_other = any(next_to_each_other_list)
                      
                    break
                  except IndexError:
                      # Retry if curve generation fails
                      pass
        else:
              # Use predefined curves
              object_1 = self.object_1
              object_2 = self.object_2
        return (object_1, object_2)
        
    def draw_stimulus(self,object_1, object_2):
        # Create RGB display (3 channels for red, green, blue)
        display = torch.zeros((1, self.n_hidden_features, self.grid_size, self.grid_size),device=self.device)
        if self.only_blue:
            # Simple task: just show the blue target
            display[:, 2, object_1[0] % self.grid_size, object_1[0]//self.grid_size] = 1
        else:
            # Target curve (object_1): color-coded by position
            for i in range(len(object_1)):
                if i == 0:
                    # Start point: RED (cue for agent to begin here)
                    display[:, 0, object_1[0] % self.grid_size, object_1[0]//self.grid_size] = 1
                elif i == len(object_1)-1:
                    # End point: BLUE (goal location)
                    display[:, 2, object_1[i] % self.grid_size, object_1[i]//self.grid_size] = 1
                else:
                    # Middle points: GREEN (path to follow)
                    display[:, 1, object_1[i] % self.grid_size, object_1[i]//self.grid_size] = 1
            
            # Distractor curve (object_2): also color-coded
            for i in range(len(object_2)):
                if i == 0:
                    # Start: GREEN (not a cue)
                    display[:, 1, object_2[0] % self.grid_size, object_2[0]//self.grid_size] = 1
                elif i == len(object_2)-1:
                    # End: BLUE (ambiguous endpoint - agent must avoid)
                    display[:, 2, object_2[i] % self.grid_size, object_2[i]//self.grid_size] = 1
                else:
                    # Middle: GREEN
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
        
        
        
    
