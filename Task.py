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

    def __init__(self, n_hidden_features,device,object_1=[],object_2=[]):

        self.grid_size = 72

        self.state = 0
        self.trial_ended = False


        self.state = 'intertrial'

        self.no_curves = False # First part of the curicculum is no curve, only the fixation point and a blue pixel

        self.current_reward = 0
        self.final_reward = 1.50
        
        self.object_1 = object_1
        self.object_2 = object_2

        self.device = device

        
        self.n_hidden_features = n_hidden_features

        self.flowcontrol = {'intertrial': self.do_intertrial,
                            'go': self.do_go}

        if self.task_type != 'searchtrace' and self.task_type != 'trace' and self.task_type != 'tracesearch':
            raise Exception('Task type must be either searchtrace or trace or tracesearch')

    def stateReset(self):
        self.trial_ended = True
        self.state = 'intertrial'
        self.input = [self.display, self.display_disk]

    def do_step(self, action):
        self.trial_ended = False
        self.flowcontrol[self.state](action)
        reward = self.current_reward
        input = self.input
        trial_ended = self.trial_ended
        self.current_reward = 0
        return(input, reward, trial_ended)

    def do_intertrial(self, action):
            self.pickTrialType()
            self.input = [self.display, self.display_disk]
            self.state = 'go'


    def do_go(self, action):
        pixel_chosen = torch.where(action == 1)[-1]
        if pixel_chosen == self.target_curve[-1]:
            self.current_reward = self.current_reward + self.final_reward * 0.8
            self.stateReset()
        else:
            self.stateReset()

    def pickTrialType(self):
        position_red_marker = np.random.randint(2)
        position_yellow_marker = 1 if position_red_marker == 0 else 0
        self.position_markers = [position_red_marker, position_yellow_marker]
        self.feature_target = np.random.randint(2) 
        object_1,object_2 = self.PickCurve()
        self.DrawStimulus(object_1,object_2)

        
class TraceCurves(Task):
    
    def __init__(self,n_hidden_features,device,middle_pixel_size,big_pixel_size,object_1=[],object_2=[]):
        self.task_type = 'trace'
        super().__init__(n_hidden_features,device,object_1,object_2)
        
        self.middle_pixel_size = middle_pixel_size
        
        self.big_pixel_size = big_pixel_size

        self.middle_grid_size = self.grid_size // self.middle_pixel_size
        
        self.big_grid_size = self.grid_size // self.big_pixel_size
        
    def check_not_adjacent(self,curve1,curve2,grid_size):
        next_to_each_other = False
        for i in range(len(curve1)):
            for j in range(len(curve2)):
                if np.abs(i-j) > 1 and np.array_equal(curve1,curve2): # Checking if the target curve loops on itself
                    if np.abs(curve1[i]-curve2[j]) == grid_size or np.abs(curve1[i]-curve2[j]) == 1:
                        next_to_each_other = True
                elif np.array_equal(curve1,curve2) == False: # Checking if the target curve and distractor curve are next to each other
                    if np.abs(curve1[i]-curve2[j]) == grid_size or np.abs(curve1[i]-curve2[j]) == 1:
                        next_to_each_other = True
        return(next_to_each_other)
    
    def PickCurve(self):
        if self.no_curves:
            red_yellow_position = np.random.randint(self.grid_size**2)
            blue_position = np.random.randint(self.grid_size**2)
            while blue_position == red_yellow_position:  ########CHANGE THAT
                blue_position = np.random.randint(self.grid_size**2)
            object_1 = [red_yellow_position, blue_position]
            object_2 = []
        else:
          if len(self.object_1) == 0:
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
    
                    curve1_middle = np.unique([((object_1[i] // self.grid_size) // self.middle_pixel_size)*self.middle_grid_size + (object_1[i] % self.grid_size) // self.middle_pixel_size for i in range(len(object_1))])
                    curve2_middle = np.unique([((object_2[i] // self.grid_size) // self.middle_pixel_size)*self.middle_grid_size + (object_2[i] % self.grid_size) // self.middle_pixel_size for i in range(len(object_2))])
                    next_to_each_other_1 = self.check_not_adjacent(curve1_middle,curve1_middle,self.middle_grid_size)
                    next_to_each_other_2 = self.check_not_adjacent(curve1_middle,curve2_middle,self.middle_grid_size)
                    
                    curve1_big = np.unique([((object_1[i] // self.grid_size) // self.big_pixel_size)*self.big_grid_size + (object_1[i] % self.grid_size) // self.big_pixel_size for i in range(len(object_1))])
                    curve2_big = np.unique([((object_2[i] // self.grid_size) // self.big_pixel_size)*self.big_grid_size + (object_2[i] % self.grid_size) // self.big_pixel_size for i in range(len(object_2))])
                    next_to_each_other_3 = self.check_not_adjacent(curve1_big,curve1_big,self.big_grid_size)
                    next_to_each_other_4 = self.check_not_adjacent(curve1_big,curve2_big,self.big_grid_size)
                    
                    next_to_each_other = next_to_each_other_1 or next_to_each_other_2 or next_to_each_other_3 or next_to_each_other_4
                    
                  break
                except IndexError:
                    pass
          else:
                object_1 = self.object_1
                object_2 = self.object_2
        return (object_1, object_2)
        
    def DrawStimulus(self,object_1, object_2):
        display = torch.zeros((1, self.n_hidden_features, self.grid_size, self.grid_size),device=self.device)
        display_disk = torch.zeros((1, self.n_hidden_features, 2))
        if self.no_curves:
                display[:, 0, object_1[0] % self.grid_size, object_1[0]//self.grid_size] = 0
                display[:, 3, object_1[1] % self.grid_size, object_1[1]//self.grid_size] = 1
        else:
                for i in range(len(object_1)):
                    if i == 0:  # red
                        display[:, 0, object_1[0] % self.grid_size, object_1[0]//self.grid_size] = 1
                    elif i == len(object_1)-1:  # blue
                        display[:, 3, object_1[i] % self.grid_size, object_1[i]//self.grid_size] = 1
                    else:  # green
                        display[:, 2, object_1[i] % self.grid_size, object_1[i]//self.grid_size] = 1
                for i in range(len(object_2)):
                    if i == 0:
                        display[:, 2, object_2[0] % self.grid_size, object_2[0]//self.grid_size] = 1
                    elif i == len(object_2)-1:
                        display[:, 3, object_2[i] % self.grid_size, object_2[i]//self.grid_size] = 1
                    else:
                        display[:, 2, object_2[i] % self.grid_size, object_2[i]//self.grid_size] = 1

        self.target_curve = object_1
        self.distractor_curve = object_2
        self.display = display
        self.display_disk = display_disk  
        

        
class TraceObjects(Task):
    
    def __init__(self,n_hidden_features,device,object_1=[],object_2=[]):
        self.task_type = 'trace'
        super().__init__(n_hidden_features,device,object_1,object_2)

    def PickCurve(self):
        if len(self.object_1) == 0:
            rad = 0.9999
            edgy = 0.9999
            n = 7
            if np.random.rand() < 0.5: #first coordinates is where the target object will be
                c = [[0,0],[30,30]]
            else:
                c = [[30,30],[0,0]]
            intersect = True
            while intersect:
              blob = []
              for  i in range(len(c)):
                a = get_random_points(n=n, scale=30) + c[i]
                x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)
                blob.append([x,y])     
              x_blob_1,y_blob_1 = polygon(blob[0][0], blob[0][1], (self.grid_size,self.grid_size)) #filling in
              x_blob_2,y_blob_2 = polygon(blob[1][0], blob[1][1], (self.grid_size,self.grid_size))
              if len(np.intersect1d(x_blob_1,x_blob_2)) == 0 or len(np.intersect1d(y_blob_1,y_blob_2)) == 0:
                intersect = False
            object_1 = np.array(x_blob_1) + np.array(y_blob_1) * self.grid_size
            object_2 = np.array(x_blob_2) + np.array(y_blob_2) * self.grid_size
            np.random.shuffle(object_1)
            np.random.shuffle(object_2)
        else:
            object_1 = self.object_1
            object_2 = self.object_2           
        return (object_1, object_2)
        
    def DrawStimulus(self,object_1, object_2):
        display = torch.zeros((1, self.n_hidden_features, self.grid_size, self.grid_size),device=self.device)
        display_disk = torch.zeros((1, self.n_hidden_features, 2))

        #green  
        ind = np.unravel_index(object_1[1:-1], display[0,2,:].shape, 'F')
        display[:,2,ind[0],ind[1]] = 1
        ind = np.unravel_index(object_2[:-1], display[0,2,:].shape, 'F')
        display[:,2,ind[0],ind[1]] = 1
        
        #blue
        display[:, 3, object_1[-1] % self.grid_size, object_1[-1]//self.grid_size] = 1
        display[:, 3, object_2[-1] % self.grid_size, object_2[-1]//self.grid_size] = 1
        
        #red
        display[:, 0, object_1[0] % self.grid_size, object_1[0]//self.grid_size] = 1
          
        self.target_curve = object_1
        self.distractor_curve = object_2
        self.display = display
        self.display_disk = display_disk     
        
        
        
    
