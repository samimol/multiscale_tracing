# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:04:10 2023

@author: Sami
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
import datetime
from FF_data import *
from FeedforwardNetwork import *
from RecurrentNetwork import *
from Task import *
from helper_functions import *
from opts import parser
import torch.optim as optim

if os.name == 'nt':
   batch_id = 0
else:
    batch_id = os.environ["SLURM_PROCID"]
    
args = parser.parse_args()

if args.num_networks > 1:
    device = torch.device('cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = int(batch_id) + datetime.datetime.now().microsecond
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
    

def train_full_network(feedforward_curve,feedforward_object,one_scale,device):
    grid_size = 36
    big_pixels_size = 3
    bigger_pixels_size = 9
    n = RecurrentNetwork(3,grid_size,big_pixels_size,bigger_pixels_size,device,feedforward_curve,feedforward_object,one_scale)

    n.duration = 30
    n.save_activities = False

    max_length = 3
    min_length = 3
    
    t=TraceCurves(3,device,3,9)
    t.only_blue = False
    t.grid_size = grid_size
    t.curve_length = 3
    
    reward = 0
    trialEnd = False
    trials = 55000
    average = 0
    trial_corrects = []
    average_all = []
    action = 0
    tac = 0
    total_length = 8

    for i in range(trials):
        trial_running = True
        new_input, reward, trialEnd= t.do_step(action)
    
        while trial_running:
          action = n.do_step(new_input,reward,trialEnd,device)
          new_input, reward, trialEnd = t.do_step(action)
          n.do_learn(reward)
          if trialEnd:
                trial_running = False
                t.curve_length = np.random.randint(max_length-min_length+1) + min_length
                if reward != 0:
                    trial_corrects.append(1)
                else:
                    trial_corrects.append(0)
        if (i-tac)%2000 == 0 and i > 0:
            (n,corrects,target_history,distr_history,display) = test_network(TraceCurves(3,device,3,9),max_length,grid_size,500,n,device,False,t.only_blue)
            average = np.mean(corrects)
            average_all.append(average)
        if max_length < total_length:
            if (i-tac) > 500 and t.only_blue:
              (n,corrects,target_history,distr_history,display) = test_network(TraceCurves(3,device,3,9),max_length,grid_size,500,n,device,False,t.only_blue)
              tac = i
              if np.mean(corrects) > 0.85:
                t.only_blue = False   
            if (i-tac) > 2000 and average >= 0.85:
              tac = i
              max_length += 1
        elif max_length == total_length and (i-tac) > 2000 and average >= 0.85:
              break   

    return(n,trial_corrects)


if __name__ == '__main__':
    
    results_folder = os.path.join('multiscale','results','recurrent_networks')
    if args.full_training:
        assert device.type == 'cuda', 'full training should be done on gpu'
        (input_3_blob,labels_3_blob,labels_3_other_blob,input_9_blob,labels_9_blob,labels_9_other_blob,input_3_curve,labels_3_curve,labels_3_other_curve,input_9_curve,labels_9_curve,labels_9_other_curve) = make_data_feedforward(device)
        feedforward_blob = train_feedforward_blob(input_3_blob,labels_3_blob,labels_3_other_blob,input_9_blob,labels_9_blob,labels_9_other_blob,device)
        feedforward_curve = train_feedforward_curve(input_3_curve,labels_3_curve,labels_3_other_curve,input_9_curve,labels_9_curve,labels_9_other_curve,device)
        n,trial_corrects = train_full_network(feedforward_curve,feedforward_blob,args.one_scale,device)
        
        filename = os.path.join(results_folder, 'n_' + batch_id + '.pt')
        torch.save(n, filename)
        
        filename = os.path.join(results_folder, 'performance_' + batch_id + '.pt')
        np.save(filename,np.array(trial_corrects))
    else:    
        feedfoward_folder = os.path.join('multiscale','results','feedforward_networks')
    
        if device.type == 'cuda':
          feedforward_curve = torch.load(os.path.join(feedfoward_folder,'FF_curve_' + batch_id + '.pt'))
        else:
            feedforward_curve = torch.load(os.path.join(feedfoward_folder,'FF_curve_' + batch_id + '.pt'), map_location=torch.device('cpu'))
        if device.type == 'cuda':
          feedforward_object = torch.load(os.path.join(feedfoward_folder,'FF_blob_' + batch_id + '.pt'))
        else:
          feedforward_object = torch.load(os.path.join(feedfoward_folder,'FF_blob_' + batch_id + '.pt'), map_location=torch.device('cpu'))
            
        n,trial_corrects = train_full_network(feedforward_curve,feedforward_object,args.one_scale,device)
        
        filename = os.path.join(results_folder, 'n_' + batch_id + '.pt')
        torch.save(n, filename)
        
        filename = os.path.join(results_folder, 'performance_' + batch_id)
        np.save(filename,np.array(trial_corrects))
    
    
    
    
    
    
    
    
    
    
    
    