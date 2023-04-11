# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 09:48:43 2023

@author: Sami
"""

import torch
import torch.nn as nn
import numpy as np
import random
from FF_data import *
from FeedforwardNetwork import *
from helper_functions import *

def make_data_feedforward(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = None
    
    (input_3_blob,labels_3_blob,labels_3_other_blob) = make_dataset_blob(72,3,9,10000,device)
    (input_9_blob,labels_9_blob,labels_9_other_blob) = make_dataset_blob(72,9,3,10000,device)

    (input_3_curve,labels_3_curve,labels_3_other_curve) = make_dataset_curve(18,3,9,50000,device)
    (input_9_curve,labels_9_curve,labels_9_other_curve) = make_dataset_curve(18,9,3,50000,device)
    return(input_3_blob,labels_3_blob,labels_3_other_blob,input_9_blob,labels_9_blob,labels_9_other_blob,input_3_curve,labels_3_curve,labels_3_other_curve,input_9_curve,labels_9_curve,labels_9_other_curve)

def train_feedforward_blob(input_3_blob,labels_3_blob,labels_3_other_blob,input_9_blob,labels_9_blob,labels_9_other_blob):
    device = None
    feedforward_blob = FeedforwardNetwork(device)
    feedforward_blob = feedforward_blob.to(device)
    criterion = [nn.BCELoss(),nn.BCELoss()]
    optimizer = optim.Adam(feedforward_blob.parameters(), lr=0.001)
    feedforward_blob.train(optimizer,criterion,torch.cat((input_3_blob.to(device),input_9_blob.to(device)),dim=0),[torch.cat((labels_3_blob.to(device),labels_9_other_blob.to(device)),dim=0),torch.cat((labels_3_other_blob.to(device),labels_9_blob.to(device)),dim=0)],epochs=80,verbose=False,batch_size=256)
    return(feedforward_blob)

def train_feedforward_curve(input_3_curve,labels_3_curve,labels_3_other_curve,input_9_curve,labels_9_curve,labels_9_other_curve):
    device = None
    feedforward_curve = FeedforwardNetwork(device)
    feedforward_curve = feedforward_blob.to(device)
    criterion = [nn.BCELoss(),nn.BCELoss()]
    optimizer = optim.Adam(feedforward_blob.parameters(), lr=0.001)
    feedforward_curve.train(optimizer,criterion,torch.cat((input_3_curve.to(device),input_9_curve.to(device)),dim=0),[torch.cat((labels_3_curve.to(device),labels_9_other_curve.to(device)),dim=0),torch.cat((labels_3_other_curve.to(device),labels_9_curve.to(device)),dim=0)],epochs=80,verbose=False,batch_size=256)
    return(feedforward_blob)

def train_full_network(feedforward_curve,feedforward_object):
    device = None
    grid_size = 36
    big_pixels_size = 3
    bigger_pixels_size = 9
    n = RecurrentNetwork(3,grid_size,big_pixels_size,bigger_pixels_size,device,feedforward_curve,feedforward_object)

    n.duration = 30
    n.save_activities = False

    max_length = 3
    min_length = 3
    
    t=TraceCurves(3,device,3,9)
    t.grid_size = grid_size
    t.curve_length = 3
    
    reward = 0
    trialEnd = False
    trials = 50000
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
            (n,corrects,target_history,distr_history,display) = test_network(TraceCurves(3,device,3,9),max_length,grid_size,500,n,device,False)
            average = np.mean(corrects)
            average_all.append(average)
        if max_length < total_length:
            if (i-tac) > 2000 and average >= 0.85:
              max_length += 1
              tac = i
        elif max_length == total_length and (i-tac) > 2000 and average >= 0.85:
              break   

    return(n,trial_corrects)


