# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:04:10 2023

@author: Sami
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from tqdm import tqdm
import numpy as np
import random
import os
import datetime

from config.model_config import parser
from src.data.feedforward_data import make_data_feedforward
from src.models.feedforward_network import train_feedforward_blob, train_feedforward_curve
from src.models.recurrent_network import RecurrentNetwork
from src.tasks.tasks import TraceCurves
from src.utils.helper_functions import test_network

if "SLURM_JOB_ID" in os.environ:
    batch_id = str(os.environ.get("SLURM_PROCID", 0))
else:
    batch_id = "0"
    
args = parser.parse_args()

if args.num_networks > 1:
    device = torch.device('cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = int(batch_id) + datetime.datetime.now().microsecond
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
    
num_scales = args.num_scales

def train_full_network(feedforward_curve,feedforward_object,one_scale,device,num_scales):
    
    grid_size = (3 ** (num_scales - 1)) * 4
    n = RecurrentNetwork(3,grid_size,device,feedforward_curve,feedforward_object,one_scale,num_scales)
    if num_scales > 3 and n.task == 'trace_curve':
        n.beta = 0.002
    else:
        n.beta = 0.02
    n.duration = 30
    n.save_activities = False
    
    max_length = 3
    min_length = 3
    
    t=TraceCurves(3,device,grid_size,num_scales)
    t.only_blue = False
    t.grid_size = grid_size
    t.curve_length = 3

    trials = 55000
    average = 0
    trial_corrects = []
    average_all = []
    action = 0
    tac = 0
    total_length = args.total_length
    
    generalization = np.zeros((total_length-2,total_length+1))
    
    for i in tqdm(range(trials)):
        trial_running = True
        new_input, reward, trialEnd= t.step(action)
        if i > 2000:
            n.beta = 0.02
        while trial_running:
          action = n.step(new_input,reward,trialEnd,device)
          new_input, reward, trialEnd = t.step(action)
          n.learn(reward)
          if trialEnd:
                trial_running = False
                t.curve_length = np.random.randint(max_length-min_length+1) + min_length
                if reward != 0:
                    trial_corrects.append(1)
                else:
                    trial_corrects.append(0)
        if (i-tac)%2000 == 0 and i > 0:
            (n,corrects,target_history,distr_history,display) = test_network(TraceCurves(3,device,grid_size,num_scales),max_length,grid_size,500,n,device,False,t.only_blue)
            average = np.mean(corrects)
            average_all.append(average)
        if max_length < total_length:
            if (i-tac) > 500 and t.only_blue:
              (n,corrects,target_history,distr_history,display) = test_network(TraceCurves(3,device,grid_size,num_scales),max_length,grid_size,500,n,device,False,t.only_blue)
              tac = i
              if np.mean(corrects) > 0.85:
                t.only_blue = False   
            if (i-tac) > 2000 and average >= 0.85:
                for k in range(max_length+1,max_length+5):
                    (n,corrects,target_history,distr_history,display) = test_network(TraceCurves(3,device,grid_size,num_scales),k+20,grid_size,500,n,device,False,t.only_blue)
                    generalization[max_length-3,k-4] = np.mean(corrects)    
                tac = i
                max_length += 1
        elif max_length == total_length and (i-tac) > 2000 and average >= 0.85:
            for k in range(max_length+1,max_length+5):
                (n,corrects,target_history,distr_history,display) = test_network(TraceCurves(3,device,grid_size,num_scales),k+20,grid_size,500,n,device,False,t.only_blue)
                generalization[max_length-3,k-4] = np.mean(corrects)      
            break   

    return(n,trial_corrects,generalization)


if __name__ == '__main__':
    
    results_folder = os.path.join(project_root,'models','recurrent')
    if args.one_scale:
        results_folder = os.path.join(results_folder,'_one_scale')
        
    if args.full_training:
        assert device.type == 'cuda', 'full training should be done on gpu'
        (input_curve,labels_curve,input_blob,labels_blob) = make_data_feedforward(device,num_scales)
        
        feedforward_blob = train_feedforward_blob(num_scales,input_blob,labels_blob,device)
        feedforward_curve =train_feedforward_curve(num_scales,input_curve,labels_curve,device)
        
        n,trial_corrects,generalization = train_full_network(feedforward_curve,feedforward_blob,args.one_scale,device,num_scales)
        
        filename = os.path.join(results_folder, 'n_' + batch_id + '.pt')
        torch.save(n, filename)
        
        filename = os.path.join(results_folder, 'performance_' + batch_id + '.pt')
        np.save(filename,np.array(trial_corrects))
    else:    
        feedfoward_folder = os.path.join(project_root,'models','feedforward')
    
        if device.type == 'cuda':
          feedforward_curve = torch.load(os.path.join(feedfoward_folder,'curve','FF_curve_' + batch_id + '.pt'),weights_only=False)
        else:
            feedforward_curve = torch.load(os.path.join(feedfoward_folder,'curve','FF_curve_' + batch_id + '.pt'), map_location=torch.device('cpu'),weights_only=False)
        if device.type == 'cuda':
          feedforward_object = torch.load(os.path.join(feedfoward_folder,'blob','FF_blob_' + batch_id + '.pt'),weights_only=False)
        else:
          feedforward_object = torch.load(os.path.join(feedfoward_folder,'blob','FF_blob_' + batch_id + '.pt'), map_location=torch.device('cpu'),weights_only=False)
            
        n,trial_corrects,generalization = train_full_network(feedforward_curve,feedforward_object,args.one_scale,device,num_scales)
        
        filename = os.path.join(results_folder, 'n_' + batch_id + '_' + str(num_scales) + '.pt')
        torch.save(n, filename)
        
        filename = os.path.join(results_folder, 'performance_' + batch_id)
        np.save(filename,np.array(trial_corrects))
    
        filename = os.path.join(results_folder, 'generalization_' + batch_id)
        np.save(filename,np.array(generalization))
    
    
    
    
    
    
    
    
    
    
    
    