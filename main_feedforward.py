# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:34:33 2023

@author: Sami
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
import datetime
from config import parser
import pickle
from feedforward_data import *
from feedforward_network import *
from helper_functions import *

    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = parser.parse_args()

seed = datetime.datetime.now().microsecond
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)



if __name__ == '__main__':
    
    results_folder = os.path.join('multiscale','results','feedforward_networks')
    data_folder = os.path.join('multiscale','results','feedforward_data')
    
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)
        
        
    num_scales = 4
        
    with open('feedforward_dataset.pkl', 'rb') as input:
        feedforward_dataset = pickle.load(input)
    
    input_blob,labels_blob = feedforward_dataset[1][0][0] , feedforward_dataset[1][0][1]
    
    
    input_curve = []
    labels_curve = [[] for i in range(num_scales-1)]
    for index_1,index_2 in enumerate([0,2,3]):
        (inputt,labels_curve_interm) = feedforward_dataset[index_2][0][0] , feedforward_dataset[index_2][0][1]
        for p in range(len(labels_curve_interm)):
            labels_curve[p].append(labels_curve_interm[p])
        input_curve.append(inputt)
    
    del feedforward_dataset
    
    for i in range(args.num_networks):

        feedforward_blob = train_feedforward_blob(num_scales,input_blob,labels_blob,device)
        feedforward_curve = train_feedforward_curve(num_scales,input_curve,labels_curve,device)
        
        filename = os.path.join(results_folder,'FF_blob_' + str(i) + '.pt')
        torch.save(feedforward_blob,filename)
        
        filename = os.path.join(results_folder,'FF_curve_' + str(i) + '.pt')
        torch.save(feedforward_curve,filename)
