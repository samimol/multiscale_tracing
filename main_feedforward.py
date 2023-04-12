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
from opts import parser
import pickle
from FF_data import *
from FeedforwardNetwork import *
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
        
    for i in range(args.num_networks):
        
        with open(os.path.join(data_folder,'data_' + str(i) + '.pkl'), 'rb') as input:
            (input_3_blob,labels_3_blob,labels_3_other_blob,input_9_blob,labels_9_blob,labels_9_other_blob,input_3_curve,labels_3_curve,labels_3_other_curve,input_9_curve,labels_9_curve,labels_9_other_curve) = pickle.load(input)
 
        feedforward_blob = train_feedforward_blob(input_3_blob,labels_3_blob,labels_3_other_blob,input_9_blob,labels_9_blob,labels_9_other_blob,device)
        feedforward_curve = train_feedforward_curve(input_3_curve,labels_3_curve,labels_3_other_curve,input_9_curve,labels_9_curve,labels_9_other_curve,device)
    
        filename = os.path.join(results_folder,'FF_blob_' + str(i) + '.pt')
        torch.save(feedforward_blob,filename)
        
        filename = os.path.join(results_folder,'FF_curve_' + str(i) + '.pt')
        torch.save(feedforward_curve,filename)
