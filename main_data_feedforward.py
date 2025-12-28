# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:59:11 2023

@author: Sami
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:34:33 2023

@author: Sami
"""

import torch
import numpy as np
import random
import os
import datetime
import pickle
from feedforward_data import *

    
device = torch.device('cpu')

if os.name == 'nt':
   batch_id = 0
else:
    batch_id = os.environ["SLURM_PROCID"]


seed = int(batch_id) + datetime.datetime.now().microsecond
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
    
if __name__ == '__main__':
    
    results_folder = os.path.join('multiscale','results','feedforward_data')
        
    data = make_data_feedforward(device,num_scales)

    filename = os.path.join(results_folder,'data_' + batch_id + '.pkl')
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

