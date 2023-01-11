# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 15:56:04 2023

@author: Sami
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class FeedforwardNetwork(nn.Module): 
    def __init__(self,device):
        super().__init__()

        feats = 6

        self.low_scale_feedforward = nn.Conv2d(4, 1, 1,stride=1,padding='same',bias=False,device=device)
        self.low_scale_feedforward.weight = torch.nn.Parameter(torch.rand(self.low_scale_feedforward.weight.shape))
        
        self.middle_scale_feedforward_interm = nn.Conv2d(1,feats, 3, stride=1,padding='same', bias=True,device=device)    
        self.middle_scale_feedforward = nn.Conv2d(feats,1, 3 ,stride=3, bias=True,device=device)
        
        
        self.high_scale_feedforward_interm = nn.Conv2d(1,feats, 9 ,stride=1,padding='same', bias=True,device=device)
        self.high_scale_feedforward = nn.Conv2d(feats,1, 9 ,stride=9, bias=True,device=device)

        
        self.sig = nn.Sigmoid()
               
    def forward(self, x):
        
        low_scale = F.relu(self.low_scale_feedforward(x))
        
        # Middle scale
        middle_scale_interm = F.relu(self.middle_scale_feedforward_interm(low_scale))
        middle_scale = self.sig(self.middle_scale_feedforward(middle_scale_interm))
                
        #High scale
        high_scale_interm = F.relu(self.high_scale_feedforward_interm(low_scale))
        high_scale = self.sig(self.high_scale_feedforward(high_scale_interm))
        
        return middle_scale, high_scale
    
    
    def train(self,optimizer,criterion,input_list,labels,epochs=2,verbose=False,print_frequency=2000):
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            count = 0
            batch_size = 1
            permut = list(range(len(labels[0])))
            random.shuffle(permut)
            for i in range(0,len(labels[0]),batch_size):
                count += 1
                index = permut[i]
                # get the inputs; data is a list of [inputs, labels]
                input = input_list[index:index+batch_size,:,:,:]
                #label = labels[i:i+batch_size,:]
    
                # zero the parameter gradients
                optimizer.zero_grad()
    
                # forward + backward + optimize
                out = self.forward(input)

                loss = 0
                for p in range(len(criterion)):
                    loss += criterion[p](out[p], labels[p][index:index+batch_size,:])
                loss.backward()
                optimizer.step()
    
                running_loss += loss.item()
                if verbose:
                    if count % print_frequency == print_frequency - 1:    # print every 2000 mini-batches
                        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.15f}')
                        running_loss = 0.0
