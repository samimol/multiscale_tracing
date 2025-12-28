# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 15:56:04 2023

@author: Sami
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim

class FeedforwardNetwork(nn.Module): 
    def __init__(self,device,num_scales):
        super().__init__()

        self.num_scales = num_scales

        feats = 20
        
        self.feedforward = []
        self.feedforward_interm = [None] #dummy module not used but just to have the right index 
        
        self.receptive_field_sizes = [3**i for i in range(num_scales)]
        
        for i in range(num_scales):
            if i == 0:
                self.feedforward.append(nn.Conv2d(3, 1, 1,stride=1,padding='same',bias=False,device=device))
                self.feedforward[0].weight = torch.nn.Parameter(torch.rand(self.feedforward[0].weight.shape))
            else:
                self.feedforward_interm.append(nn.Conv2d(1,feats, self.receptive_field_sizes[i], stride=1,padding='same', bias=True,device=device)    )
                self.feedforward.append(nn.Conv2d(feats,1, self.receptive_field_sizes[i] ,stride=self.receptive_field_sizes[i], bias=True,device=device))
        
        self.feedforward = nn.ModuleList(self.feedforward)
        self.feedforward_interm = nn.ModuleList(self.feedforward_interm)
        
        self.sig = nn.Sigmoid()
        
        self.loss = []
               
    def forward(self, x):
        
        intern_representation = [None] * (self.num_scales - 1)
        x = F.relu(self.feedforward[0](x))
        for layer in range(1,self.num_scales):
            interm = F.relu(self.feedforward_interm[layer](x))
            intern_representation[layer - 1] = self.sig(self.feedforward[layer](interm))
        
        return intern_representation
    
    
    def train_network(self,optimizer,criterion,input_list,labels,epochs=2,verbose=False,print_frequency=2000,batch_size=1):
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            count = 0
            permutation = list(range(len(labels[0])))
            random.shuffle(permutation)
            for i in range(0,len(labels[0]),batch_size):
                count += 1
                index = permutation[i]
                # get the inputs; data is a list of [inputs, labels]
                batch_input = input_list[index:index+batch_size,:,:,:]
                #label = labels[i:i+batch_size,:]
    
                # zero the parameter gradients
                optimizer.zero_grad()
    
                # forward + backward + optimize
                output = self.forward(batch_input)

                loss = 0
                for p in range(len(criterion)):
                    loss += criterion[p](output[p], labels[p][index:index+batch_size,:])
                loss.backward()
                optimizer.step()
    
                running_loss += loss.item()
                self.loss.append(loss.cpu().detach().numpy())
                if verbose:
                    if count % 10 == 0:    # print every 2000 mini-batches
                        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.15f}')
                        running_loss = 0.0


def train_feedforward_blob(num_scales,input_blob,labels_blob,device):
    feedforward_blob = FeedforwardNetwork(device,num_scales)
    feedforward_blob = feedforward_blob.to(device)
    criterion = [nn.BCELoss() for i in range(num_scales-1)]
    optimizer = optim.Adam(feedforward_blob.parameters(), lr=0.001)
    feedforward_blob.train_network(optimizer,criterion,input_blob.to(device),[labels_blob[i].to(device) for i in range(labels_blob)],epochs=80,verbose=False,batch_size=256)
    return(feedforward_blob)

def train_feedforward_curve(num_scales,input_curve,labels_curve,device):
    feedforward_curve = FeedforwardNetwork(device,num_scales)
    feedforward_curve = feedforward_curve.to(device)
    criterion = [nn.BCELoss() for i in range(num_scales-1)]
    optimizer = optim.Adam(feedforward_curve.parameters(), lr=0.001)

    feedforward_curve.train_network(optimizer,criterion,torch.cat(input_curve,dim=0).to(device),[torch.cat(labels_curve[i],dim=0).to(device) for i in range(len(labels_curve))],epochs=80,verbose=False,batch_size=256)
    return(feedforward_curve)