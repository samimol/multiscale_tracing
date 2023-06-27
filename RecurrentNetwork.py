# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:28:45 2023

@author: Sami
"""

from Layers import InputLayer, HiddenLayer, OutputLayer, FFLayer
import torch
import numpy as np

class RecurrentNetwork():

    def __init__(self, n_input_features,grid_size,device,feedforward_curve,feedforward_object,one_scale,num_scales):
        self.beta = 0.02
        self.n_input_features = n_input_features
        self.n_hidden_features = 1
        
        self.duration = 30

        self.high_feature = 6

        self.exploitation_probability = 0.95

        self.grid_size = grid_size

        self.device = device
        
        self.one_scale = one_scale
        
        self.num_scales = num_scales
        
        if num_scales == 1:
            raise Exception("Minimum two scales")

        self.layers_list = [InputLayer(self.n_input_features, self.n_hidden_features)]
        self.RF_size_list = [1] + [3**i for i in range(self.num_scales)]
        
        RF_size = 3
        for i in range(num_scales):
            if i == 0:
                self.layers_list.append(HiddenLayer(self.n_input_features, self.n_hidden_features, self.high_feature,RF_size,grid_size,change_scale_fb=True))
            elif i == 1 and num_scales > 2:
                self.layers_list.append(HiddenLayer(self.n_hidden_features, self.high_feature,self.high_feature,RF_size,grid_size,change_scale_ff=True,higher_scale=True,change_scale_fb=True,upper_ymod=True))
            elif i < num_scales - 1:
                self.layers_list.append(HiddenLayer(self.high_feature, self.high_feature,self.high_feature,RF_size,grid_size,change_scale_ff=True,higher_scale=True,change_scale_fb=True,upper_ymod=True))
            elif i == num_scales - 1:
                self.layers_list.append(HiddenLayer(self.high_feature,self.high_feature,self.high_feature,RF_size,grid_size,upper_ymod = False,change_scale_ff=True))
        self.layers_list.append(OutputLayer(self.high_feature,self.n_hidden_features, self.n_input_features, 1,self.grid_size,self.RF_size_list))
        self.feedforward_network_curve = FFLayer(feedforward_curve.feedforward,feedforward_curve.feedforward_interm,num_scales)
        self.feedforward_network_object = FFLayer(feedforward_object.feedforward,feedforward_object.feedforward_interm,num_scales)

        self.task = 'trace_curve'

        self.save_activities = False
        
        self.saved_activities = [[] for i in range(num_scales+2)]
        
        self.to()

        
    def do_step(self, input_env, reward, reset_traces, device):

        if self.save_activities:
            for layer in range(len(self.saved_activities)):
                self.saved_activities[layer].append([])

        self.pyramidal_recurrent = [torch.zeros(1, self.n_input_features, self.grid_size, self.grid_size, device=self.device)]
        self.VIP = [torch.zeros(1, self.n_input_features, self.grid_size, self.grid_size, device=self.device)]
        self.SOM = [torch.zeros(1, self.n_input_features, self.grid_size, self.grid_size, device=self.device)]
        
        for i in range(self.num_scales):
            if i == 0:
                self.pyramidal_recurrent.append(torch.zeros(1, self.n_hidden_features, self.grid_size, self.grid_size, device=self.device))
                self.VIP.append(torch.zeros(1, self.n_hidden_features, self.grid_size, self.grid_size, device=self.device))
                self.SOM.append(torch.zeros(1, self.n_hidden_features, self.grid_size, self.grid_size, device=self.device))
            else:
                self.pyramidal_recurrent.append(torch.zeros(1,self.high_feature, self.grid_size//self.RF_size_list[i+1], self.grid_size//self.RF_size_list[i+1], device=self.device))
                self.VIP.append(torch.zeros(1,self.high_feature, self.grid_size//self.RF_size_list[i+1], self.grid_size//self.RF_size_list[i+1], device=self.device))
                self.SOM.append(torch.zeros(1,self.high_feature, self.grid_size//self.RF_size_list[i+1], self.grid_size//self.RF_size_list[i+1], device=self.device))
        
        i = 0
        norm = 10
        flag = True

        if self.task == 'trace_curve':
            self.pyramidal_feedforward = self.feedforward_network_curve(input_env)
        elif self.task == 'trace_object':
            self.pyramidal_feedforward = self.feedforward_network_object(input_env)
        else:
            raise Exception("Task property can be either trace_curve or trace_object")
        self.pyramidal_feedforward = [None] + self.pyramidal_feedforward
            
        while (i < self.duration and norm > 0) or flag:

            prev_low = self.pyramidal_recurrent[1].detach()

            (self.pyramidal_recurrent[0],self.VIP[0],self.SOM[0]) = self.layers_list[0].forward(self.pyramidal_recurrent[1],input_env)
            for layer in range(1,len(self.pyramidal_recurrent)-1):
                (self.pyramidal_recurrent[layer],self.VIP[layer],self.SOM[layer]) = self.layers_list[layer].forward(self.pyramidal_feedforward[layer],lower_ymod = self.pyramidal_recurrent[layer-1],upper_ymod = self.pyramidal_recurrent[layer + 1],horiz=self.pyramidal_recurrent[layer])
            (self.pyramidal_recurrent[-1],self.VIP[-1],self.SOM[-1]) = self.layers_list[layer+1].forward(self.pyramidal_feedforward[layer+1],lower_ymod = self.pyramidal_recurrent[-2],upper_ymod = None,horiz=self.pyramidal_recurrent[-1])


            if self.save_activities:
                for layer in range(len(self.saved_activities)-1):
                    self.saved_activities[layer][len(self.saved_activities[layer]) - 1].append(self.pyramidal_recurrent[layer].detach())
          
                self.Z = self.calc_output(device)
                
                self.saved_activities[-1][len(self.saved_activities[-1]) - 1].append(self.Z.detach())

            with torch.no_grad():
                norm = torch.linalg.norm(self.pyramidal_recurrent[1][0, :, :]-prev_low[0, :, :])

            i += 1
            if i == 6:
              flag = False
              
              
        self.prev_prev_pyramidal = [None] * len(self.pyramidal_recurrent)
        self.prev_prev_VIP = [None] * len(self.VIP)
        self.prev_prev_SOM = [None] * len(self.SOM)
        
        self.prev_pyramidal = [None] * len(self.pyramidal_recurrent)
        self.prev_VIP = [None] * len(self.VIP)
        self.prev_SOM = [None] * len(self.SOM)
        
        (self.prev_prev_pyramidal[0],self.prev_prev_VIP[0],self.prev_prev_SOM[0]) = self.layers_list[0].forward(self.pyramidal_recurrent[1],input_env)
        for layer in range(1,len(self.pyramidal_recurrent)-1):
            (self.prev_prev_pyramidal[layer],self.prev_prev_VIP[layer],self.prev_prev_SOM[layer]) = self.layers_list[layer].forward(self.pyramidal_feedforward[layer],lower_ymod = self.pyramidal_recurrent[layer-1],upper_ymod = self.pyramidal_recurrent[layer + 1],horiz=self.pyramidal_recurrent[layer])
        (self.prev_prev_pyramidal[-1],self.prev_prev_VIP[-1],self.prev_prev_SOM[-1]) = self.layers_list[layer+1].forward(self.pyramidal_feedforward[layer+1],lower_ymod = self.pyramidal_recurrent[-2],upper_ymod = None,horiz=self.pyramidal_recurrent[-1])
                      

        (self.prev_pyramidal[0],self.prev_VIP[0],self.prev_SOM[0]) = self.layers_list[0].forward(self.prev_prev_pyramidal[1],input_env)
        for layer in range(1,len(self.pyramidal_recurrent)-1):
            (self.prev_pyramidal[layer],self.prev_VIP[layer],self.prev_SOM[layer]) = self.layers_list[layer].forward(self.pyramidal_feedforward[layer],lower_ymod = self.prev_pyramidal[layer-1],upper_ymod = self.prev_prev_pyramidal[layer + 1],horiz=self.prev_prev_pyramidal[layer])
        (self.prev_pyramidal[-1],self.prev_VIP[-1],self.prev_SOM[-1]) = self.layers_list[layer+1].forward(self.pyramidal_feedforward[layer+1],lower_ymod = self.prev_pyramidal[-2],upper_ymod = None,horiz=self.prev_prev_pyramidal[-1])
              
        self.pyramidal_recurrent_detached = self.detach_and_reattach(self.prev_pyramidal)
        self.VIP_detached = self.detach_and_reattach(self.prev_VIP)
        self.SOM_detached = self.detach_and_reattach(self.prev_SOM)


        (self.pyramidal_recurrent[0],self.VIP[0],self.SOM[0]) = self.layers_list[0].forward(self.pyramidal_recurrent_detached[1],input_env)
        for layer in range(1,len(self.pyramidal_recurrent)-1):
            (self.pyramidal_recurrent[layer],self.VIP[layer],self.SOM[layer]) = self.layers_list[layer].forward(self.pyramidal_feedforward[layer],lower_ymod = self.pyramidal_recurrent[layer-1],upper_ymod = self.pyramidal_recurrent_detached[layer + 1],horiz=self.pyramidal_recurrent_detached[layer])
        (self.pyramidal_recurrent[-1],self.VIP[-1],self.SOM[-1]) = self.layers_list[layer+1].forward(self.pyramidal_feedforward[layer+1],lower_ymod = self.pyramidal_recurrent[-2],upper_ymod = None,horiz=self.pyramidal_recurrent_detached[-1])
                                                                                    
        
        self.Z = self.calc_output(device)

        if self.save_activities:
            self.saved_activities[-1][len(self.saved_activities[-1]) - 1].append(self.Z.detach())

        with torch.no_grad():
            action_chosen = torch.zeros((1, 1, 2+self.grid_size**2))
            action_chosen[0, 0, self.index_selected] = 1

        return (action_chosen)

    def calc_output(self, device):
        Z = self.layers_list[-1].forward(self.pyramidal_recurrent)
        with torch.no_grad():
            ZZ =torch.flatten(Z.permute(0,1,3,2), start_dim=2) #Flatten in F order beause everything is in F order
            if np.random.rand() < self.exploitation_probability:
                winner = self.calc_maxQ(ZZ)
            else:
                ZZint = ZZ.detach()
                ZZint -= torch.max(ZZint)
                ZZint = torch.exp(ZZint) / torch.sum(torch.exp(ZZint))
                winner = self.calc_softWTA(ZZint, device)
            self.index_selected = winner # index_selected represents the index of the action chosen on a flattened grid
            winner = [torch.tensor([0]), torch.tensor([0]), torch.tensor([(winner)%self.grid_size]), torch.tensor([torch.div(winner,self.grid_size,rounding_mode='floor')])]
            self.action = winner # action represents the coordinates of the action chosen

        return Z

    def calc_maxQ(self, Z):
        winner = torch.where(Z == torch.max(Z))[-1]
        # Break ties randomly
        if len(winner) > 1:
            tiebreak = winner[torch.randint(0, len(winner), (1,))]
            winner = tiebreak
        return winner

    def calc_softWTA(self, probabilities, device):
        # Create wheel:
        probs = torch.cumsum(probabilities, 2)[0][0]

        # Select from wheel
        rnd = torch.rand((1,), device=self.device)
        for (i, prob) in enumerate(probs):
            if rnd <= prob:
                return i

    def accessory(self):

        init_pyramidal = torch.autograd.grad(self.Z[self.action], self.pyramidal_recurrent, retain_graph=True, allow_unused=True)
        init_VIP = torch.autograd.grad(self.Z[self.action], self.VIP, retain_graph=True, allow_unused=True)
        init_SOM = torch.autograd.grad(self.Z[self.action], self.SOM, retain_graph=True, allow_unused=True)
        
        Z_pyramidal = [init_pyramidal[i] for i in range(len(init_pyramidal))]
        Z_VIP = [init_VIP[i] for i in range(len(init_VIP))]
        Z_SOM = [init_SOM[i] for i in range(len(init_SOM))]

        for i in range(7):

            for layer in range(len(Z_pyramidal)-1,-1,-1):
                Z_VIP[layer] = torch.autograd.grad(self.SOM[layer], self.VIP[layer], grad_outputs=Z_SOM[layer], retain_graph=True, allow_unused=True)[0]           
                Z_VIP[layer] = Z_VIP[layer] + init_VIP[layer]
                
                Z_SOM[layer] =  torch.autograd.grad(self.pyramidal_recurrent[layer], self.SOM[layer], grad_outputs=Z_pyramidal[layer], retain_graph=True, allow_unused=True)[0]           
                Z_SOM[layer] = Z_SOM[layer] + init_SOM[layer] 
                
                if layer != 0:
                    Z_pyramidal[layer] = torch.autograd.grad(self.prev_VIP[layer], self.prev_prev_pyramidal[layer], grad_outputs=Z_VIP[layer], retain_graph=True, allow_unused=True)[0]
                    Z_pyramidal[layer] = Z_pyramidal[layer] + torch.autograd.grad(self.prev_VIP[layer-1], self.prev_prev_pyramidal[layer], grad_outputs=Z_VIP[layer-1], retain_graph=True, allow_unused=True)[0]
                if layer != len(Z_pyramidal) - 1:
                    if layer == 0:
                        Z_pyramidal[layer] = torch.autograd.grad(self.pyramidal_recurrent[layer+1], self.pyramidal_recurrent[layer], grad_outputs=Z_pyramidal[layer+1], retain_graph=True, allow_unused=True)[0]
                    else:
                        Z_pyramidal[layer] = Z_pyramidal[layer] + torch.autograd.grad(self.pyramidal_recurrent[layer+1], self.pyramidal_recurrent[layer], grad_outputs=Z_pyramidal[layer+1], retain_graph=True, allow_unused=True)[0]
                Z_pyramidal[layer] = Z_pyramidal[layer] + init_pyramidal[layer]


        return (Z_pyramidal,Z_VIP,Z_SOM)

    def do_learn(self, reward):
        with torch.no_grad():
            exp_value = self.Z[self.action]
            self.delta = reward - exp_value

        (Z_pyramidal,Z_VIP,Z_SOM) = self.accessory()

        self.layers_list[0].update_layer([self.pyramidal_recurrent[0],self.SOM[0],self.VIP[0]],[Z_pyramidal[0],Z_SOM[0],Z_VIP[0]],self.beta,self.delta)
        for layer in range(1,len(self.layers_list)-1):
            self.layers_list[layer].update_layer([self.pyramidal_recurrent[layer],self.SOM[layer],self.VIP[layer]],[Z_pyramidal[layer],Z_SOM[layer],Z_VIP[layer]],self.beta,self.delta,train_v=False)
        self.layers_list[-1].update_layer(self.Z[self.action], self.beta, self.delta)

    def detach_and_reattach(self, x):
        detached = [xx.detach() for xx in x]
        for i, xx in enumerate(detached):
            xx.requires_grad = True
        return(detached)

    def to(self):
        for layer in range(len(self.layers_list)):
            self.layers_list[layer].to(self.device)

