# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:28:45 2023

@author: Sami
"""

from Layers import InputLayer, HiddenLayer, OutputLayer, FFLayer
import torch
import numpy as np

class RecurrentNetwork():

    def __init__(self, n_input_features,grid_size,big_pixels_size,bigger_pixels_size,device,feedforward_curve,feedforward_object):
        self.beta = 0.02
        self.n_input_features = n_input_features
        self.n_hidden_features = 1
        
        self.duration = 30

        self.high_feature = 6

        self.big_pixels_size = big_pixels_size

        self.bigger_pixels_size = bigger_pixels_size

        self.exploitation_probability = 0.95

        self.grid_size = grid_size

        self.device = device


        self.input_layer = InputLayer(self.n_input_features, self.n_hidden_features,self.big_pixels_size)
        self.low_scale_layer = HiddenLayer(self.n_input_features, self.n_hidden_features, 6,self.big_pixels_size,grid_size,change_scale_u=True)

        self.middle_scale_layer = HiddenLayer(self.n_hidden_features, 6,self.high_feature,self.big_pixels_size,grid_size,change_scale_v=True,higher_scale=True,change_scale_u=True,upper_ymod=True)

        self.high_scale_layer = HiddenLayer(self.high_feature,6,self.high_feature,self.bigger_pixels_size//self.big_pixels_size,grid_size,upper_ymod = False,change_scale_v=True)

        self.output_layer = OutputLayer(self.n_hidden_features, self.n_input_features, 1,self.grid_size,self.big_pixels_size,self.bigger_pixels_size)
        self.feedforward_network_curve = FFLayer(feedforward_curve.low_scale_feedforward,feedforward_curve.middle_scale_feedforward_interm,feedforward_curve.middle_scale_feedforward,feedforward_curve.high_scale_feedforward_interm,feedforward_curve.high_scale_feedforward)
        self.feedforward_network_object = FFLayer(feedforward_object.low_scale_feedforward,feedforward_object.middle_scale_feedforward_interm,feedforward_object.middle_scale_feedforward,feedforward_object.high_scale_feedforward_interm,feedforward_object.high_scale_feedforward)

        self.task = 'trace_curve'

        self.save_activities = False
        
        self.saveXmod = []
        self.saveY2mod = []
        self.saveY3mod = []
        self.saveY6mod = []

        self.saveQ = []
        
        self.to()

        
    def do_step(self, input_env, reward, reset_traces, device):

        if self.save_activities:
            self.saveXmod.append([])
            self.saveY2mod.append([])
            self.saveY3mod.append([])
            self.saveY6mod.append([])
            self.saveQ.append([])

        self.Xmod = torch.zeros(1, self.n_input_features, self.grid_size, self.grid_size, device=self.device)
        self.VIP0 = torch.zeros(1, self.n_input_features, self.grid_size, self.grid_size, device=self.device)
        self.SOM0 = torch.zeros(1, self.n_input_features, self.grid_size, self.grid_size, device=self.device)

        self.Y2 = torch.zeros(1, self.n_hidden_features, self.grid_size, self.grid_size, device=self.device)
        self.VIP2 = torch.zeros(1, self.n_hidden_features, self.grid_size, self.grid_size, device=self.device)
        self.SOM2 = torch.zeros(1, self.n_hidden_features, self.grid_size, self.grid_size, device=self.device)
        self.Y2mod = torch.zeros(1, self.n_hidden_features, self.grid_size, self.grid_size, device=self.device)

        self.Y3 = torch.zeros(1,self.high_feature, self.grid_size//self.big_pixels_size, self.grid_size//self.big_pixels_size, device=self.device)
        self.VIP3 = torch.zeros(1, self.high_feature, self.grid_size//self.big_pixels_size, self.grid_size//self.big_pixels_size, device=self.device)
        self.SOM3 = torch.zeros(1, self.high_feature, self.grid_size//self.big_pixels_size, self.grid_size//self.big_pixels_size, device=self.device)
        self.Y3mod = torch.zeros(1, self.high_feature, self.grid_size//self.big_pixels_size, self.grid_size//self.big_pixels_size, device=self.device)

        self.Y6 = torch.zeros(1,6, self.grid_size//self.bigger_pixels_size, self.grid_size//self.bigger_pixels_size, device=self.device)
        self.VIP6 = torch.zeros(1, 6, self.grid_size//self.bigger_pixels_size, self.grid_size//self.bigger_pixels_size, device=self.device)
        self.SOM6 = torch.zeros(1, 6, self.grid_size//self.bigger_pixels_size, self.grid_size//self.bigger_pixels_size, device=self.device)
        self.Y6mod = torch.zeros(1, 6, self.grid_size//self.bigger_pixels_size, self.grid_size//self.bigger_pixels_size, device=self.device)

        i = 0
        norm = 10
        flag = True

        if self.task == 'trace_curve':
            self.low_scale,self.middle_scale,self.high_scale = self.feedforward_network_curve(input_env[0])
        elif self.task == 'trace_object':
            self.low_scale,self.middle_scale,self.high_scale = self.feedforward_network_object(input_env[0])
        else:
            raise Exception("Task property can be either trace_curve or trace_object")
            
        while (i < self.duration and norm > 0) or flag:

            prevY2mod = self.Y2mod.detach()
            (self.Xmod,self.VIP0,self.SOM0) = self.input_layer.forward(self.Y2, self.Y2mod, input_env)
            (self.Y2mod,self.VIP2,self.SOM2) = self.low_scale_layer.forward(self.low_scale, lower_ymod=self.Xmod,upper_ymod=self.Y3mod,horiz = self.Y2mod)
            (self.Y3mod,self.VIP3,self.SOM3) = self.middle_scale_layer.forward(self.middle_scale, lower_ymod=self.Y2mod, upper_ymod=self.Y6mod,horiz = self.Y3mod)
            (self.Y6mod,self.VIP6,self.SOM6) = self.high_scale_layer.forward(self.high_scale, lower_ymod=self.Y3mod, upper_ymod=None,horiz = self.Y6mod)
 

            if self.save_activities:
                self.saveXmod[len(self.saveXmod) - 1].append(self.Xmod.detach())
                self.saveY2mod[len(self.saveY2mod) - 1].append(self.Y2mod.detach())
                self.saveY3mod[len(self.saveY3mod) - 1].append(self.Y3mod.detach())
                self.saveY6mod[len(self.saveY6mod) - 1].append(self.Y6mod.detach())
          
                self.Z = self.calc_Output(device)
                self.saveQ[len(self.saveQ) - 1].append(self.Z.detach())

            with torch.no_grad():
                norm = torch.linalg.norm(self.Y2mod[0, :, :]-prevY2mod[0, :, :])

            i += 1
            if i == 6:
              flag = False

        prevY2mod = self.Y2mod.detach()
        (self.Xmod3,self.VIP03,self.SOM03) = self.input_layer.forward(self.Y2, self.Y2mod, input_env)
        (self.Y2mod3,self.VIP23,self.SOM23) = self.low_scale_layer.forward(self.low_scale, lower_ymod=self.Xmod,upper_ymod=self.Y3mod,horiz = self.Y2mod)
        (self.Y3mod3,self.VIP33,self.SOM33) = self.middle_scale_layer.forward(self.middle_scale, lower_ymod=self.Y2mod,upper_ymod=self.Y6mod,horiz = self.Y3mod)
        (self.Y6mod3,self.VIP63,self.SOM63) = self.high_scale_layer.forward(self.high_scale, lower_ymod=self.Y3mod, upper_ymod=None,horiz = self.Y6mod)

        prevY2mod = self.Y2mod.detach()
        (self.Xmod2,self.VIP02,self.SOM02) = self.input_layer.forward(self.Y2, self.Y2mod3, input_env)
        (self.Y2mod2,self.VIP22,self.SOM22) = self.low_scale_layer.forward(self.low_scale, lower_ymod=self.Xmod2,upper_ymod=self.Y3mod3,horiz = self.Y2mod3)
        (self.Y3mod2,self.VIP32,self.SOM32) = self.middle_scale_layer.forward(self.middle_scale, lower_ymod=self.Y2mod2,upper_ymod=self.Y6mod3,horiz = self.Y3mod3)
        (self.Y6mod2,self.VIP62,self.SOM62) = self.high_scale_layer.forward(self.high_scale, lower_ymod=self.Y3mod2, upper_ymod=None,horiz = self.Y6mod3)


        ([self.Y6mod2d,self.VIP62d,self.SOM62d,
          self.Y3mod2d,self.VIP32d,self.SOM32d,
          self.Y2mod2d,self.VIP22d,self.SOM22d,
          self.Xmod2d,self.VIP02d,self.SOM02d]) = self.detachAndreattach([self.Y6mod2,self.VIP62,self.SOM62,
                                                                          self.Y3mod2,self.VIP32,self.SOM32,
                                                                    self.Y2mod2,self.VIP22,self.SOM22,
                                                                    self.Xmod2,self.VIP02,self.SOM02])

        prevY2mod = self.Y2mod.detach()
        (self.Xmod,self.VIP0,self.SOM0) = self.input_layer.forward(self.Y2, self.Y2mod2d, input_env)
        (self.Y2mod,self.VIP2,self.SOM2) = self.low_scale_layer.forward(self.low_scale, lower_ymod=self.Xmod,upper_ymod=self.Y3mod2d,horiz = self.Y2mod2d)
        (self.Y3mod,self.VIP3,self.SOM3) = self.middle_scale_layer.forward(self.middle_scale, lower_ymod=self.Y2mod,upper_ymod=self.Y6mod2d,horiz = self.Y3mod2d)
        (self.Y6mod,self.VIP6,self.SOM6) = self.high_scale_layer.forward(self.high_scale, lower_ymod=self.Y3mod, upper_ymod=None,horiz = self.Y6mod2d)

        
        self.Z = self.calc_Output(device)

        if self.save_activities:
            self.saveQ[len(self.saveQ) - 1].append(self.Z.detach())

        with torch.no_grad():
            action_chosen = torch.zeros((1, 1, 2+self.grid_size**2))
            action_chosen[0, 0, self.index_selected] = 1

        return (action_chosen)

    def calc_Output(self, device):
        Z = self.output_layer.forward(self.Xmod,self.Y2mod,self.Y3mod,self.Y6mod)
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

    def Accessory(self):

        init = torch.autograd.grad(self.Z[self.action], [self.Y3mod,self.VIP3,self.SOM3,
                                                         self.Y2mod,self.VIP2,self.SOM2,
                                                         self.Xmod,self.VIP0,self.SOM0,
                                                         self.Y6mod,self.VIP6,self.SOM6,
                                                         ], retain_graph=True, allow_unused=True)


        Zy3mod = init[0]
        Zvip3 = init[1]
        Zsom3 = init[2]

        Zy2mod = init[3]
        Zvip2 = init[4]
        Zsom2 = init[5]

        Zxmod = init[6]
        Zvip0 = init[7]
        Zsom0 = init[8]

        Zy6mod = init[9]
        Zvip6 = init[10]
        Zsom6 = init[11]

        for i in range(7):

            Zvip6 = torch.autograd.grad(self.SOM6, self.VIP6, grad_outputs=Zsom6, retain_graph=True, allow_unused=True)[0]
            Zvip6 = Zvip6 + init[10]

            Zsom6 = torch.autograd.grad(self.Y6mod, self.SOM6, grad_outputs=Zy6mod, retain_graph=True, allow_unused=True)[0]
            Zsom6 = Zsom6 + init[11]

            Zy6mod = torch.autograd.grad(self.VIP62, self.Y6mod3, grad_outputs=Zvip6, retain_graph=True, allow_unused=True)[0]
            Zy6mod = Zy6mod + torch.autograd.grad(self.VIP32, self.Y6mod3, grad_outputs=Zvip3, retain_graph=True, allow_unused=True)[0]
            Zy6mod = Zy6mod + init[9]


            Zvip3 = torch.autograd.grad(self.SOM3, self.VIP3, grad_outputs=Zsom3, retain_graph=True, allow_unused=True)[0]
            Zvip3 = Zvip3 + init[1]

            Zsom3 = torch.autograd.grad(self.Y3mod, self.SOM3, grad_outputs=Zy3mod, retain_graph=True, allow_unused=True)[0]
            Zsom3 = Zsom3 + init[2]

            Zy3mod = torch.autograd.grad(self.VIP32, self.Y3mod3, grad_outputs=Zvip3, retain_graph=True, allow_unused=True)[0]
            Zy3mod = Zy3mod + torch.autograd.grad(self.VIP22, self.Y3mod3, grad_outputs=Zvip2, retain_graph=True, allow_unused=True)[0]
            Zy3mod = Zy3mod + torch.autograd.grad(self.Y6mod, self.Y3mod, grad_outputs=Zy6mod, retain_graph=True, allow_unused=True)[0]
            Zy3mod = Zy3mod + init[0]

            Zvip2 = torch.autograd.grad(self.SOM2, self.VIP2, grad_outputs=Zsom2, retain_graph=True, allow_unused=True)[0]
            Zvip2 = Zvip2 + init[4]

            Zsom2 = torch.autograd.grad(self.Y2mod, self.SOM2, grad_outputs=Zy2mod, retain_graph=True, allow_unused=True)[0]
            #Zsom2 = Zsom2 + torch.autograd.grad(self.VIP22, self.SOM23, grad_outputs=Zvip2, retain_graph=True, allow_unused=True)[0]
            Zsom2 = Zsom2 + init[5]

            Zy2mod = torch.autograd.grad(self.VIP22, self.Y2mod3, grad_outputs=Zvip2, retain_graph=True, allow_unused=True)[0]
            Zy2mod = Zy2mod + torch.autograd.grad(self.VIP02, self.Y2mod3, grad_outputs=Zvip0, retain_graph=True, allow_unused=True)[0]
            Zy2mod = Zy2mod + torch.autograd.grad(self.Y3mod, self.Y2mod, grad_outputs=Zy3mod, retain_graph=True, allow_unused=True)[0]
            Zy2mod = Zy2mod + init[3]

            Zvip0 = torch.autograd.grad(self.SOM0, self.VIP0, grad_outputs=Zsom0, retain_graph=True, allow_unused=True)[0]
            Zvip0 = Zvip0 + init[7]

            Zsom0 = torch.autograd.grad(self.Xmod, self.SOM0, grad_outputs=Zxmod, retain_graph=True, allow_unused=True)[0]
            Zsom0 = Zsom0 + init[8]

            Zxmod = torch.autograd.grad(self.Y2mod, self.Xmod, grad_outputs=Zy2mod, retain_graph=True, allow_unused=True)[0]
            Zxmod = Zxmod + init[6]

        return (Zy6mod,Zvip6,Zsom6,
            Zy3mod,Zvip3,Zsom3,
                Zy2mod,Zvip2,Zsom2,
                Zxmod,Zvip0,Zsom0)

    def do_learn(self, reward):
        with torch.no_grad():
            exp_value = self.Z[self.action]
            self.delta = reward - exp_value

        (Zy6mod,Zvip6,Zsom6,
            Zy3mod,Zvip3,Zsom3,
                Zy2mod,Zvip2,Zsom2,
                Zxmod,Zvip0,Zsom0) = self.Accessory()

        self.input_layer.update_layer([self.Xmod,self.SOM0,self.VIP0], [Zxmod,Zsom0,Zvip0], self.beta, self.delta)
        self.low_scale_layer.update_layer([self.Y2,self.Y2mod,self.SOM2,self.VIP2], [None,Zy2mod,Zsom2,Zvip2], self.beta, self.delta,train_v=False)
        self.middle_scale_layer.update_layer([self.Y3,self.Y3mod,self.SOM3,self.VIP3], [None,Zy3mod,Zsom3,Zvip3], self.beta, self.delta,train_v=False)
        self.high_scale_layer.update_layer([self.Y6,self.Y6mod,self.SOM6,self.VIP6], [None,Zy6mod,Zsom6,Zvip6], self.beta, self.delta)
        self.output_layer.update_layer(self.Z[self.action], self.beta, self.delta)

    def detachAndreattach(self, x):
        detached = [xx.detach() for xx in x]
        for i, xx in enumerate(detached):
            xx.requires_grad = True
        return(detached)

    def to(self):
        self.input_layer.to(self.device)
        self.low_scale_layer.to(self.device)
        self.middle_scale_layer.to(self.device)
        self.high_scale_layer.to(self.device)
        self.output_layer.to(self.device)

