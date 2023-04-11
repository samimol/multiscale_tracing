# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:29:28 2021

@author: Sami
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLayer(nn.Module):

    def __init__(self):
        self.initialisation_range = 0.1
        super().__init__()

    def activation_function(self, x):
        return torch.relu(x)

    def step_function(self,x):
      eps = 1
      phi = 1      
      x[x <= eps/phi] = phi * x[x <= eps/phi]
      x[x <= 0] = 0
      x[x >= eps/phi] = eps
      return x


    def gating_function(self,x):
      param = 100
      return torch.relu(param*x/(torch.sqrt(1+(param**2)*(x**2))))

    def init_weights_FF(self, layer, one_to_one=False,change_scale=False,receptive_field_size=1):
        K = torch.zeros(layer.weight.shape)
        if layer.bias is not None:
            bias = torch.zeros(layer.bias.shape)
        if change_scale:
          for f in range(K.shape[0]):
            for f2 in range(K.shape[1]):
              K[f, f2, :, :] = self.initialisation_range * np.random.rand() + 1
        else:
          for f in range(K.shape[0]):
              for f2 in range(K.shape[1]):
                  if not one_to_one:
                      lower_bound = int(0.5*(2*self.grid_size-1-receptive_field_size))
                      upper_bound = int(lower_bound + receptive_field_size)
                      K[f, f2, lower_bound:upper_bound,lower_bound:upper_bound] = 0.001 * np.random.rand()
                  else:
                    if self.layer_type != "output":
                        K[f,f2,0] =  self.initialisation_range * np.random.rand() + 1
                    else:
                        K[f, f2, 0] = 0.001 * np.random.rand()
        if layer.bias is not None:
          for f in range(bias.shape[0]):  
                bias[f] = 1
        layer.weight = torch.nn.Parameter(K)
        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(bias)

    def init_weights_FB(self, layer, change_scale=False):
        K = torch.zeros(layer.weight.shape)
        if layer.bias is not None:
            bias = torch.zeros(layer.bias.shape)                                        
        for f in range(K.shape[0]):
              for f2 in range(K.shape[1]):
                if self.layer_type != "output":
                  if change_scale:
                    K[f, f2, :, :] = self.initialisation_range * np.random.rand() + 0.5
                  else:
                      K[f, f2, 0,1] = self.initialisation_range * np.random.rand() + 0.1
                      K[f, f2, 1,0] = self.initialisation_range * np.random.rand() + 0.1
                      K[f, f2, 1,2] = self.initialisation_range * np.random.rand() + 0.1
                      K[f, f2, 2,1] = self.initialisation_range * np.random.rand() + 0.1               
        layer.weight = torch.nn.Parameter(K)
        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(bias)


    def init_weights_inhib(self, layer):
        K = torch.zeros(layer.weight.shape)
        if layer.bias is not None:
            bias = torch.zeros(layer.bias.shape)
        for f in range(K.shape[0]):
              for f2 in range(K.shape[1]):
                if f == f2 :
                    K[f, f2, 1, 1] = 1
        if layer.bias is not None:
          for f in range(bias.shape[0]):  
              bias[f] =  -1 - self.initialisation_range * np.random.rand()
        layer.weight = torch.nn.Parameter(K)
        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(bias)


    def average_traces(self, traces, mask,receptive_field_size=1):
        traces *= mask
        if self.layer_type != "output":
            m = torch.mean(traces[:,:,:,:], axis=(2,3))
            m = m[:, :, None,None]
            traces[:,:] = m  
        else:
            intermmask = torch.ones((2 * self.grid_size - 1, 2 * self.grid_size - 1))
            lower_bound = int(0.5*(2*self.grid_size-1-receptive_field_size))
            upper_bound = int(lower_bound + receptive_field_size)
            
            intermmask = torch.zeros((2 * self.grid_size - 1, 2 * self.grid_size - 1))
            intermmask[lower_bound:upper_bound,lower_bound:upper_bound] = 1
            m = torch.mean(traces[:, :, intermmask == 1], axis=2)
            traces[:, :, intermmask == 1] =  m[:,None]    
        return(traces)


    def update_weight(self, layer, upper, beta, delta, mask=None, z=None, average=True,receptive_field_size=1,inhib=False,change_scale=False):
        with torch.no_grad():
            # Getting the derivative of the output unit with respect to the 
            # weight and bias
            if layer.bias is not None:
                delta_weight = torch.autograd.grad(upper, [layer.weight, layer.bias], grad_outputs=z, retain_graph=True, allow_unused=True)
                delta_bias = delta_weight[1]
                delta_bias = torch.clamp(delta_bias, None, 1)
                #delta_bias[:] = torch.mean(delta_bias)
            else:
                delta_weight = torch.autograd.grad(upper, layer.weight, grad_outputs=z, retain_graph=True, allow_unused=True)
            delta_weight = delta_weight[0]
            delta_weight = torch.clamp(delta_weight, None, 1)
            
            if inhib == False:
                # Averaging the weight and bias and updating the weights with RPE
                if (self.layer_type == 'output' and average) or change_scale:# or self.change_scale_fb:# or self.higher_scale:
                    delta_weight = self.average_traces(delta_weight, mask,receptive_field_size=receptive_field_size)
                elif mask is not None:
                    delta_weight = delta_weight*mask
                weight_update = layer.weight + beta * delta * delta_weight
                layer.weight.copy_(weight_update)
            else:
                bias_update = layer.bias + beta * delta * delta_bias
                layer.bias.copy_(bias_update)
        return(layer)
    
    def make_mask(self,weight):
        mask = torch.clone(weight.detach())
        mask[mask != 0] = 1
        return mask
        
    def to(self, device):
      if self.layer_type == 'input':
          self.umod.to(device)
          self.wie.to(device)
          self.feedback_mask = self.feedback_mask.to(device)
          self.inhib_mask = self.inhib_mask.to(device)
      elif self.layer_type == 'hidden':
          self.t.to(device)
          self.feedforward_mask = self.feedforward_mask.to(device)
          self.horizontal_mask = self.horizontal_mask.to(device)
          self.horiz.to(device)
          if self.upper_ymod:
              self.umod.to(device)
              self.feedback_mask = self.feedback_mask.to(device)
      elif self.layer_type == 'output':
          self.high_to_output.to(device)
          self.input_to_output.to(device)
          self.low_to_output.to(device)
          self.middle_to_output.to(device)
          self.high_to_output_mask = self.high_to_output_mask.to(device)
          self.low_to_output_mask = self.low_to_output_mask.to(device)
          self.middle_to_output_mask = self.middle_to_output_mask.to(device)
                
                
class InputLayer(CustomLayer):

    def __init__(self, feature_in, feature_out):
        super().__init__()
        self.layer_type = 'input'
        K_size = 3

        self.umod = nn.Conv2d(feature_out, feature_in, K_size, stride=1, padding='same',bias=False)
        self.wie = nn.Conv2d(feature_in, feature_in, K_size, stride=1, padding='same',bias=True)

        # Initializing weights
        self.init_weights_FB(self.umod)
        self.init_weights_inhib(self.wie)

        # Setting the mask to have connections only between neighboours
        self.feedback_mask = self.make_mask(self.umod.weight)
        self.inhib_mask = self.make_mask(self.wie.weight)

    def forward(self, upper_y, upper_ymod, input_stimulus):
        Y = input_stimulus
        VIP = self.step_function(self.umod(upper_ymod)) 
        SOM = self.activation_function(1 - VIP)
        Ymod =  self.activation_function((-self.wie(SOM)) * self.gating_function(Y))
        return(Ymod,VIP,SOM)


    def update_layer(self, upper, z, beta, delta):
        self.umod = self.update_weight(self.umod, upper[2], beta, delta, self.feedback_mask, z[2],change_scale=False)
        self.wie = self.update_weight(self.wie, upper[0], beta, delta, self.inhib_mask, z[0],inhib=True)

class HiddenLayer(CustomLayer):

    def __init__(self, feature_in_lower, feature_out,feature_in_higher,big_pixels_size,grid_size, upper_ymod=True,change_scale_fb=False,change_scale_ff=False,higher_scale=False):
        super().__init__()
        self.layer_type = 'hidden'
        self.upper_ymod = upper_ymod # If the higher layer has a modulated group
        self.change_scale_fb = change_scale_fb
        self.change_scale_ff = change_scale_ff
        self.higher_scale = higher_scale
        self.big_pixels_size = big_pixels_size
        self.grid_size = grid_size
        
        # Making the weights
        if self.change_scale_ff:
          self.t = nn.Conv2d(feature_in_lower, feature_out, self.big_pixels_size, stride=self.big_pixels_size, bias=True)
        else:
            self.t = nn.Conv2d(feature_in_lower, feature_out, 1, stride=1, padding='same', bias=True)     
        
        if upper_ymod:
            self.umod = nn.ConvTranspose2d(feature_in_higher, feature_out, self.big_pixels_size, stride=self.big_pixels_size, padding=0,bias=False)
        self.horiz = nn.Conv2d(feature_out,feature_out,3,stride = 1,padding='same',bias=False)
        
        # Initializing the weights
        self.init_weights_FF(self.t,change_scale = change_scale_ff,one_to_one=True)
        if upper_ymod:
            self.init_weights_FB(self.umod,change_scale = change_scale_fb)
        self.init_weights_FB(self.horiz,change_scale=False)

        # Making the masks        
        if upper_ymod:
            self.feedback_mask = self.make_mask(self.umod.weight)
        self.feedforward_mask = self.make_mask(self.t.weight)
        self.horizontal_mask = self.make_mask(self.horiz.weight)
   

    def forward(self,current_y, lower_ymod=None, upper_ymod=None,horiz=None):
        FF = self.t(lower_ymod)
        if upper_ymod is not None:
            VIP = self.step_function(self.umod(upper_ymod) + self.horiz(horiz)) #* self.gating_function(Y)
        else:
            VIP = self.step_function(self.horiz(horiz)) #* self.gating_function(Y)
        SOM = self.activation_function(1 - VIP)
        current_ymod =  self.activation_function((FF-SOM) * self.gating_function(current_y))
        return(current_ymod,VIP,SOM)
        
    def update_layer(self, upper, z, beta, delta,train_v=True):
        self.t = self.update_weight(self.t, upper[1], beta, delta, self.feedforward_mask, z[1],change_scale=self.change_scale_ff)
        if self.upper_ymod:
            self.umod = self.update_weight(self.umod, upper[3], beta, delta, self.feedback_mask, z[3],change_scale = self.change_scale_fb)
        self.horiz = self.update_weight(self.horiz, upper[3], beta, delta, self.horizontal_mask, z[3],change_scale = False)

class OutputLayer(CustomLayer):

    def __init__(self, hidden_features, input_features, feature_out,grid_size,big_pixels_size,bigger_pixels_size,one_scale):
        super().__init__()
        self.grid_size = grid_size
        K_size = 2 * self.grid_size - 1
        self.layer_type = 'output'
        self.grid_size = grid_size
        self.big_pixels_size = big_pixels_size
        self.bigger_pixels_size = bigger_pixels_size
        self.hidden_features = hidden_features
        self.feature_out = feature_out
        self.one_scale = one_scale
        
        self.high_to_output = nn.ConvTranspose2d(6, 1,2 * self.grid_size - 1, stride=self.bigger_pixels_size, padding=int(0.5*(2 * self.grid_size - 1 - self.bigger_pixels_size)),bias=False)#nn.ConvTranspose2d(6, feature_out, 3, stride=3, padding=0,bias=False)
        self.low_to_output = nn.Conv2d(hidden_features, feature_out, K_size, stride=1, padding='same',bias=False)
        self.middle_to_output = nn.ConvTranspose2d(6, 1,2 * self.grid_size - 1, stride=self.big_pixels_size, padding=int(0.5*(2 * self.grid_size - 1 - self.big_pixels_size)),bias=False)
        self.input_to_output = nn.Conv2d(input_features, feature_out, 1, stride=1, padding='same', bias=False)

        self.low_to_output_mask = torch.zeros_like(self.low_to_output.weight)  # In numpy we do the average over 49*49-49 = 2352 whereas here over 13*13 -1 = 168 so we have to divide by 14 (number of strides)
        self.low_to_output_mask[:, :, self.grid_size - 1, self.grid_size - 1] = 1/50

        self.middle_to_output_mask = torch.zeros_like(self.middle_to_output.weight)
        self.middle_to_output_mask[:, :,self.grid_size-2:self.grid_size+1,self.grid_size-2:self.grid_size+1] = 1/50

        self.high_to_output_mask = torch.zeros_like(self.high_to_output.weight,requires_grad = False)
        self.high_to_output_mask[:, :, self.grid_size-5:self.grid_size+4,self.grid_size-5:self.grid_size+4] = 1/50

        # Initializing weights
        self.init_weights_FF(self.high_to_output,receptive_field_size=9)
        self.init_weights_FF(self.input_to_output, one_to_one=True) 
        self.init_weights_FF(self.low_to_output,receptive_field_size=1)
        self.init_weights_FF(self.middle_to_output,receptive_field_size=3)


    def forward(self, inputmod,low_scale,middle_scale,high_scale):
        Y =  self.input_to_output(inputmod) + self.low_to_output(low_scale) 
        if not self.one_scale:
            Y = Y + self.middle_to_output(middle_scale) + self.high_to_output(high_scale)
        return(Y)

    def rescale(self,new_grid_size,device):
        high_to_output = nn.ConvTranspose2d(6, 1,2 * new_grid_size - 1, stride=self.bigger_pixels_size, padding=int(0.5*(2 * new_grid_size - 1 - self.bigger_pixels_size)),bias=False)#nn.ConvTranspose2d(6, feature_out, 3, stride=3, padding=0,bias=False)
        low_to_output = nn.Conv2d(self.hidden_features, self.feature_out, 2 * self.grid_size - 1, stride=1, padding='same',bias=False)
        middle_to_output = nn.ConvTranspose2d(6, 1,2 * new_grid_size - 1, stride=self.big_pixels_size, padding=int(0.5*(2 * new_grid_size - 1 - self.big_pixels_size)),bias=False)
        
        high_to_output_weight = torch.zeros_like(high_to_output.weight)
        lower_bound = int(0.5*(2*new_grid_size-1-self.bigger_pixels_size))
        upper_bound = int(lower_bound + self.bigger_pixels_size)
        test = torch.unique(self.high_to_output.weight[self.high_to_output.weight!=0])
        test = test[None,None,None,:]
        high_to_output_weight[:,:,lower_bound:upper_bound,lower_bound:upper_bound] = test.T
        high_to_output.weight = torch.nn.Parameter(high_to_output_weight)
        self.high_to_output = high_to_output
        self.high_to_output.to(device)
        
        middle_to_output_weight = torch.zeros_like(middle_to_output.weight)
        lower_bound = int(0.5*(2*new_grid_size-1-self.big_pixels_size))
        upper_bound = int(lower_bound + self.big_pixels_size)
        test = torch.unique(self.middle_to_output.weight[self.middle_to_output.weight!=0])
        test = test[None,None,None,:]
        middle_to_output_weight[:,:,lower_bound:upper_bound,lower_bound:upper_bound] = test.T
        middle_to_output.weight = torch.nn.Parameter(middle_to_output_weight)
        self.middle_to_output = middle_to_output
        self.middle_to_output.to(device)
        
        low_to_output_weight = torch.zeros_like(low_to_output.weight)
        lower_bound = int(0.5*(2*new_grid_size-1-1))
        upper_bound = int(lower_bound + 1)
        test = torch.unique(self.low_to_output.weight[self.low_to_output.weight!=0])
        test = test[None,None,None,:]
        low_to_output_weight[:,:,lower_bound:upper_bound,lower_bound:upper_bound] = test.T
        low_to_output.weight = torch.nn.Parameter(low_to_output_weight)
        self.low_to_output = low_to_output
        self.low_to_output.to(device)
        
          
                              
        
    def update_layer(self, upper, beta, delta):
        self.input_to_output = self.update_weight(self.input_to_output, upper, beta, delta, average=False)
        self.low_to_output = self.update_weight(self.low_to_output, upper, beta, delta,self.low_to_output_mask, receptive_field_size = 1)
        self.middle_to_output = self.update_weight(self.middle_to_output, upper, beta, delta,self.middle_to_output_mask, receptive_field_size=self.big_pixels_size)
        self.high_to_output = self.update_weight(self.high_to_output, upper, beta, delta, self.high_to_output_mask,receptive_field_size = self.bigger_pixels_size)

class FFLayer(CustomLayer):

    def __init__(self,low_scale_feedforward,middle_scale_feedforward_interm,middle_scale_feedforward,high_scale_feedforward_interm,high_scale_feedforward):
        super().__init__()
        self.low_scale_feedforward = low_scale_feedforward
        
        self.middle_scale_feedforward_interm = middle_scale_feedforward_interm
        self.middle_scale_feedforward = middle_scale_feedforward
        
        self.high_scale_feedforward_interm = high_scale_feedforward_interm
        self.high_scale_feedforward = high_scale_feedforward
        self.sig = nn.Sigmoid()

    def forward(self, x):
        low_scale = F.relu(self.low_scale_feedforward(x))
        
        # Middle scale
        middle_scale_interm = F.relu(self.middle_scale_feedforward_interm(low_scale))
        middle_scale = self.sig(self.middle_scale_feedforward(middle_scale_interm))
                
        # High scale
        high_scale_interm = F.relu(self.high_scale_feedforward_interm(low_scale))
        high_scale = self.sig(self.high_scale_feedforward(high_scale_interm))
        
        # Keeping only the neurons with high probability of activation
        middle_scale = torch.relu(middle_scale - 0.7)
        high_scale = torch.relu(high_scale - 0.7)
    
        return low_scale.detach(),middle_scale.detach(),high_scale.detach()
