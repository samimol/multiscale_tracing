# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:29:28 2021

@author: Sami
"""
from typing import Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLayer(nn.Module):
    """Base class for custom neural network layers with specialized initialization and update rules.
    
    This class provides common functionality for input, hidden, and output layers including
    weight initialization, activation functions, and biologically-inspired learning rules.
    """

    def __init__(self) -> None:
        """Initialize the custom layer with default parameters."""
        self.initialisation_range = 0.1
        super().__init__()

    def activation_function(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ReLU activation function.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Activated output.
        """
        return torch.relu(x)

    def step_function(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a piecewise linear step function with saturation.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Transformed tensor with linear region and saturation.
        """
        eps = 1
        phi = 1      
        x[x <= eps/phi] = phi * x[x <= eps/phi]
        x[x <= 0] = 0
        x[x >= eps/phi] = eps
        return x


    def gating_function(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a smooth gating function for modulation.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Gated output with smooth saturation.
        """
        param = 100
        return torch.relu(param*x/(torch.sqrt(1+(param**2)*(x**2))))

    def initialize_feedforward_weights(self, layer: nn.Module, one_to_one: bool = False, change_scale: bool = False, receptive_field_size: int = 1) -> None:
        """Initialize feedforward connection weights with specific patterns.
        
        Args:
            layer (nn.Module): The layer whose weights to initialize.
            one_to_one (bool): If True, use one-to-one connectivity pattern.
            change_scale (bool): If True, initialize for scale-changing connections.
            receptive_field_size (int): Size of the receptive field for spatial connections.
        """
        # Initialize weight tensor with zeros
        weights = torch.zeros(layer.weight.shape)
        if layer.bias is not None:
            bias = torch.zeros(layer.bias.shape)
        
        if change_scale:
            # For scale-changing connections, initialize all weights uniformly
            for f in range(weights.shape[0]):
                for f2 in range(weights.shape[1]):
                    weights[f, f2, :, :] = self.initialisation_range * np.random.rand() + 1
        else:
            # For same-scale connections
            for f in range(weights.shape[0]):
                for f2 in range(weights.shape[1]):
                    if not one_to_one:
                        # Localized receptive field: only connect nearby spatial locations
                        lower_bound = self.grid_size - receptive_field_size//2 - 1
                        upper_bound = self.grid_size + receptive_field_size//2
                        weights[f, f2, lower_bound:upper_bound,lower_bound:upper_bound] = 0.001 * np.random.rand()
                    else:
                        # One-to-one connections: single weight per feature pair
                        if self.layer_type != "output":
                            weights[f,f2,0] =  self.initialisation_range * np.random.rand() + 1
                        else:
                            # Output layer uses smaller initial weights
                            weights[f, f2, 0] = 0.001 + 0.001 * np.random.rand()
        
        # Initialize biases to 1 (positive baseline)
        if layer.bias is not None:
            for f in range(bias.shape[0]):  
                bias[f] = 1
        
        # Set as learnable parameters
        layer.weight = torch.nn.Parameter(weights)
        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(bias)

    def initialize_feedback_weights(self, layer: nn.Module, change_scale: bool = False) -> None:
        """Initialize feedback connection weights.
        
        Args:
            layer (nn.Module): The layer whose weights to initialize.
            change_scale (bool): If True, initialize for scale-changing connections.
        """
        # Initialize weight tensor with zeros
        weights = torch.zeros(layer.weight.shape)
        if layer.bias is not None:
            bias = torch.zeros(layer.bias.shape)
        
        # Set up feedback connections (top-down modulation)
        for f in range(weights.shape[0]):
            for f2 in range(weights.shape[1]):
                if self.layer_type != "output":
                    if change_scale:
                        # Scale-changing feedback: full spatial connectivity
                        weights[f, f2, :, :] = self.initialisation_range * np.random.rand() + 0.5
                    else:
                        # Local feedback: connect to 4 nearest neighbors (cross pattern)
                        # This creates a plus-shaped connectivity pattern
                        weights[f, f2, 0,1] = self.initialisation_range * np.random.rand() + 0.1  # Top
                        weights[f, f2, 1,0] = self.initialisation_range * np.random.rand() + 0.1  # Left
                        weights[f, f2, 1,2] = self.initialisation_range * np.random.rand() + 0.1  # Right
                        weights[f, f2, 2,1] = self.initialisation_range * np.random.rand() + 0.1  # Bottom
        
        # Set as learnable parameters
        layer.weight = torch.nn.Parameter(weights)
        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(bias)


    def initialize_inhibitory_weights(self, layer: nn.Module) -> None:
        """Initialize inhibitory connection weights with negative biases.
        
        Args:
            layer (nn.Module): The layer whose weights to initialize.
        """
        # Initialize weight tensor with zeros
        weights = torch.zeros(layer.weight.shape)
        if layer.bias is not None:
            bias = torch.zeros(layer.bias.shape)
        
        # Set up lateral inhibition: only within same feature channel (f == f2)
        for f in range(weights.shape[0]):
            for f2 in range(weights.shape[1]):
                if f == f2:
                    # Self-connection at center of 3x3 kernel
                    weights[f, f2, 1, 1] = 1
        
        # Initialize biases to negative values for inhibitory effect
        if layer.bias is not None:
            for f in range(bias.shape[0]):  
                bias[f] = -1 - self.initialisation_range * np.random.rand()
        
        # Set as learnable parameters
        layer.weight = torch.nn.Parameter(weights)
        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(bias)


    def average_traces(self, traces: torch.Tensor, mask: torch.Tensor, receptive_field_size: int = 1) -> torch.Tensor:
        """Average gradient traces over spatial dimensions for weight updates.
        
        Args:
            traces (torch.Tensor): Gradient traces to average.
            mask (torch.Tensor): Mask indicating valid connections.
            receptive_field_size (int): Size of receptive field for output layers.
            
        Returns:
            torch.Tensor: Averaged traces.
        """
        traces *= mask
        if self.layer_type != "output":
            m = torch.mean(traces[:,:,:,:], axis=(2,3))
            m = m[:, :, None,None]
            traces[:,:] = m  
        else:
            intermmask = torch.ones((2 * self.grid_size - 1, 2 * self.grid_size - 1))
            lower_bound = self.grid_size - receptive_field_size//2 - 1
            upper_bound = self.grid_size + receptive_field_size//2
            
            intermmask = torch.zeros((2 * self.grid_size - 1, 2 * self.grid_size - 1))
            intermmask[lower_bound:upper_bound,lower_bound:upper_bound] = 1
            m = torch.mean(traces[:, :, intermmask == 1], axis=2)
            traces[:, :, intermmask == 1] =  m[:,None]    
        return(traces)


    def update_weight(self, layer: nn.Module, upper: torch.Tensor, beta: float, delta: float, mask: Optional[torch.Tensor] = None, z: Optional[torch.Tensor] = None, average: bool = True, receptive_field_size: int = 1, inhib: bool = False, change_scale: bool = False) -> nn.Module:
        """Update layer weights using reward prediction error.
        
        Args:
            layer (nn.Module): The layer to update.
            upper (torch.Tensor): Upper layer activity.
            beta (float): Learning rate.
            delta (float): Reward prediction error.
            mask (torch.Tensor, optional): Connection mask.
            z (torch.Tensor, optional): Gradient output.
            average (bool): Whether to average traces.
            receptive_field_size (int): Receptive field size.
            inhib (bool): If True, update inhibitory connections only.
            change_scale (bool): If True, handle scale-changing connections.
            
        Returns:
            nn.Module: Updated layer.
        """
        with torch.no_grad():
            # Compute gradients of output with respect to weights and biases
            # This implements eligibility traces for credit assignment
            if layer.bias is not None:
                delta_weight = torch.autograd.grad(upper, [layer.weight, layer.bias], grad_outputs=z, retain_graph=True, allow_unused=True)
                delta_bias = delta_weight[1]
                # Clamp gradient magnitudes to prevent instability
                delta_bias = torch.clamp(delta_bias, None, 1)
            else:
                delta_weight = torch.autograd.grad(upper, layer.weight, grad_outputs=z, retain_graph=True, allow_unused=True)
            
            delta_weight = delta_weight[0]
            # Clamp weight gradients to prevent large updates
            delta_weight = torch.clamp(delta_weight, None, 1)
            
            if inhib == False:
                # Update excitatory weights using the reward prediction error
                # delta_weight acts as eligibility trace, delta is reward prediction error
                if (self.layer_type == 'output' and average) or change_scale:
                    # For output and scale-changing layers, average gradients spatially
                    delta_weight = self.average_traces(delta_weight, mask, receptive_field_size=receptive_field_size)
                elif mask is not None:
                    # Apply connection mask to enforce sparse connectivity
                    delta_weight = delta_weight * mask
                
                # Weight update: w = w + learning_rate * RPE * eligibility_trace
                weight_update = layer.weight + beta * delta * delta_weight
                layer.weight.copy_(weight_update)
            else:
                # Update inhibitory biases only (weights remain fixed)
                bias_update = layer.bias + beta * delta * delta_bias
                layer.bias.copy_(bias_update)
        return(layer)
    
    def make_mask(self, weight: torch.Tensor) -> torch.Tensor:
        """Create a binary mask from weight tensor.
        
        Args:
            weight (torch.Tensor): Weight tensor.
            
        Returns:
            torch.Tensor: Binary mask (1 where weights are non-zero, 0 elsewhere).
        """
        mask = torch.clone(weight.detach())
        mask[mask != 0] = 1
        return mask
        
    def to(self, device: torch.device) -> None:
        """Move layer parameters to specified device.

        Args:
            device (torch.device): Target device (CPU or CUDA).
        """
        if self.layer_type == 'input':
          self.FB.to(device)
          self.lateral_inhibition.to(device)
          self.feedback_mask = self.feedback_mask.to(device)
          self.inhibition_mask = self.inhibition_mask.to(device)
        elif self.layer_type == 'hidden':
          self.FF.to(device)
          self.feedforward_mask = self.feedforward_mask.to(device)
          self.horizontal_mask = self.horizontal_mask.to(device)
          self.H.to(device)
          if self.has_feedback:
              self.FB.to(device)
              self.feedback_mask = self.feedback_mask.to(device)
        elif self.layer_type == 'output':
          for layer in range(len(self.skip_weights)):
              self.skip_weights[layer].to(device)
          for layer in range(1,len(self.skip_masks)):
              self.skip_masks[layer] = self.skip_masks[layer].to(device)

                
                
class InputLayer(CustomLayer):
    """Input layer with feedback modulation and lateral inhibition.
    
    This layer receives sensory input and is modulated by feedback from higher layers
    through VIP and SOM interneuron populations.
    """

    def __init__(self, feature_in: int, feature_out: int) -> None:
        """Initialize input layer.
        
        Args:
            feature_in (int): Number of input features.
            feature_out (int): Number of output features from higher layer.
        """
        super().__init__()
        self.layer_type = 'input'
        K_size = 3

        self.FB = nn.Conv2d(feature_out, feature_in, K_size, stride=1, padding='same',bias=False)
        self.lateral_inhibition = nn.Conv2d(feature_in, feature_in, K_size, stride=1, padding='same',bias=True)

        # Initializing weights
        self.initialize_feedback_weights(self.FB)
        self.initialize_inhibitory_weights(self.lateral_inhibition)

        # Setting the mask to have connections only between neighboours
        self.feedback_mask = self.make_mask(self.FB.weight)
        self.inhibition_mask = self.make_mask(self.lateral_inhibition.weight)

    def forward(self, upper_y: torch.Tensor, input_stimulus: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through input layer.
        
        Args:
            upper_y (torch.Tensor): Modulation signal from upper layer.
            input_stimulus (torch.Tensor): Sensory input stimulus.
            
        Returns:
            tuple: (modulated_output, vip_activity, som_activity)
        """
        Y = input_stimulus
        vip_activity = self.step_function(self.FB(upper_y))
        som_activity = self.activation_function(1 - vip_activity)
        modulated_output =  self.activation_function((-self.lateral_inhibition(som_activity)) * self.gating_function(Y))
        return(modulated_output, vip_activity, som_activity)


    def update_layer(self, upper: List[torch.Tensor], z: List[torch.Tensor], beta: float, delta: float) -> None:
        """Update layer weights based on reward prediction error.
        
        Args:
            upper (list): Upper layer activities [pyramidal, som, vip].
            z (list): Gradients [pyramidal, som, vip].
            beta (float): Learning rate.
            delta (float): Reward prediction error.
        """
        self.FB = self.update_weight(self.FB, upper[2], beta, delta, self.feedback_mask, z[2],change_scale=False)
        self.lateral_inhibition = self.update_weight(self.lateral_inhibition, upper[0], beta, delta, self.inhibition_mask, z[0],inhib=True)

class HiddenLayer(CustomLayer):
    """Hidden layer with feedforward, feedback, and horizontal connections.
    
    This layer integrates information from lower layers (feedforward), higher layers
    (feedback), and within the same layer (horizontal connections).
    """

    def __init__(self, feature_in_lower: int, feature_out: int, feature_in_higher: int, big_pixels_size: int, grid_size: int, has_feedback: bool = True, change_scale_fb: bool = False, change_scale_ff: bool = False, higher_scale: bool = False) -> None:
        """Initialize hidden layer.
        
        Args:
            feature_in_lower (int): Number of features from lower layer.
            feature_out (int): Number of output features.
            feature_in_higher (int): Number of features from higher layer.
            big_pixels_size (int): Stride for scale-changing connections.
            grid_size (int): Size of spatial grid.
            has_feedback (bool): Whether to include upper layer modulation.
            change_scale_fb (bool): Whether feedback changes scale.
            change_scale_ff (bool): Whether feedforward changes scale.
            higher_scale (bool): Whether this is a higher-scale layer.
        """
        super().__init__()
        self.layer_type = 'hidden'
        self.has_feedback = has_feedback # If the higher layer has a modulated group
        self.change_scale_fb = change_scale_fb
        self.change_scale_ff = change_scale_ff
        self.higher_scale = higher_scale
        self.big_pixels_size = big_pixels_size
        self.grid_size = grid_size
        
        # Making the weights
        if self.change_scale_ff:
          self.FF = nn.Conv2d(feature_in_lower, feature_out, self.big_pixels_size, stride=self.big_pixels_size, bias=True)
        else:
            self.FF = nn.Conv2d(feature_in_lower, feature_out, 1, stride=1, padding='same', bias=True)     
        
        if has_feedback:
            self.FB = nn.ConvTranspose2d(feature_in_higher, feature_out, self.big_pixels_size, stride=self.big_pixels_size, padding=0,bias=False)
        self.H = nn.Conv2d(feature_out,feature_out,3,stride = 1,padding='same',bias=False)
        
        # Initializing the weights
        self.initialize_feedforward_weights(self.FF,change_scale = change_scale_ff,one_to_one=True)
        if has_feedback:
            self.initialize_feedback_weights(self.FB,change_scale = change_scale_fb)
        self.initialize_feedback_weights(self.H,change_scale=False)

        # Making the masks        
        if has_feedback:
            self.feedback_mask = self.make_mask(self.FB.weight)
        self.feedforward_mask = self.make_mask(self.FF.weight)
        self.horizontal_mask = self.make_mask(self.H.weight)
   

    def forward(self, current_y: torch.Tensor, lower_ymod: Optional[torch.Tensor] = None, upper_y: Optional[torch.Tensor] = None, horiz: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through hidden layer.
        
        Args:
            current_y (torch.Tensor): Current layer activity.
            lower_ymod (torch.Tensor, optional): Input from lower layer.
            upper_y (torch.Tensor, optional): Modulation from upper layer.
            horiz (torch.Tensor, optional): Horizontal connections.
            
        Returns:
            tuple: (modulated_output, vip_activity, som_activity)
        """
        feedforward_input = self.FF(lower_ymod)
        if upper_y is not None:
            vip_activity = self.step_function(self.FB(upper_y) + self.H(horiz))
        else:
            vip_activity = self.step_function(self.H(horiz))
        som_activity = self.activation_function(1 - vip_activity)
        modulated_output =  self.activation_function((feedforward_input - som_activity) * self.gating_function(current_y))
        return(modulated_output, vip_activity, som_activity)
        
    def update_layer(self, upper: List[torch.Tensor], z: List[torch.Tensor], beta: float, delta: float, train_v: bool = True) -> None:
        """Update layer weights based on reward prediction error.
        
        Args:
            upper (list): Upper layer activities [pyramidal, som, vip].
            z (list): Gradients [pyramidal, som, vip].
            beta (float): Learning rate.
            delta (float): Reward prediction error.
            train_v (bool): Whether to train VIP connections.
        """
        self.FF = self.update_weight(self.FF, upper[0], beta, delta, self.feedforward_mask, z[0],change_scale=self.change_scale_ff)
        if self.has_feedback:
            self.FB = self.update_weight(self.FB, upper[2], beta, delta, self.feedback_mask, z[2],change_scale = self.change_scale_fb)
        self.H = self.update_weight(self.H, upper[2], beta, delta, self.horizontal_mask, z[2],change_scale = False)

class OutputLayer(CustomLayer):
    """Output layer that aggregates information from all hierarchical levels.
    
    This layer combines skip connections from input, hidden, and high-level layers
    to produce action values (Q-values).
    """

    def __init__(self, high_features: int, hidden_features: int, input_features: int, feature_out: int, grid_size: int, RF_size_list: List[int]) -> None:
        """Initialize output layer.
        
        Args:
            high_features (int): Number of high-level features.
            hidden_features (int): Number of hidden features.
            input_features (int): Number of input features.
            feature_out (int): Number of output features.
            grid_size (int): Size of spatial grid.
            RF_size_list (list): List of receptive field sizes for each scale.
        """
        super().__init__()
        self.grid_size = grid_size
        K_size = 2 * self.grid_size - 1
        self.layer_type = 'output'
        self.grid_size = grid_size
        self.RF_size_list = RF_size_list
        self.hidden_features = hidden_features
        self.high_features = high_features
        self.feature_out = feature_out
        
        self.skip_weights = []
        for layer in range(len(self.RF_size_list)):
            if layer == 0:
                self.skip_weights.append(nn.Conv2d(input_features, feature_out, 1, stride=self.RF_size_list[layer], padding='same', bias=False))
            elif layer == 1:
                self.skip_weights.append(nn.Conv2d(hidden_features, feature_out, K_size, stride=self.RF_size_list[layer], padding='same',bias=False))
            else:
                self.skip_weights.append(nn.ConvTranspose2d(high_features, feature_out,K_size, stride=self.RF_size_list[layer], padding=int(0.5*(2 * self.grid_size - 1 - self.RF_size_list[layer])),bias=False))
        
        self.skip_masks = [None]
        for layer in range(1,len(self.RF_size_list)):
            self.skip_masks.append(torch.zeros_like(self.skip_weights[layer].weight))
            lower_bound = self.grid_size-(self.RF_size_list[layer]//2)-1
            upper_bound = self.grid_size+(self.RF_size_list[layer]//2)
            self.skip_masks[layer][:,:,lower_bound:upper_bound,lower_bound:upper_bound] = 1/50

        # Initializing weights
        self.initialize_feedforward_weights(self.skip_weights[0], one_to_one=True) 
        for layer in range(1,len(self.RF_size_list)):
            self.initialize_feedforward_weights(self.skip_weights[layer], receptive_field_size=self.RF_size_list[layer]) 

        self.skip_weights = nn.ModuleList(self.skip_weights)

    def forward(self, pyramidal_recurrent: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass through output layer.
        
        Args:
            pyramidal_recurrent (list): List of pyramidal activities from all layers.
            
        Returns:
            torch.Tensor: Output Q-values for action selection.
        """
        Y =  self.skip_weights[0](pyramidal_recurrent[0])
        for layer in range(1,len(pyramidal_recurrent)):
            Y = Y + self.skip_weights[layer](pyramidal_recurrent[layer])
        return(Y)

    def rescale(self, new_grid_size: int, device: torch.device) -> None:
        """Rescale output layer for different grid size.
        
        Args:
            new_grid_size (int): New spatial grid size.
            device (torch.device): Device to move parameters to.
        """
        K_size = 2 * new_grid_size - 1
        skip_weights = [self.skip_weights[0]]
        for layer in range(1,len(self.RF_size_list)):
            if layer == 1:
                skip_weights.append(nn.Conv2d(self.hidden_features, self.feature_out, K_size, stride=self.RF_size_list[layer], padding='same',bias=False))
            else:
                skip_weights.append(nn.ConvTranspose2d(self.high_features, self.feature_out,K_size, stride=self.RF_size_list[layer], padding=int(0.5*(2 * new_grid_size - 1 - self.RF_size_list[layer])),bias=False))
        
        for layer in range(1,len(self.RF_size_list)):
            weight = torch.zeros_like(skip_weights[layer].weight)
            lower_bound = new_grid_size - self.RF_size_list[layer]//2 - 1
            upper_bound = new_grid_size + self.RF_size_list[layer]//2
            non_zero_weights = torch.unique(self.skip_weights[layer].weight[self.skip_weights[layer].weight!=0])
            non_zero_weights = non_zero_weights[:,None,None,None]
            weight[:,:,lower_bound:upper_bound,lower_bound:upper_bound] = non_zero_weights
            skip_weights[layer].weight = torch.nn.Parameter(weight)
            self.skip_weights[layer] = skip_weights[layer]
            self.skip_weights[layer].to(device)
 
        self.skip_masks = [None]
        for layer in range(1,len(self.RF_size_list)):
            self.skip_masks.append(torch.zeros_like(self.skip_weights[layer].weight))
            lower_bound = new_grid_size-(self.RF_size_list[layer]//2)-1
            upper_bound = new_grid_size+(self.RF_size_list[layer]//2)
            self.skip_masks[layer][:,:,lower_bound:upper_bound,lower_bound:upper_bound] = 1/50
            
        self.grid_size = new_grid_size

        
    def update_layer(self, upper: torch.Tensor, beta: float, delta: float) -> None:
        """Update all skip connection weights.
        
        Args:
            upper (torch.Tensor): Upper layer activity.
            beta (float): Learning rate.
            delta (float): Reward prediction error.
        """
        for layer in range(len(self.skip_weights)):
            if layer == 0:
                self.skip_weights[layer] = self.update_weight(self.skip_weights[layer],upper,beta,delta,average=False)
            else:
                self.skip_weights[layer] = self.update_weight(self.skip_weights[layer],upper,beta,delta,self.skip_masks[layer],receptive_field_size = self.RF_size_list[layer])

class FeedforwardLayer(CustomLayer):
    """Wrapper for pretrained feedforward networks.
    
    This layer encapsulates pretrained feedforward networks for object and curve detection
    at multiple scales.
    """

    def __init__(self, feedforward: nn.ModuleList, feedforward_interm: nn.ModuleList, num_scales: int) -> None:
        """Initialize feedforward layer.
        
        Args:
            feedforward (nn.ModuleList): List of feedforward layers.
            feedforward_interm (nn.ModuleList): List of intermediate layers.
            num_scales (int): Number of spatial scales.
        """
        super().__init__()
        self.num_scales = num_scales
        self.feedforward = feedforward
        self.feedforward_interm = feedforward_interm
        
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through feedforward network.
        
        Args:
            x (torch.Tensor): Input image.
            
        Returns:
            list: Multi-scale feature representations.
        """
        intern_representation = [None] * self.num_scales
        for layer in range(self.num_scales):
            if layer == 0:
                x = F.relu(self.feedforward[0](x))
                intern_representation[layer] = x
            else:
                interm = F.relu(self.feedforward_interm[layer](x))
                intern_representation[layer] = self.sig(self.feedforward[layer](interm))
                
        intern_representation[0] = intern_representation[0].detach()
        for layer in range(1,len(intern_representation)):
            intern_representation[layer] = torch.relu(intern_representation[layer] - 0.7)
            intern_representation[layer] = intern_representation[layer].detach()
        
        return intern_representation
