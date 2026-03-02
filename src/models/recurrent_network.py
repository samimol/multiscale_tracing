# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:28:45 2023

@author: Sami
"""

from typing import List, Tuple, Optional
from src.models.layers import InputLayer, HiddenLayer, OutputLayer, FeedforwardLayer
import torch
import numpy as np

from src.models.layers import CustomLayer


class RecurrentNetwork():
    """Hierarchical recurrent network for visual attention and curve tracing.
    
    This network implements a multi-scale architecture with feedback connections,
    VIP/SOM interneuron modulation, and reinforcement learning for sequential
    decision-making in visual tasks.
    """

    def __init__(self, n_input_features: int, grid_size: int, device: torch.device, feedforward_curve: FeedforwardLayer, feedforward_object: FeedforwardLayer, one_scale: bool, num_scales: int) -> None:
        """Initialize recurrent network.
        
        Args:
            n_input_features (int): Number of input features (e.g., RGB channels).
            grid_size (int): Size of spatial grid.
            device (torch.device): Device to place network on.
            feedforward_curve (FeedforwardNetwork): Pretrained network for curve detection.
            feedforward_object (FeedforwardNetwork): Pretrained network for object detection.
            one_scale (bool): Whether to use single-scale processing.
            num_scales (int): Number of hierarchical scales.
        """
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

        self.layers_list: List[HiddenLayer] = [InputLayer(self.n_input_features, self.n_hidden_features)]
        self.RF_size_list = [1] + [3**i for i in range(self.num_scales)]
        
        RF_size = 3
        for i in range(num_scales):
            if i == 0:
                self.layers_list.append(HiddenLayer(self.n_input_features, self.n_hidden_features, self.high_feature,RF_size,grid_size,change_scale_fb=True))
            elif i == 1 and num_scales > 2:
                self.layers_list.append(HiddenLayer(self.n_hidden_features, self.high_feature,self.high_feature,RF_size,grid_size,change_scale_ff=True,higher_scale=True,change_scale_fb=True,has_feedback=True))
            elif i < num_scales - 1:
                self.layers_list.append(HiddenLayer(self.high_feature, self.high_feature,self.high_feature,RF_size,grid_size,change_scale_ff=True,higher_scale=True,change_scale_fb=True,has_feedback=True))
            elif i == num_scales - 1:
                self.layers_list.append(HiddenLayer(self.high_feature,self.high_feature,self.high_feature,RF_size,grid_size,has_feedback = False,change_scale_ff=True))
        self.layers_list.append(OutputLayer(self.high_feature,self.n_hidden_features, self.n_input_features, 1,self.grid_size,self.RF_size_list))
        self.feedforward_network_curve = FeedforwardLayer(feedforward_curve.feedforward,feedforward_curve.feedforward_interm,num_scales)
        self.feedforward_network_object = FeedforwardLayer(feedforward_object.feedforward,feedforward_object.feedforward_interm,num_scales)

        self.task = 'trace_curve'

        self.save_activities = False
        
        self.saved_activities = [[] for i in range(num_scales+2)]
        
        self.to_device()

        
    def step(self, input_env: torch.Tensor, reward: float, reset_traces: bool, device: torch.device) -> torch.Tensor:
        """Perform one step of network dynamics and action selection.
        
        This method runs the recurrent dynamics until convergence, then selects
        an action based on the output Q-values.
        
        Args:
            input_env (torch.Tensor): Environmental input (visual stimulus).
            reward (float): Reward from previous action.
            reset_traces (bool): Whether to reset eligibility traces.
            device (torch.device): Device for computation.
            
        Returns:
            torch.Tensor: One-hot encoded action (flattened spatial location).
        """

        # Initialize activity storage if needed for visualization/analysis
        if self.save_activities:
            for layer in range(len(self.saved_activities)):
                self.saved_activities[layer].append([])

        # Initialize pyramidal neuron activities for all layers
        # Layer 0 is the input layer with full resolution
        self.pyramidal_recurrent = [torch.zeros(1, self.n_input_features, self.grid_size, self.grid_size, device=self.device)]
        self.VIP = [torch.zeros(1, self.n_input_features, self.grid_size, self.grid_size, device=self.device)]
        self.SOM = [torch.zeros(1, self.n_input_features, self.grid_size, self.grid_size, device=self.device)]
        
        # Initialize hidden layers at multiple scales
        for i in range(self.num_scales):
            if i == 0:
                # First hidden layer: same resolution as input
                self.pyramidal_recurrent.append(torch.zeros(1, self.n_hidden_features, self.grid_size, self.grid_size, device=self.device))
                self.VIP.append(torch.zeros(1, self.n_hidden_features, self.grid_size, self.grid_size, device=self.device))
                self.SOM.append(torch.zeros(1, self.n_hidden_features, self.grid_size, self.grid_size, device=self.device))
            else:
                # Higher layers: progressively coarser spatial resolution
                self.pyramidal_recurrent.append(torch.zeros(1,self.high_feature, self.grid_size//self.RF_size_list[i+1], self.grid_size//self.RF_size_list[i+1], device=self.device))
                self.VIP.append(torch.zeros(1,self.high_feature, self.grid_size//self.RF_size_list[i+1], self.grid_size//self.RF_size_list[i+1], device=self.device))
                self.SOM.append(torch.zeros(1,self.high_feature, self.grid_size//self.RF_size_list[i+1], self.grid_size//self.RF_size_list[i+1], device=self.device))
        
        # Initialize convergence tracking
        i = 0  # Iteration counter
        norm = 10  # Change in activity (starts high)
        flag = True  # Force at least 6 iterations

        # Get feedforward features from pretrained network based on task
        if self.task == 'trace_curve':
            self.pyramidal_feedforward = self.feedforward_network_curve(input_env)
        elif self.task == 'trace_object':
            self.pyramidal_feedforward = self.feedforward_network_object(input_env)
        else:
            raise Exception("Task property can be either trace_curve or trace_object")
        # Add None at index 0 to align with layer indexing
        self.pyramidal_feedforward = [None] + self.pyramidal_feedforward
        
        # Run recurrent dynamics until convergence or max iterations
        # Continues while: (not converged AND under time limit) OR still in minimum iteration period
        while (i < self.duration and norm > 0) or flag:
            # Store previous state to check convergence
            prev_low = self.pyramidal_recurrent[1].detach()

            # Update all layers in hierarchy
            # Input layer: receives feedback from layer 1 and sensory input
            self.pyramidal_recurrent[0],self.VIP[0],self.SOM[0] = self.layers_list[0].forward(self.pyramidal_recurrent[1],input_env)
            
            # Hidden layers: integrate feedforward, feedback, and horizontal connections
            for layer in range(1,len(self.pyramidal_recurrent)-1):
                (self.pyramidal_recurrent[layer],self.VIP[layer],self.SOM[layer]) = self.layers_list[layer].forward(current_y=self.pyramidal_feedforward[layer],lower_ymod = self.pyramidal_recurrent[layer-1],upper_y = self.pyramidal_recurrent[layer + 1],horiz=self.pyramidal_recurrent[layer])
            
            # Top layer: no feedback from above
            (self.pyramidal_recurrent[-1],self.VIP[-1],self.SOM[-1]) = self.layers_list[layer+1].forward(self.pyramidal_feedforward[layer+1],lower_ymod = self.pyramidal_recurrent[-2],upper_y = None,horiz=self.pyramidal_recurrent[-1])

            # Save activities for analysis if requested
            if self.save_activities:
                for layer in range(len(self.saved_activities)-1):
                    self.saved_activities[layer][len(self.saved_activities[layer]) - 1].append(self.pyramidal_recurrent[layer].detach())
          
                self.output_values = self.calculate_output(device)
                self.saved_activities[-1][len(self.saved_activities[-1]) - 1].append(self.output_values.detach())

            # Check convergence: measure change in lowest hidden layer activity
            with torch.no_grad():
                norm = torch.linalg.norm(self.pyramidal_recurrent[1][0, :, :]-prev_low[0, :, :])

            i += 1
            # After 6 iterations, allow early stopping if converged
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
            (self.prev_prev_pyramidal[layer],self.prev_prev_VIP[layer],self.prev_prev_SOM[layer]) = self.layers_list[layer].forward(self.pyramidal_feedforward[layer],lower_ymod = self.pyramidal_recurrent[layer-1],upper_y = self.pyramidal_recurrent[layer + 1],horiz=self.pyramidal_recurrent[layer])
        (self.prev_prev_pyramidal[-1],self.prev_prev_VIP[-1],self.prev_prev_SOM[-1]) = self.layers_list[layer+1].forward(self.pyramidal_feedforward[layer+1],lower_ymod = self.pyramidal_recurrent[-2],upper_y = None,horiz=self.pyramidal_recurrent[-1])
                      

        (self.prev_pyramidal[0],self.prev_VIP[0],self.prev_SOM[0]) = self.layers_list[0].forward(self.prev_prev_pyramidal[1],input_env)
        for layer in range(1,len(self.pyramidal_recurrent)-1):
            (self.prev_pyramidal[layer],self.prev_VIP[layer],self.prev_SOM[layer]) = self.layers_list[layer].forward(current_y=self.pyramidal_feedforward[layer],lower_ymod = self.prev_pyramidal[layer-1],upper_y = self.prev_prev_pyramidal[layer + 1],horiz=self.prev_prev_pyramidal[layer])
        (self.prev_pyramidal[-1],self.prev_VIP[-1],self.prev_SOM[-1]) = self.layers_list[layer+1].forward(self.pyramidal_feedforward[layer+1],lower_ymod = self.prev_pyramidal[-2],upper_y = None,horiz=self.prev_prev_pyramidal[-1])
              
        self.pyramidal_recurrent_detached = self.detach_and_reattach(self.prev_pyramidal)
        self.VIP_detached = self.detach_and_reattach(self.prev_VIP)
        self.SOM_detached = self.detach_and_reattach(self.prev_SOM)


        (self.pyramidal_recurrent[0],self.VIP[0],self.SOM[0]) = self.layers_list[0].forward(self.pyramidal_recurrent_detached[1],input_env)
        for layer in range(1,len(self.pyramidal_recurrent)-1):
            (self.pyramidal_recurrent[layer],self.VIP[layer],self.SOM[layer]) = self.layers_list[layer].forward(self.pyramidal_feedforward[layer],lower_ymod = self.pyramidal_recurrent[layer-1],upper_y = self.pyramidal_recurrent_detached[layer + 1],horiz=self.pyramidal_recurrent_detached[layer])
        (self.pyramidal_recurrent[-1],self.VIP[-1],self.SOM[-1]) = self.layers_list[layer+1].forward(self.pyramidal_feedforward[layer+1],lower_ymod = self.pyramidal_recurrent[-2],upper_y = None,horiz=self.pyramidal_recurrent_detached[-1])
                                                                                    
        
        self.output_values = self.calculate_output(device)

        if self.save_activities:
            self.saved_activities[-1][len(self.saved_activities[-1]) - 1].append(self.output_values.detach())

        with torch.no_grad():
            action_chosen = torch.zeros((1, 1, 2+self.grid_size**2))
            action_chosen[0, 0, self.index_selected] = 1

        return (action_chosen)

    def calculate_output(self, device: torch.device) -> torch.Tensor:
        """Calculate output Q-values and select action.
        
        Args:
            device (torch.device): Device for computation.
            
        Returns:
            torch.Tensor: Q-values for all possible actions.
        """
        output = self.layers_list[-1].forward(self.pyramidal_recurrent)
        with torch.no_grad():
            flattened_output = torch.flatten(output.permute(0,1,3,2), start_dim=2)
            if np.random.rand() < self.exploitation_probability:
                winner = self.calculate_max_q_value(flattened_output)
            else:
                probabilities = flattened_output.detach()
                probabilities -= torch.max(probabilities)
                probabilities = torch.exp(probabilities) / torch.sum(torch.exp(probabilities))
                winner = self.calculate_soft_winner_take_all(probabilities, device)
            self.index_selected = winner
            winner = [torch.tensor([0]), torch.tensor([0]), torch.tensor([(winner)%self.grid_size]), torch.tensor([torch.div(winner,self.grid_size,rounding_mode='floor')])]
            self.action = winner

        return output

    def calculate_max_q_value(self, q_values: torch.Tensor) -> int:
        """Select action with maximum Q-value (exploitation).
        
        Args:
            q_values (torch.Tensor): Q-values for all actions.
            
        Returns:
            int: Index of selected action.
        """
        winner = torch.where(q_values == torch.max(q_values))[-1]
        if len(winner) > 1:
            tiebreak = winner[torch.randint(0, len(winner), (1,))]
            winner = tiebreak
        return winner

    def calculate_soft_winner_take_all(self, probabilities: torch.Tensor, device: torch.device) -> int:
        """Select action using softmax sampling (exploration).
        
        Args:
            probabilities (torch.Tensor): Probability distribution over actions.
            device (torch.device): Device for computation.
            
        Returns:
            int: Index of sampled action.
        """
        cumulative_probs = torch.cumsum(probabilities, 2)[0][0]
        random_value = torch.rand((1,), device=self.device)
        for (i, prob) in enumerate(cumulative_probs):
            if random_value <= prob:
                return i

    def compute_gradients(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Compute gradients through recurrent dynamics using backpropagation.
        
        This implements truncated backpropagation through time with multiple
        iterations to propagate gradients through the recurrent connections.
        
        Returns:
            tuple: (gradient_pyramidal, gradient_vip, gradient_som) for all layers.
        """

        # Compute initial gradients: dOutput/dActivity for all neuron types
        # These are the direct gradients from the output Q-value to each layer
        initial_gradient_pyramidal = torch.autograd.grad(self.output_values[self.action], self.pyramidal_recurrent, retain_graph=True, allow_unused=True)
        initial_gradient_vip = torch.autograd.grad(self.output_values[self.action], self.VIP, retain_graph=True, allow_unused=True)
        initial_gradient_som = torch.autograd.grad(self.output_values[self.action], self.SOM, retain_graph=True, allow_unused=True)
        
        # Convert to lists for iterative updates
        gradient_pyramidal = [initial_gradient_pyramidal[i] for i in range(len(initial_gradient_pyramidal))]
        gradient_vip = [initial_gradient_vip[i] for i in range(len(initial_gradient_vip))]
        gradient_som = [initial_gradient_som[i] for i in range(len(initial_gradient_som))]

        # Iteratively propagate gradients through recurrent connections
        # Multiple iterations approximate backpropagation through time
        for i in range(7):
            # Process layers from top to bottom (reverse order)
            for layer in range(len(gradient_pyramidal)-1,-1,-1):
                # VIP gradient: propagate through SOM -> VIP connection
                # VIP neurons gate information flow based on SOM activity
                gradient_vip[layer] = torch.autograd.grad(self.SOM[layer], self.VIP[layer], grad_outputs=gradient_som[layer], retain_graph=True, allow_unused=True)[0]           
                gradient_vip[layer] = gradient_vip[layer] + initial_gradient_vip[layer]
                
                # SOM gradient: propagate through pyramidal -> SOM connection
                # SOM neurons provide inhibition based on pyramidal activity
                gradient_som[layer] =  torch.autograd.grad(self.pyramidal_recurrent[layer], self.SOM[layer], grad_outputs=gradient_pyramidal[layer], retain_graph=True, allow_unused=True)[0]           
                gradient_som[layer] = gradient_som[layer] + initial_gradient_som[layer] 
                
                # Pyramidal gradient: accumulate from multiple sources
                if layer != 0:
                    # Feedback gradients from VIP neurons at this and lower layer
                    gradient_pyramidal[layer] = torch.autograd.grad(self.prev_VIP[layer], self.prev_prev_pyramidal[layer], grad_outputs=gradient_vip[layer], retain_graph=True, allow_unused=True)[0]
                    gradient_pyramidal[layer] = gradient_pyramidal[layer] + torch.autograd.grad(self.prev_VIP[layer-1], self.prev_prev_pyramidal[layer], grad_outputs=gradient_vip[layer-1], retain_graph=True, allow_unused=True)[0]
                
                if layer != len(gradient_pyramidal) - 1:
                    # Feedforward gradients from layer above
                    if layer == 0:
                        gradient_pyramidal[layer] = torch.autograd.grad(self.pyramidal_recurrent[layer+1], self.pyramidal_recurrent[layer], grad_outputs=gradient_pyramidal[layer+1], retain_graph=True, allow_unused=True)[0]
                    else:
                        gradient_pyramidal[layer] = gradient_pyramidal[layer] + torch.autograd.grad(self.pyramidal_recurrent[layer+1], self.pyramidal_recurrent[layer], grad_outputs=gradient_pyramidal[layer+1], retain_graph=True, allow_unused=True)[0]
                
                # Add direct gradient contribution
                gradient_pyramidal[layer] = gradient_pyramidal[layer] + initial_gradient_pyramidal[layer]

        return (gradient_pyramidal, gradient_vip, gradient_som)

    def learn(self, reward: float) -> None:
        """Update network weights using reward prediction error.
        
        This implements a biologically-inspired learning rule based on the
        difference between received and expected reward.
        
        Args:
            reward (float): Reward received for the action.
        """
        with torch.no_grad():
            expected_value = self.output_values[self.action]
            self.delta = reward - expected_value

        (gradient_pyramidal, gradient_vip, gradient_som) = self.compute_gradients()

        self.layers_list[0].update_layer([self.pyramidal_recurrent[0],self.SOM[0],self.VIP[0]],[gradient_pyramidal[0],gradient_som[0],gradient_vip[0]],self.beta,self.delta)
        for layer in range(1,len(self.layers_list)-1):
            self.layers_list[layer].update_layer([self.pyramidal_recurrent[layer],self.SOM[layer],self.VIP[layer]],[gradient_pyramidal[layer],gradient_som[layer],gradient_vip[layer]],self.beta,self.delta,train_v=False)
        self.layers_list[-1].update_layer(self.output_values[self.action], self.beta, self.delta)

    def detach_and_reattach(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Detach tensors from computation graph and reattach for new gradients.
        
        Args:
            x (list): List of tensors to detach and reattach.
            
        Returns:
            list: Detached tensors with gradients enabled.
        """
        detached = [xx.detach() for xx in x]
        for i, xx in enumerate(detached):
            xx.requires_grad = True
        return(detached)

    def to_device(self) -> None:
        """Move all network layers to the specified device."""
        for layer in range(len(self.layers_list)):
            self.layers_list[layer].to(self.device)

