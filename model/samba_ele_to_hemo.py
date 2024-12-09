 
import numpy as np

import torch
import torch.nn as nn
from einops import rearrange

from args import cosine_embedding_loss
from nn.temporal_encoder import PerParcelHrfLearning, WaveletAttentionNet



class SambaEleToHemo(nn.Module):
    """
    Converts electrophysiological (ele) recordings to hemodynamic (hemo) signals.

    This module incorporates a hierarchical model consisting of:
    1. Per-parcel hemodynamic response function (HRF) learning.
    2. A temporal encoder that uses a wavelet attention network to process
       high-frequency electrophysiological data into a compressed representation
       resembling hemodynamic characteristics.
    3. An attention-based graph upsampling network as a spatial decoder to
       accurately map the input into the hemodynamic space.
    """
    def __init__(self, args):
        super(SambaEleToHemo, self).__init__()
        self.args = args
        self.result_list = []
        self.training_losses = []
        self.validation_losses = []

        self.initialize_networks(args)
        self.mse_loss = nn.MSELoss()

    def initialize_networks(self, args):
        """
        Initializes the network components for HRF learning, temporal encoding,
        and spatial decoding.
        """
        
        # Per-parcel hrf-learning
        self.hrf_learning = PerParcelHrfLearning(args)
        
        # Per-parcel attention-based wavelet dcomposition learning
        self.temporal_encoder = WaveletAttentionNet(args)
        
        # Over parcels graph decoder/upsampling
        from nn.spatial_decoder import GMWANet  
        self.spatial_decoder = GMWANet(args).to(args.device)

    def forward(self, x_ele, x_hemo, sub_ele, sub_hemo):
        """
        Processes electrophysiological and hemodynamic data through the network.

        Args:
        - x_ele (torch.float32): Electrophysiological data, shape [batch_size, ele_spatial_dim, ele_temporal_dim].
        - x_hemo (torch.float32): Hemodynamic data, shape [batch_size, hemo_spatial_dim, hemo_temporal_dim].
        - sub_ele (list): List of electrophysiological subject indices for loading adjacency matrices.
        - sub_hemo (list): List of hemodynamic subject indices for loading adjacency matrices.

        Returns:
        - x_hemo_hat (torch.float32): Reconstruction of x_hemo.
        - x_ele_hrf (torch.float32): Inferred per-parcel HRFs.
        - alphas (torch.float32): Inferred wavelet attentions.
        - h_att (torch.float32): Attention graph tensor as a squared matrix.
        """
        
        batch_size = x_ele.shape[0]

        # HRF learning 
        x_ele_hrf = self.hrf_learning(x_ele)                            # [10, 200, 12000]  --> [10, 200, 12000]
  
        # Temporal encoding with wavelet attention
        x_ele_wavelet, alphas = self.temporal_encoder(x_ele_hrf)        #  [10, 200, 12000]  -> [10, 200, 30]

        # Spatial decoding with graph attention
        teacher_forcing_ratio = self.args.ele_to_hemo_teacher_forcing_ratio if self.training else 0.0
        
        x_hemo_hat, h_att = [], []
        if self.args.mc_probabilistic: 
            for i in range(self.args.mc_n_sampling):
                x_hemo_hat_i, h_att_i = self.spatial_decoder(x_ele_wavelet, x_hemo, batch_size, sub_hemo, sub_ele, teacher_forcing_ratio) 
                x_hemo_hat.append(x_hemo_hat_i.unsqueeze(0))
                h_att.append(h_att_i.unsqueeze(0)) 
        else:
            x_hemo_hat, h_att = self.spatial_decoder(
                x_ele_wavelet, x_hemo, batch_size, sub_hemo, sub_ele, teacher_forcing_ratio
            )
            
        if self.args.mc_probabilistic: 
            x_hemo_hat = torch.cat(x_hemo_hat, dim=0)
            h_att = torch.cat(h_att, dim=0)
        
            x_hemo_hat_std = torch.std(x_hemo_hat, dim=0)
            h_att_std = torch.std(h_att, dim=0)
            
            x_hemo_hat = torch.mean(x_hemo_hat, dim=0)
            h_att = torch.mean(h_att, dim=0)
            
        else:
            x_hemo_hat_std, h_att_std = None, None
             
        # Reshape the HRF (will be used for visualization later) 
        x_ele_hrf = x_ele_hrf.view(-1, x_ele.shape[1], x_ele_hrf.shape[-1])

        return x_hemo_hat, x_hemo_hat_std, x_ele_hrf, alphas, h_att, h_att_std

    def loss(self, x_ele, x_hemo, sub_f, sub_m, iteration):
        """
        Calculates and records the loss between predicted and actual hemodynamic data. 
        
        Args:
        - x_ele (torch.Tensor): Electrophysiological data.
        - x_hemo (torch.Tensor): Actual hemodynamic data.
        - sub_f (list): Indices for electrophysiological subjects.
        - sub_m (list): Indices for hemodynamic subjects.
        - iteration (int): Current iteration number (unused in the method but might be useful for logging).

        Returns:
        - loss (torch.Tensor): Computed loss for the current forward pass.
        - x_hemo_hat (torch.Tensor): Predicted hemodynamic data, returned only during validation.
        - x_hemo (torch.Tensor): Ground truth hemodynamic data, returned only during validation.
        - [zm_hrf, h_att, alphas] (list): Additional outputs from the model, returned only during validation.
        """
         
        x_hemo_hat, x_hemo_hat_std, zm_hrf, alphas, h_att, h_att_std = self.forward(x_ele, x_hemo, sub_f, sub_m)
          
        loss = self.mse_loss(
            rearrange(x_hemo_hat, 'b d t -> b t d'),   
            rearrange(x_hemo, 'b d t -> b t d')
        )  
         
        # Append the loss for later printing 
        if self.training:
            self.training_losses.append(loss.item())
            return loss
        else:
            self.validation_losses.append(loss.item())
            # During validation, also return the predictions,...
            return loss, x_hemo.detach().cpu(), x_hemo_hat.detach().cpu(), [zm_hrf, h_att, alphas], [x_hemo_hat_std.detach().cpu(), h_att_std.detach().cpu()]
    
  
   
    def print_results(self, it):
        """
        Prints and logs the average training and validation losses for a given iteration.
        Args:
        - it (int): The current iteration number.
        """
        
        # Calculate average losses  
        tr_cossloss_average = sum(self.training_losses) / len(self.training_losses) if self.training_losses else 0
        va_cossloss_average = sum(self.validation_losses) / len(self.validation_losses) if self.validation_losses else 0

        # Clear the lists 
        self.training_losses, self.validation_losses = [], []

        # Print
        result = f'itr.: {it:5d} loss (train/valid): {tr_cossloss_average:.3f}/{va_cossloss_average:.3f}'
        print(result)

        # Append the result
        self.result_list.append(result)
