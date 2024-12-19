import torch.nn as nn
import torch.nn.functional as F 
import torch
import random 
import numpy as np 
from einops import rearrange
from labml_helpers.module import Module
# from nn.jointrecurrent import Encoder, Decoder

 
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__() 
        
        self.layer1 = nn.Linear(input_size, hidden_size) 
        self.layer2 = nn.Linear(hidden_size, hidden_size) 
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x): 
        x = F.relu(self.layer1(x)) 
        x = F.relu(self.layer2(x)) 
        x = self.layer3(x)
        return x

class GraphAttentionV2Layer(Module): 
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True, 
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = False): 
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights 
        if is_concat:                                                     
            assert out_features % n_heads == 0 
            self.n_hidden = out_features // n_heads                       
        else: 
            self.n_hidden = out_features 
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)  
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False) 
        self.attn = nn.Linear(self.n_hidden, 1, bias=False) 
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope) 
        self.softmax = nn.Softmax(dim=1) 
        # self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor): 
         
        # import pdb;pdb.set_trace() 
        n_nodes = h.shape[0]                                                # n_nodes = 200
        g_l = self.linear_l(h).view(n_nodes, self.n_heads, self.n_hidden)   # [200, 8, 128]  
        g_r = self.linear_r(h).view(n_nodes, self.n_heads, self.n_hidden)   # [200, 8, 128]     
         
        g_l_repeat = g_l.repeat(n_nodes, 1, 1)                              # [40000, 8, 128] 
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=0)       # [40000, 8, 128]
        
        g_sum = g_l_repeat + g_r_repeat_interleave                          # [40000, 8, 128]
        g_sum = g_sum.view(n_nodes, n_nodes, self.n_heads, self.n_hidden)   # [200, 200, 8, 128] 
        e = self.attn(self.activation(g_sum))                               # [200, 200, 8, 1]
 
        e = e.squeeze(-1)                                                   # [200, 200, 8, 1]
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads
        
        e = e.masked_fill(adj_mat == 0, float('-inf'))                     # [200, 200, 8]
        a = self.softmax(e)                                                # [200, 200, 8]
        # a = self.dropout(a) 
        attn_res = torch.einsum('ijh,jhf->ihf', a, g_r)                    # [200, 8, 128]
        if self.is_concat: 
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden) # [200, 1024]
        else: 
            return attn_res.mean(dim=1), a.squeeze(-1)                     # [200, 128]
        

class GMWA(nn.Module):  
    """Graph Mapper With Attenation 
    aims at upsample or downsample from source_graph structure to target_graph structure 
    Args:
        nn (_type_): _description_
    """
    def __init__(self, in_features, out_features, n_heads, n_source_nodes, n_target_nodes, dim_head=64, dropout=0.1, device='cuda'):
        super(GMWA, self).__init__()
        
        self.device = device
        self.n_patches = 20 
        
        self.drop = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity() 
        self.gat_layer1 = GraphAttentionV2Layer(
            in_features=in_features, 
            out_features=out_features, 
            n_heads=n_heads, 
            is_concat=False, 
            share_weights=False
        )   
        
        self.gat_layer2 = GraphAttentionV2Layer(
            in_features=in_features, 
            out_features=out_features,
            n_heads=n_heads, 
            is_concat=False, 
            share_weights=False
        )   
           
        enc_seq_len=in_features 
        enc_features= 500 #200 
        enc_embedding_dim=500 #200 
        
        dropout=0.5
        device=device  
          
        dec_embedding_dim=500
        dec_features=500  
        
        n_super_source_nodes = int(n_source_nodes/self.n_patches)
        n_super_target_nodes = int(n_target_nodes/self.n_patches)
        self.mapper = []
        for _ in range(self.n_patches):
            self.mapper.append(nn.Linear(n_super_source_nodes, n_super_target_nodes).to(device)
        )  
        self.activation = nn.ELU()   
        
        
    def g_upsampling(self, z): 
        z = rearrange(z, '(d1 d2) t -> t d1 d2', d1 = self.n_patches) 
        for p in range(self.n_patches):
            z_mapper_i = self.mapper[p](z[:, p, :]).unsqueeze(1)
            if p==0:
                z_mapper = z_mapper_i
            else:
                z_mapper = torch.cat((z_mapper, z_mapper_i), dim=1)
        
        z_mapper = rearrange(z_mapper, 't p1 p2 -> (p1 p2) t')  
        return z_mapper
    
    def forward(self, x, adj_source, adj_target):  
        z, _ = self.gat_layer1(x, adj_source.unsqueeze(-1))   # [200, 256] 
        z = self.g_upsampling(z)                              # [500, 256]   
        z, att = self.gat_layer2(z, adj_target.unsqueeze(-1))
        return rearrange(z, 'd t -> t d').unsqueeze(0) , att.unsqueeze(0) 
 
          
           
 
class GMWANet(nn.Module):
    def __init__(self, in_features, out_features, n_heads, n_source_nodes, n_target_nodes, dim_head, dropout, device):
        super(GMWANet, self).__init__()  
        
        self.gmwa_layer = GMWA(
            in_features, out_features, n_heads, 
            n_source_nodes, 
            n_target_nodes, 
            dim_head, 
            dropout,
            device
        )  
        
    def forward(self, x, meg_adjacency_matrix, fmri_adjacency_matrix):   
        
        for b in range(x.shape[0]):   
            xfm_graph_b, h_att_b = self.gmwa_layer(
                x[b, :, :],  
                meg_adjacency_matrix.float(),  
                fmri_adjacency_matrix.float() 
            )     
                
            if b==0:
                xfm_graph = xfm_graph_b 
                h_att = h_att_b
            else:
                xfm_graph = torch.cat((xfm_graph, xfm_graph_b), dim=0)   
                h_att = torch.cat((h_att, h_att_b), dim=0)  
                    
        return rearrange(xfm_graph, 'b t p -> b p t'), h_att 
    

 