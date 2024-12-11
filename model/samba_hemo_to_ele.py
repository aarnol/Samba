import torch  
import random 
import numpy as np
import torch.nn as nn 
from einops import rearrange    
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward  
from pytorch_wavelets import DWT1DInverse  
from nn.fmri_2_meg.graphnet import GMWANet  
from nn.utils import data_post, cosine_embedding_loss 
from nilearn.glm.first_level import hemodynamic_models 


class Decoder(nn.Module):
    
  def __init__(self, seq_len, embedding_dim, n_features, dropout):
    super(Decoder, self).__init__()

    self.seq_len, self.embedding_dim = seq_len, embedding_dim
    self.n_features =  n_features
    self.dropout = dropout

    self.rnn = nn.LSTM(
      input_size=self.n_features,
      hidden_size=self.embedding_dim,
      num_layers=2,
      dropout = self.dropout,
      batch_first=True
    ) 
     
    self.output_layer1 = nn.Linear(self.embedding_dim, 100)     # dense output layer; 
    self.output_layer2 = nn.Linear(100, self.n_features)        # dense output layer; 
    self.relu = nn.ReLU()

  def forward(self, x, h_0, c_0):
    size = len(x) 
    x, (hidden_n, cell_n) = self.rnn(x, (h_0, c_0))
    x = x.reshape((size, 1, self.embedding_dim))
    
    x = self.relu(self.output_layer1(x))
    x = F.dropout(x, p=self.dropout, training=True) 
    
    x = self.output_layer2(x)
    x = F.dropout(x, p=self.dropout, training=True) 
    
    x = x.reshape((size, 1, self.n_features))

    return (x, hidden_n, cell_n)


class ThreeLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ThreeLayerMLP, self).__init__() 
        
        self.layer1 = nn.Linear(input_size, hidden_size) 
        self.layer2 = nn.Linear(hidden_size, hidden_size) 
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x): 
        x = F.relu(self.layer1(x)) 
        x = F.relu(self.layer2(x)) 
        x = self.layer3(x)
        return x
    
class ConvTransposeModel(nn.Module):
    def __init__(self, out_channels=200):
        super(ConvTransposeModel, self).__init__()
          
        self.conv_transpose1 = nn.ConvTranspose1d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=30,    # Specified kernel size
            stride=6,          # Keeping an initial guess for stride
            padding=12,        # Minimal padding
            output_padding=0
        )  # Adjusting output padding to correct the length
        # self.upsample = nn.Linear(2000, 12000) 
         
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm1d(out_channels) 
        
    def forward(self, x): 
        x = self.bn1(self.gelu(self.conv_transpose1(x)))   
        # x = self.upsample(x) 
        return x
 
  
 
class samba_hemo_to_ele_v1(nn.Module): 
    def __init__(self, args):
        super(samba_hemo_to_ele_v1, self).__init__()     
        """ MEG --> fMRI
            TimeSpaceFlipFlow_v2
            This function converts MEG signal to fMRI through the following steps:

            1) Perform MEG-based parcel and subject-specific conditioned HRF learning.
            2) Carry out Parcel-based joint embedding learning temporal.
            3) Utilize Multi-head attention graph neural networks.

            The first two steps are temporal modules, while the third is a spatial module.
        """
        self.args = args 
        self.loss_mse = nn.MSELoss()
        
        self.fmri_adjacency_matrix = torch.tensor(np.load(self.args.hemo_adjacency_matrix_dir)).float().to(args.device)
        self.meg_adjacency_matrix = torch.tensor(np.load(self.args.ele_adjacency_matrix_dir)).float().to(args.device) 
        
        self.result_list, self.trlosseslist_rec, self.valosseslist_rec = [], [], [] 
        self.trlosseslist_skip, self.valosseslist_skip = [], []   
        # self.loss_mse_parcells, self.training_counter = [0] * 200, 0 
          
        self.training_counter = 0 
        self.fixed_hrf = hemodynamic_models.glover_hrf(tr=1, oversampling=94)[:3000]
        self.fixed_hrf = torch.tensor(self.fixed_hrf).float().to(args.device) 
        
        n_target_nodes,n_source_nodes = 200, 500 
        graph_dim = 30
        self.graph = GMWANet(
            in_features=graph_dim, 
            out_features=graph_dim, 
            n_heads=1,                 #
            n_source_nodes=n_source_nodes, 
            n_target_nodes=n_target_nodes, 
            dim_head=64, 
            dropout=0.1, 
            device=args.device
        ).to(args.device)
        
         
        wavelet_dim =  graph_dim
        dims = [1004, 506, 257, 257]
        self.dwt = DWT1DForward(wave='db5', J=3, mode='zero').to(args.device) 
        self.idwt = DWT1DInverse(wave='db5', mode='zero').to(args.device) 
        
        self.g1_inv, self.g2_inv, self.g3_inv, self.g4_inv = [], [], [], [] 
        for p in range(n_target_nodes):
            self.g1_inv.append(ThreeLayerMLP(input_size=wavelet_dim, hidden_size=1024, output_size=dims[0]).to(args.device))
            self.g2_inv.append(ThreeLayerMLP(input_size=wavelet_dim, hidden_size=1024, output_size=dims[1]).to(args.device))
            self.g3_inv.append(ThreeLayerMLP(input_size=wavelet_dim, hidden_size=1024, output_size=dims[2]).to(args.device))
            self.g4_inv.append(ThreeLayerMLP(input_size=wavelet_dim, hidden_size=1024, output_size=dims[3]).to(args.device)) 

        self.deconv = ConvTransposeModel().to(args.device) 
          
        num_layers = 2
        dec_features=200 
        hidden_dim = dec_features 
        dec_seq_len=12000
        dec_embedding_dim=200
        
        self.lstm = Decoder(dec_seq_len, dec_embedding_dim, dec_features, dropout=0.1).to(args.device)
        self.h0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_dim))
        self.c0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_dim))
        
    def rnn_forward(self, x, y, teacher_forcing_ratio):  
        batch_size = y.shape[0]
        trg_len = y.shape[1]
        trg_dim = y.shape[-1]  
        
        hidden = self.h0.repeat(1, batch_size, 1)
        cell = self.c0.repeat(1, batch_size, 1)  
        input = y[:, 0, :]     
           
        for t in range(trg_len): 
            input = input.reshape((batch_size, 1, trg_dim))                     # [20, 1, 200] 
            output, hidden, cell = self.lstm(input, hidden, cell)               # [20, 0, 200]      
              
            if t==0:
                outputs = output
            else:
                outputs = torch.cat((outputs, output), dim=1) 
                
            if random.random() < teacher_forcing_ratio:
                input = y[:, t, :] 
            else:
                input = x[:, t, :]   
                     
        return rearrange(outputs, ' b t p ->  b p t') 
    
    def deconv_forward(self, x): 
        for p in range(x.shape[1]):
            x_deconv_p = self.deconv[p](x[:, p, :].unsqueeze(1))
            if p == 0:
                x_deconv = x_deconv_p
            else:
                x_deconv = torch.cat((x_deconv, x_deconv_p), dim=1) 
        return x_deconv
    
    def dewave_forward(self, x, x_meg): 
        for p in range(x.shape[1]):
            x_fmri_graph_p = x[:, p, :]
            g1_inv_p = self.g1_inv[p](x_fmri_graph_p).unsqueeze(1)
            g2_inv_p = self.g2_inv[p](x_fmri_graph_p).unsqueeze(1)
            g3_inv_p = self.g3_inv[p](x_fmri_graph_p).unsqueeze(1)
            g4_inv_p = self.g4_inv[p](x_fmri_graph_p).unsqueeze(1) 
            
            if p == 0:
                xm_inv_wave_l1 = g1_inv_p
                xm_inv_wave_l2 = g2_inv_p
                xm_inv_wave_l3 = g3_inv_p
                xm_inv_wave_l4 = g4_inv_p 
            else:
                xm_inv_wave_l1 = torch.cat((xm_inv_wave_l1, g1_inv_p), dim=1)
                xm_inv_wave_l2 = torch.cat((xm_inv_wave_l2, g2_inv_p), dim=1)
                xm_inv_wave_l3 = torch.cat((xm_inv_wave_l3, g3_inv_p), dim=1)
                xm_inv_wave_l4 = torch.cat((xm_inv_wave_l4, g4_inv_p), dim=1) 
          
          
        fixed_hrf = self.fixed_hrf.unsqueeze(0).unsqueeze(0)   # [1, 1, 3000]
        for i in range(x_meg.shape[1]):
            zm_p_hrf = F.conv1d(x_meg[:, i, :].unsqueeze(1), fixed_hrf, padding=499, stride=5) 
            if i == 0:
                zm_hrf_raw = zm_p_hrf
            else:
                zm_hrf_raw = torch.cat((zm_hrf_raw, zm_p_hrf), dim=1)
        
        zm_hrf_raw = self.dwt(zm_hrf_raw)   
        # import pdb;pdb.set_trace()    
        loss1 = self.loss_mse(zm_hrf_raw[0], xm_inv_wave_l4) 
        loss2 = self.loss_mse(zm_hrf_raw[1][2], xm_inv_wave_l3) 
        loss3 = self.loss_mse(zm_hrf_raw[1][1], xm_inv_wave_l2) 
        loss4 = self.loss_mse(zm_hrf_raw[1][0], xm_inv_wave_l1)  
        skip_loss = (loss1+loss2+loss3+loss4)/4
          
        x_fmri_idwt = self.idwt(
            (xm_inv_wave_l4, [xm_inv_wave_l1, xm_inv_wave_l2, xm_inv_wave_l3])
        ) 
        
        return x_fmri_idwt, skip_loss 
        
    
    def forward(self, x_meg, x_fmri):
        
        if self.train_mode: teacher_forcing_ratio=0.5
        else:teacher_forcing_ratio=0
          
        # graph encoder ([10, 500, 30] -> [10, 30, 200])
        x_fmri_graph, h_att = self.graph( 
            x_fmri,
            self.fmri_adjacency_matrix,  
            self.meg_adjacency_matrix,
        )  
        
        # wavelet decoder ([10, 500, 180] -> [10, 500, 2000])
        x_fmri_idwt, skip_loss = self.dewave_forward(
            x_fmri_graph, x_meg
        ) 
         
        # hrf decoder ([10, 500, 30]  ->  [10, 500, 180])
        x_meg_hat = self.deconv(
            x_fmri_idwt
        )         
             
        x_meg_lstm = self.rnn_forward(
            x=rearrange(x_meg_hat, 'b d t -> b t d'), 
            y=rearrange(x_meg, 'b d t -> b t d'), 
            teacher_forcing_ratio=teacher_forcing_ratio
        )   
           
        return x_meg_lstm, h_att, skip_loss
      
        
    def loss(self, x_meg, x_fmri, ym_batch, yf_batch, iteration, train_mode=False, visualization=False):       
         
        
        self.train_mode = train_mode
        x_meg_hat, h_att, skip_loss = self.forward(x_meg, x_fmri)    
        
        loss_rec = cosine_embedding_loss(
            rearrange(x_meg_hat, 'b d t -> b t d'), 
            rearrange(x_meg, 'b d t -> b t d')
        )
        
        loss = skip_loss+loss_rec   
         
        self.train_mode = train_mode
        if self.train_mode: 
            self.trlosseslist_skip.append(skip_loss.item()) 
            self.trlosseslist_rec.append(loss_rec.item()) 
            self.training_counter += 1   
            return loss
        
        indices = torch.randint(0, x_meg_hat.shape[-1], (30,))
        self.valosseslist_rec.append(loss_rec.item())
        self.valosseslist_skip.append(skip_loss.item())
        alphas = None
        if visualization: 
            return loss, x_meg_hat[:, :, indices].detach().cpu(), x_meg[..., indices].detach().cpu(), h_att, alphas 
        else:
            return loss, x_meg_hat[:, :, indices].detach().cpu(), x_meg[..., indices].detach().cpu()
        
    def print_fn(self, it): 
        prt_meg =  '  itr.: %5d     loss (cos-mse) (t/v):  %4.3f/%4.3f  |  %4.3f/%4.3f '   
        tr_cossloss_average_rec = sum(np.array(self.trlosseslist_rec))/len(self.trlosseslist_rec)
        tr_cossloss_average_skip = sum(np.array(self.trlosseslist_skip))/len(self.trlosseslist_skip)
        
        va_cossloss_average_rec = sum(np.array(self.valosseslist_rec))/len(self.valosseslist_rec)  
        va_cossloss_average_skip = sum(np.array(self.valosseslist_skip))/len(self.valosseslist_skip)  
        self.trlosseslist, self.valosseslist = [], []
        print(prt_meg % (it, tr_cossloss_average_rec, tr_cossloss_average_skip,
                         va_cossloss_average_rec, va_cossloss_average_skip))  
        self.result_list.append(prt_meg % (it, tr_cossloss_average_rec, tr_cossloss_average_skip,
                         va_cossloss_average_rec, va_cossloss_average_skip))  