import os 
import numpy as np
import tqdm 
import pandas as pd 
from einops import rearrange
import torch
from schaeferparcel_kong2022_17network import SchaeferParcel_Kong2022_17Network

data_source = './datasets/NEUROSCIENCE/MEGfMRI/forrest_gump/minute_dataset/'
data_save = './datasets/NEUROSCIENCE/MEGfMRI/forrest_gump/second_dataset/'

fmri_miniute_dataset = ['fmri-200parcell-avg',  'fmri-500parcell-avg']
fmri_sub_list = ['02', '03', '04', '05', '06', '09', '10', '14', '15', '16', '17', '18', '19', '20'] 


meg_miniute_dataset = ['meg-200parcell-avg',  'meg-500parcell-avg']
meg_sub_list = ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11'] 


def second_data_creator(sub_list, miniute_dataset):  
    
    for miniute_dataset_i in miniute_dataset: 
        save_path = data_save + miniute_dataset_i
        soruce_path = data_source + miniute_dataset_i
        
        if not os.path.exists(save_path): 
            os.mkdir(save_path)
            

        for sub_indx in sub_list:
            sub_name = 'sub-'+sub_indx
            save_path_i = save_path+'/'+sub_name
            source_path_i = soruce_path+'/'+sub_name
            
            if not os.path.exists(save_path_i): os.mkdir(save_path_i)
            source_file_path_i = [f for f in os.listdir(source_path_i) if os.path.isfile(os.path.join(source_path_i, f))]
            
            for source_file_i in source_file_path_i: 
                x = torch.load(source_path_i + '/' + source_file_i) 
                x_0 = x[0::2, :, :] 
                x_1 = x[1::2, :, :]
                
                torch.save(x_0, save_path_i + '/' + source_file_i[:-3]+'_0.pt')
                torch.save(x_1, save_path_i + '/' + source_file_i[:-3]+'_1.pt')
         
second_data_creator(meg_sub_list, meg_miniute_dataset) 
second_data_creator(fmri_sub_list, fmri_miniute_dataset)
    
    
