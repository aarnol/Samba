import os 
import numpy as np
import tqdm 
import pandas as pd 
from einops import rearrange
import torch
from schaeferparcel_kong2022_17network import SchaeferParcel_Kong2022_17Network

data_source = '/home/aa2793/gibbs/arman/datasets/NEUROSCIENCE/MEGfMRI/forrest_gump/minute_dataset/'
 
def parselizer(x, parcels, dataset_mode): 
    parcel_index = set(parcels)
    first_time = True
    for idx in parcel_index: 
        # parcel_idx = parcels == idx
        parcel_idx = [element == idx for element in parcels]
        if dataset_mode == 'fmri':
            x_region = x[:, parcel_idx].mean(-1).unsqueeze(1).unsqueeze(1)
        elif dataset_mode == 'meg':
            # import pdb;pdb.set_trace()
            x_region = x[:, :, parcel_idx].mean(-1).unsqueeze(1).unsqueeze(1)
            # x_region = x[:, :, parcel_idx].mean(-1).unsqueeze(1).unsqueeze(1)
        if first_time and idx > 0: # the 0 is reserved for background
            first_time = False
            x_out = x_region
        elif idx>0:
            x_out = torch.cat((x_out, x_region), dim=1) 
    return x_out
    
def parcell_avg_roi(schaefer_parcel_number, parcel_labels, dataset_mode = 'meg'): 
    if dataset_mode == 'meg': 
        sub_list = ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11']   
    elif dataset_mode == 'fmri':
        sub_list = ['02', '03', '04', '05', '06', '09', '10', '14', '15', '16', '17', '18', '19', '20'] 
    else:
        raise ValueError("dataset type should be either meg or fmri!")
         
    load_dir = data_source + dataset_mode + '/' 
    save_dir = data_source + dataset_mode + '-' +str(schaefer_parcel_number)+'parcell-avg/' 
    
    if not os.path.exists(save_dir): 
        os.mkdir(save_dir)  
        
    for sub_index in sub_list:
        save_dir_temp = save_dir+'sub-'+sub_index+'/'
        if not os.path.exists(save_dir_temp): 
            os.mkdir(save_dir_temp)    
        for minute_counter in range(1, 107):
            sub_name = 'sub-' + sub_index 
            load_dir_i = load_dir +  sub_name + '/'  
            if minute_counter<10:
                file_path =  '-min-00'+str(minute_counter)+'.pt'
            elif minute_counter>99:
                file_path = '-min-'+str(minute_counter)+'.pt'
            else:
                file_path = '-min-0'+str(minute_counter)+'.pt'  
            x = torch.load(load_dir_i+ sub_name+file_path) 
            x = [torch.nan_to_num(x_i, nan=0.0, posinf=0.0) for x_i in x]
            # x = torch.cat(x, dim=0)
            for i, x_i in enumerate(x):
                x_i = x_i.unsqueeze(0)
                if i == 0:
                    x = x_i
                else:
                    x = torch.cat((x, x_i), dim=0)
            # import pdb;pdb.set_trace()        
            x_parcels = parselizer(x, parcel_labels, dataset_mode)       
            torch.save(x_parcels, save_dir_temp+'sub-'+sub_index+ file_path) 
            print('sub-'+sub_index+ file_path, ': ', x_parcels.shape)
           
if __name__ == '__main__':
    schaefer_parcel_number = 200
    dataset_mode = 'fmri'

    parcels_dir = '/home/aa2793/palmer_scratch/datasets/fmri-meg/forrest_gump/schaefer_parcellation_labels/'
    parcel_labels, parcel_names, ctabs = SchaeferParcel_Kong2022_17Network(parcel_number = schaefer_parcel_number)  

    parcell_avg_roi(schaefer_parcel_number, parcel_labels, dataset_mode)