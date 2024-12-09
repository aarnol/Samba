import os 
import numpy as np
import tqdm 
import pandas as pd 
from einops import rearrange
from prep_00_brainloader import BrainLoader
import torch
 
save_dir = '/home/aa2793/scratch/datasets/NEUROSCIENCE/MEGfMRI/forrest_gump/minute_dataset/'

def fmri_minute_conversion():   
    dataset_mode = 'fmri'
    data_loader = BrainLoader(dataset_mode = dataset_mode)   
    sampling_rate = 30
    sub_list = ['01', '02', '03', '04', '05', '06', '09', '10', '14', '15', '16', '17', '18', '19', '20']
    sub_counter = 0
    first_time = True
    sub_run_index = []
    for i, (x_i, sub_run_index_i) in enumerate(data_loader): 
        import pdb;pdb.set_trace()
        print(x_i.shape) 
        if sub_run_index_i[4:6] == sub_list[sub_counter] and first_time:  
            x = x_i
            sub_run_index.append(sub_run_index_i) 
            first_time = False
            
        elif sub_run_index_i[4:6] == sub_list[sub_counter]:
            x = torch.cat((x, x_i), dim=0) 
            sub_run_index.append(sub_run_index_i) 
            
        elif sub_run_index_i[4:6] != sub_list[sub_counter]:
            print(x.shape)
            print(sub_run_index)
            
            # save_dir_i = '/vast/palmer/scratch/krishnaswamy_smita/aa2793/fmri-meg/'+dataset_mode+'/'+sub_run_index_i[:6] 
            save_dir_i = save_dir+dataset_mode 
            if not os.path.exists(save_dir_i): 
                os.mkdir(save_dir_i)
            save_dir_i = save_dir_i+'/'+sub_run_index_i[:6] 
            if not os.path.exists(save_dir_i): 
                os.mkdir(save_dir_i)
                    
            x_residual = x[int(x.shape[0]/sampling_rate)*sampling_rate:, :]
            x = x[:int(x.shape[0]/sampling_rate)*sampling_rate, :] 
            x = rearrange(x, '(m s) d -> m s d', s=sampling_rate) 
            for m in range(1, x.shape[0]+1):
                x_min_i = x[m-1, :, :] 
                if m<10:
                    torch.save(x_min_i, save_dir_i + '/'+sub_run_index_i[:6] + '-min-00'+str(m)+'.pt') 
                elif m>99:
                    torch.save(x_min_i, save_dir_i + '/'+sub_run_index_i[:6] +'-min-'+str(m)+'.pt') 
                else:
                    torch.save(x_min_i, save_dir_i + '/'+sub_run_index_i[:6] + '-min-0'+str(m)+'.pt')  
            
            torch.save(x_residual, save_dir_i  + '/'+sub_run_index_i[:6] +'-min-'+str(m+1)+'.pt')      
            
            print('-----------------------------------------------------------------------------------') 
            x = x_i 
            sub_run_index = [] 
            sub_run_index.append(sub_run_index_i) 
            sub_counter += 1
        
    print('-----------------------------------------------------------------------------------') 
    # for the last subject 
    save_dir_i = save_dir+dataset_mode 
    if not os.path.exists(save_dir_i): 
        os.mkdir(save_dir_i)
    save_dir_i = save_dir_i+'/'+sub_run_index_i[:6] 
    if not os.path.exists(save_dir_i): os.mkdir(save_dir_i)
            
    x_residual = x[int(x.shape[0]/sampling_rate)*sampling_rate:, :]
    x = x[:int(x.shape[0]/sampling_rate)*sampling_rate, :] 
    x = rearrange(x, '(m s) d -> m s d', s=sampling_rate) 
    for m in range(1, x.shape[0]+1):
        x_i = x[m-1, :, :] 
        if m<10:
            torch.save(x_i, save_dir_i + '/'+sub_run_index_i[:6] + '-min-00'+str(m)+'.pt') 
        elif m>99:
            torch.save(x_i, save_dir_i + '/'+sub_run_index_i[:6] +'-min-'+str(m)+'.pt') 
        else:
            torch.save(x_i, save_dir_i + '/'+sub_run_index_i[:6] + '-min-0'+str(m)+'.pt')  

    torch.save(x_residual, save_dir_i + '/'+sub_run_index_i[:6] +'-min-'+str(m+1)+'.pt') 
    
    
# torch.Size([450, 20484])
def meg_to_fmri_align(save_dir): 
    dataset_mode = 'meg_aligned_to_fmri'
    data_meg = BrainLoader(dataset_mode = dataset_mode) 
    meg_sub_list = ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11'] 
    sampling_rate = 400
    sub_list = meg_sub_list

    # save_dir = '/vast/palmer/scratch/krishnaswamy_smita/aa2793/fmri-meg/'+dataset_mode 
    save_dir = save_dir+dataset_mode
    if not os.path.exists(save_dir): 
        os.mkdir(save_dir)
    x_residual = []  
    for i, (x_i, sub_run_index_i) in enumerate(data_meg): 
        
        x_i = rearrange(x_i, 'a b -> b a') 
    
        # should be removed 
        if sub_run_index_i[7:] in ['run-01', 'run-02', 'run-07']:
            x_residual_i = torch.cat((x_i[:100, :], x_i[x_i.shape[0]-100:, :]), dim=0)
            x_i = x_i[100:x_i.shape[0]-100]
        
        elif sub_run_index_i[7:] == 'run-03':
            x_residual_i = torch.cat((x_i[:108, :], x_i[x_i.shape[0]-108:, :]), dim=0)
            x_i = x_i[108:x_i.shape[0]-108]
            
        elif sub_run_index_i[7:] == 'run-04':
            x_residual_i = torch.cat((x_i[:56, :], x_i[x_i.shape[0]-56:, :]), dim=0)
            x_i = x_i[56:x_i.shape[0]-56]
        
        elif sub_run_index_i[7:] == 'run-08':
            x_residual_i = torch.cat((x_i[:252, :], x_i[x_i.shape[0]-252:, :]), dim=0)
            x_i = x_i[252:x_i.shape[0]-252]
            
        # should be added 
        elif sub_run_index_i[7:] == 'run-05':
            x_i = torch.cat((x_residual[:68, :], x_i, x_residual[:68, :]), dim=0) 
            x_residual = x_residual[68:x_residual.shape[0]-68]
            
        elif sub_run_index_i[7:] == 'run-06':
            x_i = torch.cat((x_residual[:48, :], x_i, x_residual[:48, :]), dim=0) 
            x_residual = x_residual[48:x_residual.shape[0]-48]
            
        if sub_run_index_i[7:] == 'run-01':
            x_residual = x_residual_i
        if sub_run_index_i[7:] in ['run-02', 'run-03', 'run-04', 'run-05', 'run-07', 'run-08']:
            x_residual = torch.cat((x_residual, x_residual_i), dim=0)
    
        print(sub_run_index_i, x_i.shape, x_residual.shape)
        print('-----------------------------------------------------')
        
        # save_dir_i = '/vast/palmer/scratch/krishnaswamy_smita/aa2793/fmri-meg/'+dataset_mode+'/'+sub_run_index_i[:6] 
        save_dir_i = save_dir +'/'+sub_run_index_i[:6] 
        if not os.path.exists(save_dir_i): 
            os.mkdir(save_dir_i)   
        torch.save(x_i, save_dir_i  + '/'+sub_run_index_i +'.pt') 
        torch.save(x_residual, save_dir_i  + '/'+sub_run_index_i[:6] +'_residual.pt') 
    
def meg_minute_conversion(save_dir, regions_of_interest=None, parcels_name=None):
    dataset_mode = 'meg'
    meg_sub_list = ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11']  
    meg_run_list = ['01', '02', '03', '04', '05', '06', '07', '08']
    load_dir = save_dir+'meg_aligned_to_fmri/'
    save_dir = save_dir+dataset_mode+'/'
    if not os.path.exists(save_dir): 
        os.mkdir(save_dir)
    
    sampling_rate = 200*60 
    for sub_index  in meg_sub_list:
        sub_name = 'sub-' + sub_index 
        if not os.path.exists(save_dir+sub_name): 
            os.mkdir(save_dir+sub_name)
        minute_counter1 = 1
        minute_counter2 = 1
        x_residual = None
        for run_index in meg_run_list:
            file_name = sub_name + '/' +sub_name + '-run-' + run_index
            
            load_dir_i = load_dir + file_name
            
            x = torch.load(load_dir_i +'.pt')  
            if x_residual is not None:
                x = torch.cat((x, x_residual), dim=0)
                x_residual = None     # rest the pervious residual
            
            x_residual = x[int(x.shape[0]/sampling_rate)*sampling_rate:, :]
            x = x[:int(x.shape[0]/sampling_rate)*sampling_rate, :] 
            x = rearrange(x, '(m s) d -> m s d', s=sampling_rate) 
            
            save_dir_i = save_dir +  sub_name + '/' + sub_name
            print(file_name, x.shape)  
            for m in range(1, x.shape[0]+1):
                x_min_i = x[m-1, :, :]  
                x_min_i = rearrange(x_min_i, '(m s) d -> m s d', s=400) 
                if minute_counter1<10:
                    torch.save(x_min_i, save_dir_i + '-min-00'+str(minute_counter1)+'.pt') 
                elif minute_counter1>99:
                    torch.save(x_min_i, save_dir_i + '-min-'+str(minute_counter1)+'.pt') 
                else:
                    torch.save(x_min_i, save_dir_i + '-min-0'+str(minute_counter1)+'.pt') 
                minute_counter1 += 1    
    
if __name__ == '__main__': 
    fmri_minute_conversion() 
    # meg_to_fmri_align(save_dir)
    meg_minute_conversion(save_dir)