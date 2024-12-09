import os 
import numpy as np
import tqdm 
import pandas as pd 
from einops import rearrange
import torch

load_dir = '/home/aa2793/scratch/datasets/NEUROSCIENCE/MEGfMRI/forrest_gump/'

class BrainLoader:
    def __init__(self, dataset_mode = 'fmri'):
        """Convert fMRI and MEG's recordings in different rurns/parts to miniute.  
           there are two methods (*_sub_dic_list) uploading the fmri and meg recordings. 
        Args:
            dataset_mode {fmri or meg}: types of the data which could be fmri or meg.
        """ 
        
        self.sub_run_dic, self.sub_run_list = {}, []
        self.run_list, self.sub_list = [], [] 
        # data_dir = source_data_dir + dataset_mode
        
        if dataset_mode == 'fmri': 
            data_dir = load_dir + 'rawdata/fmri/fsaverage_trimmed/'
            self.data_dir = data_dir
            self.dir_list = os.listdir(data_dir) 
            self.fmri_sub_dic_list()  

        elif dataset_mode == 'meg': 
            data_dir = load_dir + 'rawdata/meg/trimmed_new/' 
            self.data_dir = data_dir
            self.dir_list = os.listdir(data_dir)  
            self.meg_sub_dic_list()  

        else:
            raise TypeError("only fmri or meg is allowed!")

        

    def fmri_sub_dic_list(self):
        """ loads a subject's right and left hippocampus fmri recording. 
        """ 
        # iterating through the subjects 
        file_counter = 0 
        for i in range(1, len(self.dir_list)): 
            sub_add_list = True
            if i < 10:
                sub_i = '0'+ str(i)
            else: 
                sub_i = str(i)   
            # iterating through the runs 
            run_list = []
            for j in range(1, len(self.dir_list)):
                if j < 10:
                    run_j = '0'+ str(j)
                else: 
                    run_j = str(j)    
                lh_dir = self.data_dir + 'fmri_sub-'+sub_i+'_ses-movie_task-movie_run-'+run_j+'_fmri_resampled_trimmed_lh.npy'
                rh_dir = self.data_dir + 'fmri_sub-'+sub_i+'_ses-movie_task-movie_run-'+run_j+'_fmri_resampled_trimmed_rh.npy'
                if os.path.exists(lh_dir) and os.path.exists(rh_dir):  
                    self.sub_run_dic[str(file_counter)] = [lh_dir , rh_dir]   
                    self.sub_run_list.append('sub-'+sub_i+'-run-'+run_j) 
                    file_counter += 1 
                    if sub_add_list:
                        sub_add_list = False
                        self.sub_list.append('sub-'+sub_i)
                    run_list.append(run_j)
            if len(run_list) > 0:
                self.run_list.append(run_list)
    

    def meg_sub_dic_list(self):    
        # iterating through the subjects 
        n_sub, n_run, n_parts = 20, 20, 20  
        file_counter = 0
        for i in range(1, n_sub): 
            sub_add_list = True
            if i < 10:
                sub_i = '0'+ str(i)
            else: 
                sub_i = str(i)   
            # iterating through the runs 
            run_list = []
            for j in range(1, n_run):
                if j < 10:
                    run_j = '0'+ str(j)
                else: 
                    run_j = str(j)   
                # iterating through the parts
                sub_run_dic_parts = []
                for k in range(1, n_parts):
                    if k < 10:
                        part_k = '0'+ str(k)
                    else: 
                        part_k = str(k)   
                    mge_dir = self.data_dir + 'meg_sub-'+sub_i+'_ses-movie_task-movie_run-'+run_j+'_trimmed_src_part-'+part_k+'.npy'   
                    if os.path.exists(mge_dir):  
                        sub_run_dic_parts.append(mge_dir)   
                        
                        if sub_add_list:
                            sub_add_list = False
                            self.sub_list.append('sub-'+sub_i)
                        run_list.append(run_j)
                         
                if sub_run_dic_parts:
                    self.sub_run_list.append('sub-'+sub_i+'-run-'+run_j) 
                    self.sub_run_dic[str(file_counter)] = sub_run_dic_parts
                    file_counter += 1  
                     
                if len(run_list) > 0:
                    self.run_list.append(run_list)
 
    def __getitem__(self, i):
        """ loads a subject's right and left hippocampus recording. 
           Args: 
                dir_i path of the *.npy dataset to be download 
            Return:
                [xh_i, xh_i] for fmri 
                [x_i] for meg
        """ 
        sub_run_index = self.sub_run_list[i]  
        for i, data_dir in enumerate(self.sub_run_dic[str(i)]):  
            if i == 0:
                data = torch.tensor(np.load(data_dir)).to(torch.float16)
            else:
                data = torch.cat((data, 
                           torch.tensor(np.load(data_dir)).to(torch.float16)), dim=-1)   
        return data, sub_run_index 