import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from einops import rearrange 
import sys 


class NumpyBatchDataset(Dataset):
    def __init__(self, meg_dir, fmri_dir, split, n_way, meg_sub_list, fmri_sub_list, single_subj=False):  
        self.meg_dir = meg_dir  
        self.fmri_dir = fmri_dir  
        self.n_way = n_way 
        self.label = [x for x in range(self.n_way)] 
        self.label = Variable(torch.tensor(self.label))  
        self.split = split
        self.single_subj = single_subj
        
        self.meg_subs = meg_sub_list
        self.fmri_subs = fmri_sub_list 
         
        self.test_splits = ['079_1', '066_1', '021_0', '045_1', '012_1', '063_0', '080_0', '065_1', '084_1', '077_0', 
                            '048_1', '042_1', '007_0', '091_0', '032_1', '075_1', '060_0', '004_1', '103_1', '100_0']
        
        self.valid_splits = ['030_1', '040_0', '019_1', '089_0', '052_0', '002_1', '102_1']
        
        self.train_splits = ['001_0', '001_1', '002_0', '003_0', '003_1', '004_0', '005_0', '005_1', 
                             '006_0', '006_1', '007_1', '008_0', '008_1', '009_0', '009_1', '010_0', '010_1', 
                             '011_0', '011_1', '012_0', '013_0', '013_1', '014_0', '014_1', '015_0', '015_1', 
                             '016_0', '016_1', '017_0', '017_1', '018_0', '018_1', '019_0', '020_0', '020_1', 
                             '021_1', '022_0', '022_1', '023_0', '023_1', '024_0', '024_1', '025_0', '025_1', 
                             '026_0', '026_1', '027_0', '027_1', '028_0', '028_1', '029_0', '029_1', '030_0', 
                             '031_0', '031_1', '032_0', '033_0', '033_1', '034_0', '034_1', '035_0', '035_1', 
                             '036_0', '036_1', '037_0', '037_1', '038_0', '038_1', '039_0', '039_1', '040_1', 
                             '041_0', '041_1', '042_0', '043_0', '043_1', '044_0', '044_1', '045_0', 
                             '046_0', '046_1', '047_0', '047_1', '048_0', '049_0', '049_1', '050_0', '050_1', 
                             '051_0', '051_1', '052_1', '053_0', '053_1', '054_0', '054_1', '055_0', '055_1', 
                             '056_0', '056_1', '057_0', '057_1', '058_0', '058_1', '059_0', '059_1', '060_1', 
                             '061_0', '061_1', '062_0', '062_1', '063_1', '064_0', '064_1', '065_0', 
                             '066_0', '067_0', '067_1', '068_0', '068_1', '069_0', '069_1', '070_0', '070_1', 
                             '071_0', '071_1', '072_0', '072_1', '073_0', '073_1', '074_0', '074_1', '075_0',   
                             '076_0', '076_1', '077_1', '078_0', '078_1', '079_0', '080_1', 
                             '081_0', '081_1', '082_0', '082_1', '083_0', '083_1', '084_0', '085_0', '085_1', 
                             '086_0', '086_1', '087_0', '087_1', '088_0', '088_1', '089_1', '090_0', '090_1', 
                             '091_1', '092_0', '092_1', '093_0', '093_1', '094_0', '094_1', '095_0', '095_1', 
                             '096_0', '096_1', '097_0', '097_1', '098_0', '098_1', '099_0', '099_1', '100_1', 
                             '101_0', '101_1', '102_0', '103_0', '104_0', '104_1', '105_0', '105_1', 
                             '106_0', '106_1']
         
        self.train_splits = list(filter(lambda item: item not in self.test_splits, self.train_splits)) 
        
    def rand_sub(self):
        rand_meg_sub, rand_fmri_sub = [], []
        for w in range(self.n_way): 
            rand_meg_sub.append(str(np.random.choice(self.meg_subs, 1, replace=False).tolist()[0]))
            rand_fmri_sub.append(str(np.random.choice(self.fmri_subs, 1, replace=False).tolist()[0])) 
 
        return rand_meg_sub, rand_fmri_sub

    def rand_time(self):  
        return np.random.choice(self.class_list, 1, replace=False).tolist()
    
    def normalize_tensor(self, input_tensor, desired_mean, desired_std): 
        current_mean = input_tensor.mean()
        current_std = input_tensor.std()

        # Normalize the tensor to have mean 0 and std 1.
        normalized_tensor = (input_tensor - current_mean) / current_std

        # Scale and shift the normalized tensor to have the desired mean and std.
        final_tensor = normalized_tensor * desired_std + desired_mean

        return final_tensor
    
    def normalize_01_tensor(self, x):
        """
        `Normalize a tensor to have values between 0 and 1

        Args:
        x (torch.Tensor): Input tensor to normalize.

        Returns:
        torch.Tensor: Normalized tensor with values between 0 and 1.
        """
        x_min = x.min()
        x_max = x.max()
        x_norm = (x - x_min) / (x_max - x_min)
        return x_norm
        
    def __getitem__(self, idx):
        rand_meg_sub, rand_fmri_sub = self.rand_sub() 
        y_batch, y_meta = [], []
        rand_time_points = []
        for w in range(self.n_way):
            rand_meg_sub_w = rand_meg_sub[w]
            rand_fmri_sub_w = rand_fmri_sub[w] 
             
            n_folders_m = self.n_folders(os.path.join(self.meg_dir, 'sub-'+ rand_meg_sub_w)+'/')
            n_folders_f = self.n_folders(os.path.join(self.fmri_dir, 'sub-'+ rand_fmri_sub_w)+'/') 
            
            if self.split == 'train':
                n_time = min(n_folders_m, n_folders_f, 106) 
                self.n_time_pints_train(n_time = n_time)   
                self.class_list = self.train_splits
                # self.class_list = [item for item in self.class_list if item not in self.test_splits]
                # self.class_list = [item for item in self.class_list if item not in self.valid_splits] 
                
            elif self.split == 'test':
                self.class_list = self.test_splits

            elif self.split == 'valid':
                self.class_list = self.valid_splits

            rand_w = self.rand_time()[0]   
            x_meg_w = torch.load(os.path.join(self.meg_dir, 'sub-'+ rand_meg_sub_w, ('sub-'+ rand_meg_sub_w+'-min-'+rand_w+'.pt'))).squeeze(2).unsqueeze(0) 
            x_fmri_w = torch.load(os.path.join(self.fmri_dir, 'sub-'+ rand_fmri_sub_w, ('sub-'+ rand_fmri_sub_w+'-min-'+rand_w+'.pt'))).squeeze(2).unsqueeze(0) 
            
            x_meg_w = torch.nan_to_num(x_meg_w, nan=2.0, posinf=0.0)
            x_fmri_w = torch.nan_to_num(x_fmri_w, nan=2.0, posinf=0.0)
            
            x_meg_w = rearrange(x_meg_w, 'b t p d -> b p (d t)') 
            x_fmri_w = rearrange(x_fmri_w, 'b d p -> b p d')
            
            x_meg_w = self.normalize_tensor(input_tensor=x_meg_w, desired_mean=0.00498728035017848, desired_std=0.7188116312026978)
            x_fmri_w = self.normalize_tensor(input_tensor=x_fmri_w, desired_mean=0.00498728035017848, desired_std=0.7188116312026978)
               
            y_batch.append(list(map(int, rand_w[:-2])))
            y_meta.append(self.label)  
            rand_time_points.append(rand_w[:-2])
            if w == 0:
                x_meg = x_meg_w
                x_fmri = x_fmri_w
            else:
                x_meg = torch.cat((x_meg, x_meg_w), dim=0)
                x_fmri = torch.cat((x_fmri, x_fmri_w), dim=0)
        
        x_meg = torch.stack([arr for arr in x_meg]).float() 
        x_fmri = torch.stack([arr for arr in x_fmri]).float()    
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  
        return (x_meg.to(device), x_fmri.to(device), y_meta, y_batch), [rand_time_points, rand_fmri_sub, rand_meg_sub] 

    def n_time_pints_train(self, n_time):    
        self.class_list = []
        class_couter = 1
        for i in range(n_time):
            if class_couter < 10:
                self.class_list.append('00'+str(class_couter))
            elif class_couter > 99:
                self.class_list.append(str(class_couter))
            else:
                self.class_list.append('0'+str(class_couter)) 
            class_couter += 1

    def n_folders(self, dir_path):  
        return len(os.listdir(dir_path))
  
    def __len__(self):
        return 1
    
    

# path = '/home/aa2793/gibbs/arman/datasets/NEUROSCIENCE/MEGfMRI/forrest_gump/30_second_dataset/fmri-200parcell-avg/sub-02/'    
# source_file_path_i = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

# sample_lists = [x[11:] for x in source_file_path_i]
# sample_lists = [x[:-3] for x in sample_lists]

# import pdb;pdb.set_trace()