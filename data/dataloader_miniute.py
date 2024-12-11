import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from einops import rearrange 
import sys 


class NumpyDataset(Dataset):
    def __init__(self, meg_dir, fmri_dir): 
        
        self.npy_files = [] 
        # self.labels = [] # for subject based labels  
        for i, subdir in enumerate(sorted(os.listdir(meg_dir))):
            for file in sorted(os.listdir(os.path.join(meg_dir, subdir))):
                if file.endswith('.npy'):
                    self.npy_files.append(os.path.join(meg_dir, subdir, file))
                    # self.labels.append(i)

        self.fmri_subs = [x for x in os.listdir(fmri_dir)] 
        self.fmri_dir = fmri_dir  

    def __len__(self):
        return len(self.npy_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist() 
        meg_npy_path = self.npy_files[idx]
        meg_sample = np.load(meg_npy_path)
        label = int(meg_npy_path[-7:-4])
        subj = meg_npy_path.split(os.sep)[-2]

        rand_fmri_sub = str(np.random.choice(self.fmri_subs, 1, replace=False).tolist()[0]) 
        fmri_sample = np.load(os.path.join(self.fmri_dir, rand_fmri_sub, (rand_fmri_sub+meg_npy_path[-12:])))  
        return meg_sample, fmri_sample, label, subj

 
class NumpyMetaDataset(Dataset):
    def __init__(self, meg_dir, fmri_dir, split, n_way):  
        self.meg_dir = meg_dir  
        self.fmri_dir = fmri_dir  
        self.n_way = n_way 
        self.label = [x for x in range(self.n_way)] 
        self.label = Variable(torch.tensor(self.label))  
        self.split = split

        self.test_splits = ['079', '066', '021', '045', '012', '063', '080', '065', '084', '077', 
                            '048', '042', '007', '091', '071', '075', '067', '014', '059', '083']
        
        self.valid_splits = ['054', '088', '011', '015', '047', '002', '082', '046', '034', '035', 
                             '073', '089', '027', '072', '036', '018']
        
        self.train_splits = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', 
                             '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', 
                             '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036', 
                             '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', 
                             '049', '050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', 
                             '061', '062', '063', '064', '065', '066', '067', '068', '069', '070', '071', '072', 
                             '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '084', 
                             '085', '086', '087', '088', '089', '090', '091', '092', '093', '094', '095', '096', 
                             '097', '098', '099', '100', '101', '102', '103', '104', '105', '106']
 
        self.train_splits = list(filter(lambda item: item not in self.test_splits, self.train_splits)) 
        
    def rand_sub(self):
         
        self.meg_subs = [x for x in os.listdir(self.meg_dir)] 
        self.fmri_subs = [x for x in os.listdir(self.fmri_dir)] 
 
        rand_meg_sub = str(np.random.choice(self.meg_subs, 1, replace=False).tolist()[0]) 
        rand_fmri_sub = str(np.random.choice(self.fmri_subs, 1, replace=False).tolist()[0]) 
 
        return rand_meg_sub, rand_fmri_sub

    def rand_time(self):  
        return np.random.choice(self.class_list, self.n_way, replace=False).tolist()

     
    def __getitem__(self, idx):
        rand_meg_sub, rand_fmri_sub = self.rand_sub()  
        n_folders_m = self.n_folders(os.path.join(self.meg_dir, rand_meg_sub)+'/')
        n_folders_f = self.n_folders(os.path.join(self.fmri_dir, rand_fmri_sub)+'/') 
        
        
        if self.split == 'train':
            n_time = min(n_folders_m, n_folders_f, 106) 
            self.n_time_pints_train(n_time = n_time)   
            self.class_list = [item for item in self.class_list if item not in self.test_splits]
            self.class_list = [item for item in self.class_list if item not in self.valid_splits] 
             
        elif self.split == 'test':
            self.class_list = self.test_splits

        elif self.split == 'valid':
            self.class_list = self.valid_splits

        rand_time_points = self.rand_time()   
        
        meg_sample = [torch.load(os.path.join(self.meg_dir, rand_meg_sub, (rand_meg_sub+'-min-'+rand_t+'.pt')))  
                      for rand_t in rand_time_points]  
        
        fmri_sample = [torch.load(os.path.join(self.fmri_dir, rand_fmri_sub, (rand_fmri_sub+'-min-'+rand_t+'.pt'))) 
                    for rand_t in rand_time_points]  
        
        y_batch = list(map(int, rand_time_points))
        y_meta = self.label  
          
        meg_sample = [torch.nan_to_num(meg_sample_i, nan=2.0, posinf=0.0) for meg_sample_i in meg_sample]
        fmri_sample = [torch.nan_to_num(fmri_sample_i, nan=2.0, posinf=0.0) for fmri_sample_i in fmri_sample]
        
        return (meg_sample, fmri_sample, y_meta, y_batch), [rand_time_points, rand_fmri_sub, rand_meg_sub] 

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



class NumpyBatchDataset(Dataset):
    def __init__(self, meg_dir, fmri_dir, split, n_way, single_subj=False, single_subj_list=None):  
        self.meg_dir = meg_dir  
        self.fmri_dir = fmri_dir  
        self.n_way = n_way 
        self.label = [x for x in range(self.n_way)] 
        self.label = Variable(torch.tensor(self.label))  
        self.split = split
        self.single_subj = single_subj
        
        self.meg_subs = [x for x in os.listdir(self.meg_dir)] 
        self.fmri_subs = [x for x in os.listdir(self.fmri_dir)]
         
        if self.single_subj and (single_subj_list == None):
            self.meg_subs = [str(np.random.choice(self.meg_subs, 1, replace=False).tolist()[0])]
            self.fmri_subs = [str(np.random.choice(self.fmri_subs, 1, replace=False).tolist()[0])]
        elif self.single_subj:
            self.meg_subs = single_subj_list[0]
            self.fmri_subs = single_subj_list[1]
            
        
        self.test_splits = ['079', '066', '021', '045', '012', '063', '080', '065', '084', '077', 
                            '048', '042', '007', '091', '071', '075', '067', '014', '059', '083']
        
        self.valid_splits = ['054', '088', '011', '015', '047', '002', '082', '046', '034', '035', 
                             '073', '089', '027', '072', '036', '018']
        
        self.train_splits = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', 
                             '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', 
                             '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036', 
                             '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', 
                             '049', '050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', 
                             '061', '062', '063', '064', '065', '066', '067', '068', '069', '070', '071', '072', 
                             '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '084', 
                             '085', '086', '087', '088', '089', '090', '091', '092', '093', '094', '095', '096', 
                             '097', '098', '099', '100', '101', '102', '103', '104', '105', '106']
 
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
        
            n_folders_m = self.n_folders(os.path.join(self.meg_dir, rand_meg_sub_w)+'/')
            n_folders_f = self.n_folders(os.path.join(self.fmri_dir, rand_fmri_sub_w)+'/') 
            
            if self.split == 'train':
                n_time = min(n_folders_m, n_folders_f, 106) 
                self.n_time_pints_train(n_time = n_time)   
                self.class_list = [item for item in self.class_list if item not in self.test_splits]
                self.class_list = [item for item in self.class_list if item not in self.valid_splits] 
                
            elif self.split == 'test':
                self.class_list = self.test_splits

            elif self.split == 'valid':
                self.class_list = self.valid_splits

            rand_w = self.rand_time()[0]  
            
            x_meg_w = torch.load(os.path.join(self.meg_dir, rand_meg_sub_w, (rand_meg_sub_w+'-min-'+rand_w+'.pt'))).squeeze(2).unsqueeze(0) 
            x_fmri_w = torch.load(os.path.join(self.fmri_dir, rand_fmri_sub_w, (rand_fmri_sub_w+'-min-'+rand_w+'.pt'))).squeeze(2).unsqueeze(0) 
            
            x_meg_w = torch.nan_to_num(x_meg_w, nan=2.0, posinf=0.0)
            x_fmri_w = torch.nan_to_num(x_fmri_w, nan=2.0, posinf=0.0)
            
            x_meg_w = rearrange(x_meg_w, 'b t p d -> b p (d t)') 
            x_fmri_w = rearrange(x_fmri_w, 'b d p -> b p d')
            
            x_meg_w = self.normalize_tensor(input_tensor=x_meg_w, desired_mean=0.00498728035017848, desired_std=0.7188116312026978)
            x_fmri_w = self.normalize_tensor(input_tensor=x_fmri_w, desired_mean=0.00498728035017848, desired_std=0.7188116312026978)
              
            y_batch.append(list(map(int, rand_w)))
            y_meta.append(self.label)  
            rand_time_points.append(rand_w)
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