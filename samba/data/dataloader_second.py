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
         
        self.test_splits = ['002', '004', '057', '088', '089', '039', '042', '131', '133', '126', 
                            '126', '137', '178', '199', '194', '192', '182', '174', '161', '180']
        
        self.valid_splits = ['184', '149', '100', '090', '131', '095', '020']
        
        self.train_splits = [
            '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', 
            '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', 
            '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', 
            '031', '032', '033', '034', '035', '036', '037', '038', '039', '040', 
            '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', 
            '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', 
            '061', '062', '063', '064', '065', '066', '067', '068', '069', '070', 
            '071', '072', '073', '074', '075', '076', '077', '078', '079', '080', 
            '081', '082', '083', '084', '085', '086', '087', '088', '089', '090', 
            '091', '092', '093', '094', '095', '096', '097', '098', '099', '100', 
            '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', 
            '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', 
            '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', 
            '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', 
            '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', 
            '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', 
            '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', 
            '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', 
            '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', 
            '191', '192', '193', '194', '195', '196', '197', '198', '199', '200']
   
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
                
            elif self.split == 'test':
                self.class_list = self.test_splits

            elif self.split == 'valid':
                self.class_list = self.valid_splits

            rand_w = self.rand_time()[0]   
            x_meg_w = torch.load(os.path.join(self.meg_dir, 'sub-'+ rand_meg_sub_w, ('sub-'+ rand_meg_sub_w+'-sec-'+rand_w+'.pt')), weights_only=False).squeeze(2)#.unsqueeze(0) 
            x_fmri_w = torch.load(os.path.join(self.fmri_dir, 'sub-'+ rand_fmri_sub_w, ('sub-'+ rand_fmri_sub_w+'-sec-'+rand_w+'.pt')), weights_only=False).squeeze(2)#.unsqueeze(0) 
            
            x_meg_w = torch.nan_to_num(x_meg_w, nan=2.0, posinf=0.0)
            x_fmri_w = torch.nan_to_num(x_fmri_w, nan=2.0, posinf=0.0)
             
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
    
    

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.autograd import Variable
from einops import rearrange
import nilearn
def load_fmri_data(fmri_folder):
    '''Load fMRI data from a specified folder. All files are nii.gz format.'''
    fmri_data = []
    for file in os.listdir(fmri_folder):
        if file.endswith('.nii.gz'):
            try:
                # Load the fMRI image using nilearn
                img = nilearn.image.load_img(os.path.join(fmri_folder, file)).get_fdata()

            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
            subject_id = file[:6]  # Assuming subject ID is the first 6 characters of the filename
            encoding = file[6] # Assuming encoding is the 7th character of the filename
            condition = file[7]
            label = file[8]
            ex = {
                "pheno": {
                    "subjectId": subject_id,
                    "encoding": encoding,
                    "condition": condition,
                    "label": label
                },
                "img": img
            }
            fmri_data.append(ex)
    if not fmri_data:
        raise ValueError(f"No fMRI data found in folder: {fmri_folder}", flush=True)
    print(len(fmri_data), "fMRI data loaded from", fmri_folder, flush = True)
    return fmri_data

class NumpyBatchDatasetHCP(Dataset):
    def __init__(self, fnirs_file, fmri_folder, split, n_way, fnirs_sub_list, fmri_sub_list, single_subj=False):
        self.fnirs_data = torch.load(fnirs_file, weights_only=False)  # list of dicts
        self.fmri_data = load_fmri_data(fmri_folder)   # list of dicts
        self.n_way = n_way
        self.label = Variable(torch.tensor([x for x in range(self.n_way)]))
        self.split = split
        self.single_subj = single_subj #doesnt do anything lol

        self.fnirs_subs = fnirs_sub_list
        self.fmri_subs = fmri_sub_list

        # Define subject splits (if needed)
        self.test_splits = ['002', '004', '057', '088', '089', '039']
        self.valid_splits = ['184', '149', '100']
        self.train_splits = list(filter(lambda s: s not in self.test_splits, [f'{i:03}' for i in range(1, 201)]))

    def normalize_tensor(self, input_tensor, desired_mean=0.0, desired_std=1.0):
        current_mean = input_tensor.mean()
        current_std = input_tensor.std()
        normalized = (input_tensor - current_mean) / current_std
        return normalized * desired_std + desired_mean

    def get_data_by_subject(self, data_list, subject_id):
        
        return [entry for entry in data_list if entry["pheno"]["subjectId"] == subject_id]

    def __getitem__(self, idx):
        x_fnirs, x_fmri, y_batch, y_meta, time_points, sub_f_list, sub_n_list = [], [], [], [], [], [], []

        for w in range(self.n_way):
            fnirs_sub = np.random.choice(self.fnirs_subs)
            fmri_sub = np.random.choice(self.fmri_subs)
            print(f"Processing subject {fnirs_sub} for fNIRS and {fmri_sub} for fMRI", flush=True)
            fnirs_entries = self.get_data_by_subject(self.fnirs_data, fnirs_sub)
            fmri_entries = self.get_data_by_subject(self.fmri_data, fmri_sub)

            # Randomly pick one timepoint
            rand_idx = np.random.randint(min(len(fnirs_entries), len(fmri_entries)))
            fnirs_ts = torch.tensor(fnirs_entries[rand_idx]['roiTimeseries']).float()
            fmri_ts = torch.tensor(fmri_entries[rand_idx]['img']).float()

            fnirs_ts = self.normalize_tensor(fnirs_ts)
            fmri_ts = self.normalize_tensor(fmri_ts)

            x_fnirs.append(fnirs_ts)
            x_fmri.append(fmri_ts)
            y_batch.append(rand_idx)
            y_meta.append(self.label)
            time_points.append(str(rand_idx))
            sub_f_list.append(fmri_sub)
            sub_n_list.append(fnirs_sub)

        x_fnirs = torch.stack(x_fnirs).squeeze()
        x_fmri = torch.stack(x_fmri).squeeze()

        return (x_fnirs, x_fmri, y_meta, y_batch), [time_points, sub_f_list, sub_n_list]

    def __len__(self):
        return 1
