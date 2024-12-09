 
import os
import torch
from torch.utils.data import Dataset, DataLoader
import re


def remove_after_run(s):
    """
    Removes everything after the first occurrence of *_run* in a string.
    
    Args:
        s (str): Input string.
    
    Returns:
        str: Modified string with everything after *_run* removed.
    """
    match = re.search(r"_run", s)
    if match:
        return s[:match.start()]
    return s

import os
import torch
from torch.utils.data import Dataset

class PTFileDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Custom dataset that loads .pt files containing dictionary data for classification.
        
        Args:
            dataset_path (str): Path to the dataset folder containing subfolders as classes.
        """
        self.dataset_path = dataset_path
        self.eeg_path = os.path.join(dataset_path, 'eeg-200parcell-avg/')
        self.fmri_path = os.path.join(dataset_path, 'fmri-500parcell-avg/')
        self.samples = []
        self.class_to_idx = {}

        # Map each class folder to an index
        for idx, class_folder in enumerate(sorted(os.listdir(self.eeg_path))):
            eeg_folder_path = os.path.join(self.eeg_path, class_folder)
            fmri_folder_path = os.path.join(self.fmri_path, class_folder)
            if os.path.isdir(eeg_folder_path) and os.path.isdir(fmri_folder_path):
                self.class_to_idx[class_folder] = idx
                for file_name in os.listdir(eeg_folder_path):
                    if file_name.endswith(".pt") and os.path.exists(os.path.join(fmri_folder_path, file_name)):
                        eeg_file_path = os.path.join(eeg_folder_path, file_name)
                        fmri_file_path = os.path.join(fmri_folder_path, file_name)
                        self.samples.append((eeg_file_path, fmri_file_path, idx, class_folder))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Returns the filename, both EEG and fMRI data, class label, and folder name.
        
        Args:
            index (int): Index of the sample in the dataset.

        Returns:
            tuple: (filename, eeg_data, fmri_data, label, folder_name)
        """
        eeg_file, fmri_file, label, folder_name = self.samples[index]
        filename = os.path.basename(eeg_file)

        # Load EEG data
        eeg_data_dict = torch.load(eeg_file)
        x_eeg = eeg_data_dict['x']
        y_eeg = eeg_data_dict['y']
        sub_eeg = eeg_data_dict['sub']

        # Load fMRI data
        fmri_data_dict = torch.load(fmri_file)
        x_fmri = fmri_data_dict['x']
        y_fmri = fmri_data_dict['y']
        sub_fmri = fmri_data_dict['sub']

        # Ensure consistency between EEG and fMRI data
        assert y_eeg == y_fmri and sub_eeg == sub_fmri, f"Inconsistent labels/sub IDs between EEG and fMRI for file {filename}"

        # Optionally, you can merge or process the data as needed
        return filename, {'eeg': x_eeg, 'fmri': x_fmri}, label, folder_name




dataset_path = "/home/aa2793/gibbs/arman/datasets/NEUROSCIENCE/EEGfMRI/naturalistic_viewing/minute_dataset_classification/"  # Replace with your dataset path
dataset = PTFileDataset(dataset_path)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

for batch in data_loader:
    filenames, data, labels, folders = batch
    eeg_data = data['eeg']
    fmri_data = data['fmri']

 