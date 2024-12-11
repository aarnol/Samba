import sys
from scipy.spatial.distance import pdist, squareform 
from einops import rearrange

import torch
import os



def mkdir_fun(path):
    if not os.path.exists(path): 
        os.mkdir(path)
 
def calculate_adjacency_matrix(spatiotemporal_tensor):
    """
    Calculates the adjacency matrix based on the correlation of the 'time' dimension of a spatiotemporal tensor
    using 1 - squareform(pdist(..., 'correlation')).

    :param spatiotemporal_tensor: A 2D numpy array of shape [node, time]
    :return: A 2D numpy array representing the adjacency matrix of shape [node, node]
    """ 
    adjacency_matrix = 1 - squareform(pdist(spatiotemporal_tensor, 'correlation'))
    return adjacency_matrix


import glob
import os

def list_pt_files_glob(directory):
    # Create a pattern to match all '.pt' files
    pattern = os.path.join(directory, '*.pt')
    # Use glob to find all matching files
    pt_files = glob.glob(pattern)
    # Extract just the filenames, not the full paths
    pt_files = [os.path.basename(file) for file in pt_files if os.path.isfile(file)]
    return pt_files


import os

def list_folders(directory):
    try:
        # List all entries in the directory
        entries = os.listdir(directory)
        # Filter out entries that are directories
        folders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
        return folders
    except FileNotFoundError:
        print(f"Error: The directory '{directory}' does not exist.")
    except PermissionError:
        print(f"Error: Permission denied for accessing '{directory}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return []



if __name__ == "__main__":   

    eeg_n_parcels = '200'
    fmri_n_parcels = '500'
    eeg_folder = 'eeg-'+eeg_n_parcels+'parcell-avg' 
    fmri_folder = 'fmri-'+fmri_n_parcels+'parcell-avg' 

    data_path = '../dataset/naturalistic_view_eeg_fmri_1min/'
    eeg_data_path = data_path+eeg_folder+'/'
    fmri_data_path = data_path+fmri_folder+'/'
    
    import numpy as np
    for subj in list_folders(eeg_data_path): 
        
        eeg_file_path = eeg_data_path+subj+'/'  
        fmri_file_path = fmri_data_path+subj+'/'  
        x_eeg_list, x_fmri_list = [], []
        for files in list_pt_files_glob(eeg_file_path):  
            x_eeg = torch.load(eeg_file_path+files, weights_only=False)  
            x_fmri = torch.load(fmri_file_path+files, weights_only=False)  
            x_fmri = torch.nan_to_num(x_fmri, nan=2.0, posinf=0.0)
            x_eeg = torch.nan_to_num(x_eeg, nan=2.0, posinf=0.0)

            x_eeg_list.append(x_eeg)
            x_fmri_list.append(x_fmri)

        x_fmri = torch.cat(x_fmri_list)  
        x_eeg = torch.cat(x_eeg_list)  

        x_eeg = rearrange(x_eeg, 'b p t -> p (b t)') 
        x_eeg_graph = calculate_adjacency_matrix(x_eeg) 
        x_eeg_graph = np.nan_to_num(x_eeg_graph)

        eeg_graph_path = '../dataset/graphs_eegfmri/eeg_'+eeg_n_parcels+'_parcels/'
        mkdir_fun(eeg_graph_path) 
        np.save(eeg_graph_path+subj+'.npy', x_eeg_graph)
         

        x_fmri = rearrange(x_fmri, 'b p t -> p (b t)')
        x_fmri_graph = calculate_adjacency_matrix(x_fmri)
        x_fmri_graph = np.nan_to_num(x_fmri_graph)

        fmri_graph_path = '../dataset/graphs_eegfmri/fmri_'+fmri_n_parcels+'_parcels/'
        mkdir_fun(fmri_graph_path) 
        np.save(fmri_graph_path+subj+'.npy', x_fmri_graph)

   
 