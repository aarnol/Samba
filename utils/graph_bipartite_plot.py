import numpy as np
from mne_connectivity.viz import plot_connectivity_circle
import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from data.schaeferparcel_kong2022_17network import SchaeferParcel_Kong2022_17Network
import torch 
import heapq
import json
import os
from utils.utils import schaefer_to_anatomical, get_main_anatomical_region
from matplotlib.collections import LineCollection
from utils.utils import get_color_palette, get_region_labels

def mkdir_fun(path):
    if not os.path.exists(path): 
        os.mkdir(path)
        
mkdir_fun('../outputs/figures/group/attention/bipartite/')
 


def find_substring(main_string, substrings):
    """
    Searches for the first occurrence of any substring from a list in a given string.

    Args:
        main_string (str): The string in which to search.
        substrings (list): A list of substrings to search for.

    Returns:
        str: The first substring found in the main string; returns None if no substring is found.
    """
    for substring in substrings:
        if main_string.find(substring) != -1:
            return substring
    return None


def generate_nodes(parcels_names, lh_rh_lob_names):
    """
    Generates a list of nodes where each node is a dictionary containing the parcel name and its group index 

    Returns:
        list: A list of dictionaries, each containing the name of the parcel and its group index.
    """
    data_nodes = []
    for parcels_name in parcels_names:
        for group_idx, parcel_group in enumerate(lh_rh_lob_names):
            if parcel_group in parcels_name:
                data_nodes.append({"name": parcels_name, "group": group_idx})
    return data_nodes

 
 


# def rio_bipartite_plot(n_regions, roi_index, data_links, regions_names, circle_size=2, row_distance=5, column_margin=1.5, figure_size=(20, 10), brain_sides=None):
def rio_bipartite_plot(n_regions_show, roi_index, data_links, regions_names, circle_size=2, row_distance=5, column_margin=1.5, figure_size=(20, 10), brain_sides=None):
    fig, ax = plt.subplots(figsize=figure_size)
    ax.set_aspect('equal')
    ax.axis('off')
    region_labels = get_region_labels() 
    roi = region_labels[roi_index] 
    n_regions = len(region_labels)
    
    colors = get_color_palette()
    down_row = [(i * (1 + column_margin), 0) for i in range(n_regions)]
    upper_row = [(i * (1 + column_margin), row_distance) for i in range(n_regions)]

    # Draw lines from data_links
    source_list, target_list = [], []
    alpha_list, value_list = [], []
    line_color_list = []
    for link in data_links:
        source = region_labels.index(regions_names[link['source']])
        target = region_labels.index(regions_names[link['target']]) 
        if regions_names[link['source']] == roi:
            line_color = 'blue'
            alpha = 0.8
            value = link['value']  
        else:
            line_color = 'gray'
            alpha = 0.005
            value = link['value']
            
        source_list.append(source) 
        target_list.append(target) 
        alpha_list.append(alpha)  
        value_list.append(value) 
        line_color_list.append(line_color) 

    linewights = np.zeros((n_regions, n_regions))+1 
    for i in range(len(source_list)): 
        linewights[source_list[i], target_list[i]] += 0.2
        
        if source_list[i] != target_list[i]:
            source_i = source_list[i]
            target_i = target_list[i]  
            if source_i<n_regions_show: 
                ax.plot([down_row[source_i][0], upper_row[target_i][0]], [down_row[source_i][1], upper_row[target_i][1]],
                        color=line_color_list[i], alpha=alpha_list[i], linewidth=linewights[source_i, target_i])
             
 
    # linewights = (linewights - linewights.min()) / (linewights.max() - linewights.min()) 
    # Draw circles and labels
    
    # down row
    for index, (x, y) in enumerate(down_row):  
        if index == n_regions_show: break
        if region_labels[index] == roi:
            cricle_color = colors[index]   # red
            circle_alpha = 0.99
            text_color = 'black'
            text_yes = True
        else:
            cricle_color = "lightgray"
            circle_alpha = 0.2
            text_color = 'gray'
            text_yes = False
              
        circle_size_i = circle_size
        circle = plt.Circle((x, y), circle_size_i, fill=True, color=cricle_color, edgecolor='gray', zorder=2, alpha=circle_alpha)
        ax.add_patch(circle) 
        if text_yes:
            ax.text(x, y - circle_size_i-2.6, region_labels[index], ha='center', zorder=3, fontsize=27,  color=text_color) 

    # upper row
    for index, (x, y) in enumerate(upper_row): 
        if index == n_regions_show: break
        if index == roi_index: 
            # circle_size_i =linewights[index, :].sum()+circle_size 
            circle_size_i = circle_size 
            cricle_color =  'lightgray'  #colors[index]  
            circle_alpha = 0.99
            text_color = 'black'
            text_yes = True
        if (linewights[roi_index, index] == 0) :
            cricle_color = 'lightgray'   
            circle_alpha = 0.2
            text_color = 'gray'
            text_yes = False
        else:
            cricle_color = colors[index]
            circle_alpha = 0.99
            text_color = 'black'
            text_yes = True
            # circle_size_i = linewights[roi_index, index]*6+circle_size
            circle_size_i = circle_size 
        circle = plt.Circle((x, y), circle_size_i, fill=True, color=cricle_color, edgecolor='gray', zorder=2, alpha=circle_alpha)
        ax.add_patch(circle)
        if text_yes:
            ax.text(x+circle_size_i-4, y + circle_size_i , region_labels[index], ha='center', zorder=3, fontsize=27,  color=text_color, rotation=45)

    if n_regions_show< 7:
        save_path_i = './xoutputs/bipartite/main_paper/'+brain_sides+'/'
    else:
        save_path_i = './xoutputs/bipartite/supp_mat/'+brain_sides+'/'        
    mkdir_fun(save_path_i)
    fig.savefig(save_path_i+str(roi_index+1)+'_'+roi+'_'+brain_sides+'.png', 
            format='png', 
            dpi=400, 
            bbox_inches='tight')

    
    




  
def bipartite_plot(h_att, model, brain_sides, k=1, n_regions_show = 5):
    
    colors = get_color_palette()  
    lh_rh_lob_names = get_region_labels()  
     
    schaefer_anatomical_parcels_names, anatomical_main_regions  = schaefer_to_anatomical( h_att.shape[0])  
    valid_indices = [i for i, parcel in enumerate(schaefer_anatomical_parcels_names) if parcel is not None] 
    h_att = h_att[np.ix_(valid_indices, valid_indices)]
    parcels_names = [name for name in schaefer_anatomical_parcels_names if name is not None]
    
    n_nodes = h_att.shape[0]
    n_nodes_half = int(n_nodes/2)
       
    if brain_sides == "left_heme": 
        parcels_names = parcels_names[:n_nodes_half]
        h_att = h_att[:n_nodes_half, :n_nodes_half]
    elif brain_sides == "right_heme": 
        parcels_names = parcels_names[n_nodes_half+1:]            # drop the unkown parcel   
        h_att = h_att[n_nodes_half:, n_nodes_half:] 
    else:
        raise ValueError('the brain side should be right_heme or left_heme') 
    
    
    sorted_indices = np.argsort([x if x is not None else '' for x in parcels_names]) 
    parcels_names = [parcels_names[i] for i in sorted_indices] 
    h_att = h_att[np.ix_(sorted_indices, sorted_indices)]  
    
    node_color = []
    for i, parcel_str in enumerate(parcels_names):
        for j, region_string in enumerate(lh_rh_lob_names):  
            if region_string in parcel_str:
                break 
        node_color.append(colors[j])  
        
    data_links = [] 
    data_nodes = generate_nodes(parcels_names, lh_rh_lob_names)
    h_att = (h_att - np.min(h_att)) / (np.max(h_att) - np.min(h_att)) 
 
    parcels_names_short, regions_names = [], []
    for i, parcels_name in enumerate(parcels_names):  
        region_name_i = find_substring(parcels_name, lh_rh_lob_names)
        regions_names.append(region_name_i) 
        parcels_names_short.append(parcels_name)
                
        h_att_i = h_att[i, :]   
        # mask = np.array(parcels_names) == parcels_name 
        # h_att_i[mask] = 0
        
        top_k_elements_with_indices = heapq.nlargest(k, enumerate(h_att_i), key=lambda x: x[1])  
        for j, item in enumerate(top_k_elements_with_indices):
            index, att_value = item[0], item[1]
            if j == 0:
                data_links.append({"source": i,"target": index, "value": 1}) 
            if att_value > h_att_i.mean()*1.6:
                data_links.append({"source": i,"target": index, "value": 1}) 
    
    
    connect_matrix_source = np.zeros((h_att.shape[0], h_att.shape[1])) 
    connect_matrix_target = np.zeros((h_att.shape[0], h_att.shape[1]))  
    for node_index in range(h_att.shape[0]): 
        for data in data_links:
            if data['source']==node_index:
                connect_matrix_source[node_index, data['target']] = data['value'] 
            if data['target']==node_index:
                connect_matrix_target[node_index, data['target']] = data['value'] 
 
    for n in range(n_regions_show):
        rio_bipartite_plot(n_regions_show, roi_index=n, data_links=data_links,
                                regions_names=regions_names,
                            circle_size=6, row_distance=40,
                            column_margin=10, figure_size=(30, 10),
                            brain_sides = brain_sides)





 