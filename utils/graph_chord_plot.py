 

from data_eegfmri.schaeferparcel_kong2022_17network import SchaeferParcel_Kong2022_17Network
from model.model_registry import str2model 
from scipy.signal import savgol_filter 
from auxiliary.utils import print_gpu_info
from torch.optim import Adam   
from args import params_fn 
import torch.nn as nn
import numpy as np
import torch
import time 
import os
import matplotlib.path as mpath
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import heapq
import pandas as pd
import holoviews as hv
from holoviews import opts, dim
from bokeh.sampledata.les_mis import data
from bokeh.io import export_png, export_svgs
from auxiliary.visualizations.attention.utils import schaefer_to_anatomical, get_main_anatomical_region
import cairosvg
import io
from PIL import Image
import svgutils.transform as sg

import matplotlib.image as mpimg
from nxviz import CircosPlot 
import networkx as nx 
import holoviews as hv
from holoviews import opts
import numpy as np
hv.extension('bokeh')

from utils.utils import get_color_palette, get_region_labels

# Function to generate distinct colors
def distinct_colors(n):
    return hv.Cycle('Category20b').values if n <= 20 else [hv.util.colors.Color('#' + ''.join([np.random.choice(list('0123456789ABCDEF')) for j in range(6)])) for i in range(n)]

 
 
def remove_substrings(main_string, substrings):
    for substring in substrings:
        main_string = re.sub(re.escape(substring), '', main_string)
    return main_string 

def list_folders(directory): 
    return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
       

def node_fn(parcels_names, lh_rh_lob_names):
    data_node = []
    for parcels_name in parcels_names:
        for group_idx, parcel_group in enumerate(lh_rh_lob_names):
            if parcel_group in parcels_name: 
                parcels_name = remove_substrings(parcels_name, lh_rh_lob_names)
                parcels_name = remove_substrings(parcels_name, ['_', 'RH', 'LH'])  
                data_node.append({"name": parcels_name,"group": group_idx})
    return data_node

 


def attention_parcel_viz(model, h_att, save_name, brain_sides="left_heme", k = 3): 
    hv.extension('bokeh') 
    hv.output(size=300) # 650
     
    # remove the self-attentions
    np.fill_diagonal(h_att, 0)
    
     
    colors = get_color_palette() 
    model.args.lh_rh_lob_names = get_region_labels() 
    lh_rh_lob_names = model.args.lh_rh_lob_names 
     
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
        for j, region_string in enumerate(model.args.lh_rh_lob_names): 
            if region_string in parcel_str:
                break 
        node_color.append(colors[j])  
        
    data_links = [] 
    data_nodes = node_fn(parcels_names, lh_rh_lob_names)
    h_att = (h_att - np.min(h_att)) / (np.max(h_att) - np.min(h_att)) 
    
    k=1
    for i, parcels_name in enumerate(parcels_names):  
        h_att_zero_self =h_att 
        h_att_i = h_att_zero_self[i, :]  
        mask = np.array(parcels_names) == parcels_name 
        h_att_i[mask] = 0
                
        # valid_indices = [i for i, parcel in enumerate(parcels_names) if parcel != parcels_name] 
        
        # import pdb;pdb.set_trace()
        top_k_elements_with_indices = heapq.nlargest(k, enumerate(h_att_i), key=lambda x: x[1])   
        for j, item in enumerate(top_k_elements_with_indices):
            # import pdb;pdb.set_trace()
            index, att_value = item[0], item[1]
            if j == 0:
                data_links.append({"source": i,"target": index, "value": 1}) 
            if att_value > h_att_i.mean()*1.6:
                data_links.append({"source": i,"target": index, "value": 1})  
    
    data_nodes = tuple(data_nodes)
    data_links = tuple(data_links)
     
    data_nodes = pd.DataFrame(data_nodes)
    data_links = pd.DataFrame(data_links)
     
    data_nodes = hv.Dataset(data_nodes, 'index') 
    chord = hv.Chord((data_links, data_nodes)) #.select(value=(4, None))
        
    chord.opts(
        opts.Chord(
            # cmap='TolRainbow', edge_cmap='TolRainbow', 
            cmap=node_color,        # distinct_colors(n_nodes_half),   
            edge_cmap=node_color,  # A good sequential colormap for edges
            edge_color=dim('source').str(), 
            labels='name', node_color=dim('index').str(),
            label_text_font_size='1pt', edge_line_width=1,
            label_text_color='white', 
            node_size=1
            )
    )

    plot = hv.render(chord) 
    export_png(plot, filename=save_name) 
     
    img = Image.open(save_name) 
    width, height = img.size 
    
    left = width * 0.07
    top = height * 0.07
    right = width * 0.93
    bottom = height * 0.93
    img = img.crop((left, top, right, bottom)) 
    
    return img 

def plot_axes(img, axes, title):
    axes.imshow(img)
    axes.set_xlabel('')
    axes.set_ylabel('') 
    axes.set_xticks([])
    axes.set_yticks([]) 
    axes.spines['top'].set_alpha(0.2)
    axes.spines['right'].set_alpha(0.2)
    axes.spines['bottom'].set_alpha(0.2)
    axes.spines['left'].set_alpha(0.2) 
    axes.text(0.01, 0.95, title, transform=axes.transAxes, 
              verticalalignment='top', horizontalalignment='left', fontsize=12)#, color='white')
    colors = get_color_palette()
    dash_marker = make_dash_marker()
    labels = get_region_labels() 
    for label, color in zip(labels, colors):
        axes.scatter([], [], label=label, color=color, marker=dash_marker, s=100, linewidths=2)
    axes.legend(loc='lower center', ncol=4, fontsize='small', frameon=False)

def make_dash_marker(): 
    marker_path = mpath.Path(np.array([
        [-1, 0],
        [1, 0]
    ]))
    # Codes = [mpath.Path.MOVETO, mpath.Path.LINETO]
    # marker_path = mpath.Path(Vertices, Codes)
    return marker_path
    
def regions_plot_axes(img, axes):
    axes.set_xlabel('')
    axes.set_ylabel('') 
    axes.set_xticks([])
    axes.set_yticks([]) 
    axes.spines['top'].set_alpha(0.0)
    axes.spines['right'].set_alpha(0.0)
    axes.spines['bottom'].set_alpha(0.0)
    axes.spines['left'].set_alpha(0.0)
  
    width, height = img.size  
    zoom_percentage = 20
    zoom_factor = zoom_percentage / 100
    left = width * zoom_factor / 2
    top = height * zoom_factor / 2
    right = width - left
    bottom = height - top
    img = img.crop((left, top, right, bottom)) 
    colors = get_color_palette()
      
    labels = get_region_labels() 
    for label, color in zip(labels, colors):
        axes.scatter([], [], label=label, color=color, marker='o')
    axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fontsize='x-small')
     
    axes.imshow(img) 
    
    
def count_nodes_in_group(group_idx, group):
    return sum([1 for idx in group_idx if idx == group])
       
def graph_chord_draw(model, h_att, save_name):
    save_path = './xoutputs/chord/' 
    if not os.path.exists(save_path): 
        os.mkdir(save_path) 
    
    # save_path = './xoutputs/chord/' 
    # if not os.path.exists(save_path): 
    #     os.mkdir(save_path) 
    
    # save_path = './xoutputs/chord/' 
    # if not os.path.exists(save_path): 
    #     os.mkdir(save_path) 
    # save_path = './xoutputs/chord/'    
    save_name = save_path+save_name+'.png'
    
    img_p_lh = attention_parcel_viz(model, h_att, save_name, brain_sides="left_heme")   
    img_p_rh = attention_parcel_viz(model, h_att, save_name, brain_sides="right_heme") 
    # img_r_lh = brain_regions_viz(model, h_att, save_name, brain_sides="left_heme") 
    # img_r_rh = brain_regions_viz(model, h_att, save_name, brain_sides="right_heme") 
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 
    plot_axes(img_p_lh, axes[0], title='Left Hemisphere') 
    plot_axes(img_p_rh, axes[1], title='Right Hemisphere') 
     
    
    plt.tight_layout() 
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig(save_name, dpi=400)
     
    
     