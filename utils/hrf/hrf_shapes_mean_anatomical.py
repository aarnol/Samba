from brainiak.utils.fmrisim import _double_gamma_hrf as hrf_func
import matplotlib.pyplot as plt
from matplotlib import cm 
import pandas as pd
import numpy as np
import joypy
import os  
import re
import matplotlib.image as mpimg 
import matplotlib.patheffects as pe
from io import BytesIO
import random
# from utils.hrf.utils import get_main_anatomical_region, anatomical_parcel_name, anatomical_parcel_names

def get_main_anatomical_region(specific_region):
    # Mapping from specific to main anatomical regions
    region_mapping = {
        'Frontal': ['Medial Prefrontal Cortex', 'Dorsal Prefrontal Cortex', 'Lateral Prefrontal Cortex',
                    'Ventral Prefrontal Cortex', 'Frontal Pole', 'Frontal Medial Cortex', 'Orbitofrontal Cortex'],
        'Temporal': ['Temporal Cortex', 'Temporal Pole', 'Superior Temporal', 'Temporal Occipital', 'Parahippocampal Cortex'],
        'Parietal': ['Inferior Parietal Lobule', 'Superior Parietal Lobule', 'Intraparietal Sulcus', 'Precuneus',
                     'Postcentral Gyrus', 'Parietal Operculum'],
        'Occipital': ['Extra-Striate Cortex', 'Extra-Striate Inferior', 'Extra-Striate Superior', 'Striate Cortex'],
        'Cingulate': ['Middle Cingulate Cortex', 'Posterior Cingulate Cortex', 'Retrosplenial Cortex'],
        'Insular': ['Insula'],
        'Somatomotor': ['Somatomotor Cortex', 'Precentral Gyrus'],
        'Unknown': ['Unknown', 'Primary Cortex']  # Added Primary Cortex to 'Unknown' for lack of a better category
    }

    # Iterate through the mapping and find the main anatomical region
    for main_region, regions in region_mapping.items():
        if specific_region in regions:
            return main_region

    # If the specific region is not found, return 'Unknown'
    return 'Unknown'

def anatomical_parcel_name():
    schaefer200_anatomical_regions = [
    'Inferior Parietal Lobule', 'Medial Prefrontal Cortex', 'Temporal Cortex', 'Temporal Cortex',
    'Temporal Pole', 'Temporal Pole', 'Precuneus/Posterior Cingulate Cortex', 'Frontal Pole',
    'Inferior Parietal Lobule', 'Inferior Parietal Lobule', 'Dorsal Prefrontal Cortex', 'Dorsal Prefrontal Cortex',
    'Dorsal Prefrontal Cortex', 'Lateral Prefrontal Cortex', 'Ventral Prefrontal Cortex', 'Temporal Pole',
    'Inferior Parietal Lobule', 'Dorsal Prefrontal Cortex', 'Parahippocampal Cortex', 'Retrosplenial Cortex',
    'Precuneus', 'Inferior Frontal Gyrus', 'Inferior Frontal Gyrus', 'Lateral Prefrontal Cortex', 'Temporal Cortex',
    'Middle Cingulate Cortex', 'Intraparietal Sulcus', 'Intraparietal Sulcus', 'Lateral Prefrontal Cortex',
    'Lateral Prefrontal Cortex', 'Temporal Cortex', 'Inferior Parietal Lobule', 'Intraparietal Sulcus',
    'Orbitofrontal Cortex', 'Orbitofrontal Cortex', 'Posterior Cingulate Cortex', 'Medial Prefrontal Cortex',
    'Temporal Cortex', 'Posterior Cingulate Cortex', 'Frontal Pole', 'Medial Prefrontal Cortex',
    'Medial Prefrontal Cortex', 'Parahippocampal Cortex', 'Precuneus', 'Frontal Medial Cortex', 'Frontal Medial Cortex',
    'Insula', 'Insula', 'Superior Parietal Lobule', 'Inferior Frontal Gyrus', 'Inferior Parietal Lobule', 'Insula',
    'Orbitofrontal Cortex', 'Orbitofrontal Cortex', 'Lateral Prefrontal Cortex', 'Inferior Parietal Lobule',
    'Inferior Parietal Lobule', 'Intraparietal Sulcus', 'Dorsal Prefrontal Cortex', 'Precentral Gyrus', 'Superior Parietal Lobule',
    'Superior Parietal Lobule', 'Temporal Occipital', 'Temporal Occipital', 'Temporal Pole', 'Postcentral Gyrus',
    'Precuneus', 'Superior Parietal Lobule', 'Superior Parietal Lobule', 'Superior Temporal', 'Superior Temporal',
    'Superior Temporal', 'Superior Temporal', 'Parietal Operculum', 'Somatomotor Cortex', 'Somatomotor Cortex',
    'Somatomotor Cortex', 'Somatomotor Cortex', 'Somatomotor Cortex', 'Somatomotor Cortex', 'Insula', 'Somatomotor Cortex',
    'Somatomotor Cortex', 'Somatomotor Cortex', 'Somatomotor Cortex', 'Extra-Striate Cortex', 'Extra-Striate Cortex',
    'Extra-Striate Cortex', 'Extra-Striate Cortex', 'Extra-Striate Cortex', 'Primary Cortex', 'Superior Parietal Lobule',
    'Extra-Striate Inferior', 'Extra-Striate Inferior', 'Extra-Striate Superior', 'Extra-Striate Superior', 'Striate Cortex',
    'Extra-Striate', 'Extra-Striate', 'Striate Cortex', #'Unknown'
    ]
    anatomical_main_regions = ['Frontal', 'Temporal', 'Parietal', 'Occipital', 'Cingulate', 'Insular', 'Somatomotor']#, 'Unknown']
             
    return schaefer200_anatomical_regions,  anatomical_main_regions


def anatomical_parcel_names(hrfs):
    left_hrfs = hrfs[:100]
    right_hrfs = hrfs[100:]
    # equaliance of the anatomical regions of the functaional regions
    schaefer200_anatomical_regions,  anatomical_main_regions = anatomical_parcel_name()
    
    sorted_regions, right_lob_index, parcels_name = [], [], []
    left_hrfs_anatomical_sorted, right_hrfs_anatomical_sorted = [], []
    counter = 0
    for main_region in anatomical_main_regions: 
        right_lob_index.append(counter)
        for i, schaefer_region_i in enumerate(schaefer200_anatomical_regions):
            main_region_hrf_i = get_main_anatomical_region(schaefer_region_i)
            if main_region_hrf_i == main_region: 
                sorted_regions.append(main_region_hrf_i)
                left_hrfs_anatomical_sorted.append(left_hrfs[i])
                right_hrfs_anatomical_sorted.append(right_hrfs[i])
                parcels_name.append(schaefer_region_i)
                counter += 1
        # right_lob_index.append(len(sorted_regions)-1)
    right_lob_index.append(counter)
    left_lob_index = [x+int(len(left_hrfs_anatomical_sorted)) for x in right_lob_index] 
    hrfs = left_hrfs_anatomical_sorted+right_hrfs_anatomical_sorted
    parcels_name = parcels_name+parcels_name

    return parcels_name, anatomical_main_regions, hrfs, right_lob_index, left_lob_index



def remove_substrings(main_string, substrings):
    for substring in substrings:
        main_string = re.sub(re.escape(substring), '', main_string)
    return main_string

def hrf_shape_plot_innner_loop(hrfs, model, label_add=False):
    y_offset=.3 
    fig, axs = plt.subplots(1, 2, figsize=(4, 12))  
    first_ax = True
    parcels_name_functional = model.args.parcels_name   
    parcels_name, anatomical_main_regions, hrfs, right_lob_index, left_lob_index = anatomical_parcel_names(hrfs) 
    # parcels_name = schaefer200_anatomical_regions
    
    label_add=True
    label_conter = 0
    for ax in axs:   
        cut_boundary= 2500 
        
        rh_lob_index = right_lob_index#model.args.rh_lob_index  
        lh_lob_index = left_lob_index
        lh_rh_lob_names = anatomical_main_regions #model.args.lh_rh_lob_names
         
        #   
        if first_ax: 
            lob_index = rh_lob_index
            first_ax = False
            ax.annotate('Left Hemisphere', xy=(0.5, .03), xycoords='axes fraction', ha='center', va='center',
                        bbox=dict(facecolor='orange', edgecolor='none', alpha=0.5)) 
            lower = 0
            num_hrfs= int((len(hrfs)-1)/2) 
            # lower = 0
            # num_hrfs=100
            parcel_counter = 0 
            parcel_annot_position = 0.05
            parcel_annot_steps = (0.92 - parcel_annot_position)/(num_hrfs - lower-1)  
            fontsize = 5
            ax.annotate(lh_rh_lob_names[0], xy=(0.99, 0.13), xycoords='axes fraction', rotation='vertical', ha='left', va='center', fontsize=fontsize+10) #, bbox=annotation_bg)
            ax.annotate(lh_rh_lob_names[1], xy=(0.99, 0.28), xycoords='axes fraction', rotation='vertical', ha='left', va='center', fontsize=fontsize+10) #, bbox=annotation_bg)
            ax.annotate(lh_rh_lob_names[2], xy=(0.99, 0.50), xycoords='axes fraction', rotation='vertical', ha='left', va='center', fontsize=fontsize+10) #, bbox=annotation_bg)
            ax.annotate(lh_rh_lob_names[3], xy=(0.99, 0.67), xycoords='axes fraction', rotation='vertical', ha='left', va='center', fontsize=fontsize+10) #, bbox=annotation_bg)
            ax.annotate(lh_rh_lob_names[4], xy=(0.99, 0.75), xycoords='axes fraction', rotation='vertical', ha='left', va='center', fontsize=fontsize+7) # , bbox=annotation_bg)
            ax.annotate(lh_rh_lob_names[5], xy=(1.1, 0.80), xycoords='axes fraction', rotation='vertical', ha='left', va='center', fontsize=fontsize+7) # , bbox=annotation_bg)
            ax.annotate(lh_rh_lob_names[6], xy=(0.99, 0.87), xycoords='axes fraction', rotation='vertical', ha='left', va='center', fontsize=fontsize+7) #, bbox=annotation_bg) 
        else:  
            lob_index = lh_lob_index  
            ax.annotate('Right Hemisphere', xy=(0.5, .03), xycoords='axes fraction', ha='center', va='center',
                         bbox=dict(facecolor='orange', edgecolor='none', alpha=0.5)) 
            lower = int((len(hrfs)-1)/2)
            num_hrfs=len(hrfs)   
            # lower = 100
            # num_hrfs=200    
            parcel_counter = lower 
            parcel_annot_position = 0.06
            parcel_annot_steps = (0.915 - parcel_annot_position)/(num_hrfs - lower-1)
            
            fontsize = 10 
            # ax.annotate(lh_rh_lob_names[0], xy=(0.99, 0.13), xycoords='axes fraction', rotation='vertical', ha='left', va='center', fontsize=fontsize+10) #, bbox=annotation_bg)
            # ax.annotate(lh_rh_lob_names[1], xy=(0.99, 0.28), xycoords='axes fraction', rotation='vertical', ha='left', va='center', fontsize=fontsize+10) #, bbox=annotation_bg)
            # ax.annotate(lh_rh_lob_names[2], xy=(0.99, 0.50), xycoords='axes fraction', rotation='vertical', ha='left', va='center', fontsize=fontsize+10) #, bbox=annotation_bg)
            # ax.annotate(lh_rh_lob_names[3], xy=(0.99, 0.67), xycoords='axes fraction', rotation='vertical', ha='left', va='center', fontsize=fontsize+10) #, bbox=annotation_bg)
            # ax.annotate(lh_rh_lob_names[4], xy=(0.99, 0.75), xycoords='axes fraction', rotation='vertical', ha='left', va='center', fontsize=fontsize+5) # , bbox=annotation_bg)
            # ax.annotate(lh_rh_lob_names[5], xy=(0.05, 0.80), xycoords='axes fraction', rotation='vertical', ha='left', va='center', fontsize=fontsize+5) # , bbox=annotation_bg)
            # ax.annotate(lh_rh_lob_names[6], xy=(0.99, 0.88), xycoords='axes fraction', rotation='vertical', ha='left', va='center', fontsize=fontsize+10) #, bbox=annotation_bg) 
            # ax.annotate(lh_rh_lob_names[7], xy=(0.7, 0.85), xycoords='axes fraction', rotation='vertical', ha='left', va='center', fontsize=fontsize) #, bbox=annotation_bg)
            
        cm_list = ['Greens', 'hot', 'Oranges', 'Greys', 'Purples', 'Blues', 'bone', 'Reds']  
        cm_conter = 0    
        alpha=0.3 
        initial = False
        white_space_betwen_initial_hrf = 0 
        # import pdb;pdb.set_trace()
        # for i in range(lower, num_hrfs+white_space_betwen_initial_hrf+1):   
            # if i == (lob_index[cm_conter]+white_space_betwen_initial_hrf+1): 
        for i in range(lower, num_hrfs):     
            if i == lob_index[cm_conter]:  
                # print(parcel_counter) 
                colormap = cm.get_cmap(cm_list[cm_conter], lob_index[cm_conter+1]-lob_index[cm_conter]) 
                if cm_conter==0:
                    alpha=0.5
                    color_conter = lob_index[cm_conter+1]-lob_index[cm_conter]+0
                if cm_conter==5:
                    alpha=0.7 
                    color_conter = 0  
                else:
                    color_conter = lob_index[cm_conter+1]-lob_index[cm_conter]+0
                    alpha=0.2 
                cm_conter += 1
                 
            vertical_shift = i * y_offset 
            
            if i>(lower+10):  
                hrf = hrfs[parcel_counter] #.forward()[:cut_boundary].detach().cpu().numpy() 
                if parcel_counter == 100:
                    label='Unknown'
                else:
                    # label=str(parcel_counter)+'   '+parcels_name[parcel_counter]
                    
                    label = parcels_name[parcel_counter] 
                    # label = remove_substrings(label, lh_rh_lob_names)
                    # label = remove_substrings(label, ['_', 'RH', 'LH', 'Language']) 
                # import pdb;pdb.set_trace()     
                if label_add:
                    label_conter +=1
                    ax.plot(range(0, len(hrf)), np.array(hrf) + vertical_shift, color=colormap(color_conter), linewidth=0.35, label=label+'-'+str(label_conter)) 
                else:
                    ax.plot(range(0, len(hrf)), np.array(hrf) + vertical_shift, color=colormap(color_conter), linewidth=0.35)
                
                
                ax.plot(range(0, len(hrf)), np.array(hrf) + vertical_shift, color='black', linewidth=0.20)
                handles, labels = ax.get_legend_handles_labels() 
                # Reverse the order
                handles = handles[::-1]
                labels = labels[::-1]
                # legend = ax.legend(handles, labels,
                #     loc='upper right', bbox_to_anchor=(1.0, -0.074, 0.2, 0.985),  fontsize=5.408
                #     )  
                legend = ax.legend(handles, labels,
                    loc='upper right', bbox_to_anchor=(1.05, -0.99, 0.2, 1),  fontsize=6
                    )  
                parcel_annot_position += parcel_annot_steps  
                parcel_counter += 1
                if cm_conter-1 == 1:
                    ax.fill_between(range(0, len(hrf)), np.array(hrf) + vertical_shift, vertical_shift, color='yellow', alpha=0.2)
                else:
                    ax.fill_between(range(0, len(hrf)), np.array(hrf) + vertical_shift, vertical_shift, color=colormap(color_conter), alpha=alpha)
                # color_conter +=1 
                if cm_conter==6:
                    color_conter +=1 
                else: 
                    color_conter -=1 
                 
            if i==lower and initial:
                hrf = hrf_func(temporal_resolution=model.temporal_resolution)[:cut_boundary]
                ax.plot(range(0, len(hrf)), np.array(hrf) + vertical_shift, color='black')
                ax.fill_between(range(0, len(hrf)), np.array(hrf) + vertical_shift, vertical_shift, color='black', alpha=0.3)
 
            if i<(lower+white_space_betwen_initial_hrf) and i != lower:
                hrf = cut_boundary*[0]
  
        ax.axis('off')   
  
  
    # fig.suptitle('Subject-'+str(ym_batch[-2:])+' (inferred HRFs)', fontsize = 12,
    fig.suptitle(' Mean of all subjects (inferred HRFs)', fontsize = 12,
                 bbox=dict(facecolor='yellow', edgecolor='none', alpha=0.7))
    plt.axis('off')  # Turn off the axis.
    plt.tight_layout(pad=0)  
    
    plt.savefig(model.save_path+'/group_shape_plot_mean.png', dpi=500,  pad_inches=0) 
    # return fig
    
  
    
def group_hrf_shapes_mean(model):
    
    hrf_vectors = []
    cut_boundary= 2500
    for hrf_subj in model.hrfs:
        hrf_vectors_subj = []
        for hrf_prcel in hrf_subj:
            hrf_vectors_subj.append(hrf_prcel.forward()[:cut_boundary].detach().cpu().numpy())
        hrf_vectors.append(hrf_vectors_subj)
    
    hrf_mean_vectors = hrf_vectors[0]
    for i in range(1, len(hrf_vectors)):
        for p in range(len(hrf_mean_vectors)): 
            hrf_mean_vectors[p] = hrf_mean_vectors[p]/2 + hrf_vectors[i][p]/2
           
    hrf_shape_plot_innner_loop(hrf_mean_vectors, model) 