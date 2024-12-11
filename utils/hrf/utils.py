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
    # schaefer200_anatomical_regions = [
    # 'Inferior Parietal Lobule', 'Medial Prefrontal Cortex', 'Temporal Cortex', 'Temporal Cortex',
    # 'Temporal Pole', 'Temporal Pole', 'Precuneus/Posterior Cingulate Cortex', 'Frontal Pole',
    # 'Inferior Parietal Lobule', 'Inferior Parietal Lobule', 'Dorsal Prefrontal Cortex', 'Dorsal Prefrontal Cortex',
    # 'Dorsal Prefrontal Cortex', 'Lateral Prefrontal Cortex', 'Ventral Prefrontal Cortex', 'Temporal Pole',
    # 'Inferior Parietal Lobule', 'Dorsal Prefrontal Cortex', 'Parahippocampal Cortex', 'Retrosplenial Cortex',
    # 'Precuneus', 'Inferior Frontal Gyrus', 'Inferior Frontal Gyrus', 'Lateral Prefrontal Cortex', 'Temporal Cortex',
    # 'Middle Cingulate Cortex', 'Intraparietal Sulcus', 'Intraparietal Sulcus', 'Lateral Prefrontal Cortex',
    # 'Lateral Prefrontal Cortex', 'Temporal Cortex', 'Inferior Parietal Lobule', 'Intraparietal Sulcus',
    # 'Orbitofrontal Cortex', 'Orbitofrontal Cortex', 'Posterior Cingulate Cortex', 'Medial Prefrontal Cortex',
    # 'Temporal Cortex', 'Posterior Cingulate Cortex', 'Frontal Pole', 'Medial Prefrontal Cortex',
    # 'Medial Prefrontal Cortex', 'Parahippocampal Cortex', 'Precuneus', 'Frontal Medial Cortex', 'Frontal Medial Cortex',
    # 'Insula', 'Insula', 'Superior Parietal Lobule', 'Inferior Frontal Gyrus', 'Inferior Parietal Lobule', 'Insula',
    # 'Orbitofrontal Cortex', 'Orbitofrontal Cortex', 'Lateral Prefrontal Cortex', 'Inferior Parietal Lobule',
    # 'Inferior Parietal Lobule', 'Intraparietal Sulcus', 'Dorsal Prefrontal Cortex', 'Precentral Gyrus', 'Superior Parietal Lobule',
    # 'Superior Parietal Lobule', 'Temporal Occipital', 'Temporal Occipital', 'Temporal Pole', 'Postcentral Gyrus',
    # 'Precuneus', 'Superior Parietal Lobule', 'Superior Parietal Lobule', 'Superior Temporal', 'Superior Temporal',
    # 'Superior Temporal', 'Superior Temporal', 'Parietal Operculum', 'Somatomotor Cortex', 'Somatomotor Cortex',
    # 'Somatomotor Cortex', 'Somatomotor Cortex', 'Somatomotor Cortex', 'Somatomotor Cortex', 'Insula', 'Somatomotor Cortex',
    # 'Somatomotor Cortex', 'Somatomotor Cortex', 'Somatomotor Cortex', 'Extra-Striate Cortex', 'Extra-Striate Cortex',
    # 'Extra-Striate Cortex', 'Extra-Striate Cortex', 'Extra-Striate Cortex', 'Primary Cortex', 'Superior Parietal Lobule',
    # 'Extra-Striate Inferior', 'Extra-Striate Inferior', 'Extra-Striate Superior', 'Extra-Striate Superior', 'Striate Cortex',
    # 'Extra-Striate', 'Extra-Striate', 'Striate Cortex', #'Unknown'
    # ]
    schaefer200_anatomical_regions = ['Inferior Parietal Lobule',
    'Medial Prefrontal Cortex',
    'Temporal Cortex',
    'Temporal Cortex 2',
    'Temporal Pole',
    'Temporal Pole 2',
    'Precuneus/Posterior Cingulate Cortex',
    'Frontal Pole',
    'Inferior Parietal Lobule 2',
    'Inferior Parietal Lobule 3',
    'Dorsal Prefrontal Cortex',
    'Dorsal Prefrontal Cortex 2',
    'Dorsal Prefrontal Cortex 3',
    'Lateral Prefrontal Cortex',
    'Ventral Prefrontal Cortex',
    'Temporal Pole 3',
    'Inferior Parietal Lobule 4',
    'Dorsal Prefrontal Cortex 4',
    'Parahippocampal Cortex',
    'Retrosplenial Cortex',
    'Precuneus',
    'Inferior Frontal Gyrus',
    'Inferior Frontal Gyrus 2',
    'Lateral Prefrontal Cortex 2',
    'Temporal Cortex 3',
    'Middle Cingulate Cortex',
    'Intraparietal Sulcus',
    'Intraparietal Sulcus 2',
    'Lateral Prefrontal Cortex 3',
    'Lateral Prefrontal Cortex 4',
    'Temporal Cortex 4',
    'Inferior Parietal Lobule 5',
    'Intraparietal Sulcus 3',
    'Orbitofrontal Cortex',
    'Orbitofrontal Cortex 2',
    'Posterior Cingulate Cortex',
    'Medial Prefrontal Cortex 2',
    'Temporal Cortex 5',
    'Posterior Cingulate Cortex 2',
    'Frontal Pole 2',
    'Medial Prefrontal Cortex 3',
    'Medial Prefrontal Cortex 4',
    'Parahippocampal Cortex 2',
    'Precuneus 2',
    'Frontal Medial Cortex',
    'Frontal Medial Cortex 2',
    'Insula',
    'Insula 2',
    'Superior Parietal Lobule',
    'Inferior Frontal Gyrus 3',
    'Inferior Parietal Lobule 6',
    'Insula 3',
    'Orbitofrontal Cortex 3',
    'Orbitofrontal Cortex 4',
    'Lateral Prefrontal Cortex 5',
    'Inferior Parietal Lobule 7',
    'Inferior Parietal Lobule 8',
    'Intraparietal Sulcus 4',
    'Dorsal Prefrontal Cortex 5',
    'Precentral Gyrus',
    'Superior Parietal Lobule 2',
    'Superior Parietal Lobule 3',
    'Temporal Occipital',
    'Temporal Occipital 2',
    'Temporal Pole 4',
    'Postcentral Gyrus',
    'Precuneus 3',
    'Superior Parietal Lobule 4',
    'Superior Parietal Lobule 5',
    'Superior Temporal',
    'Superior Temporal 2',
    'Superior Temporal 3',
    'Superior Temporal 4',
    'Parietal Operculum',
    'Somatomotor Cortex',
    'Somatomotor Cortex 2',
    'Somatomotor Cortex 3',
    'Somatomotor Cortex 4',
    'Somatomotor Cortex 5',
    'Somatomotor Cortex 6',
    'Insula 4',
    'Somatomotor Cortex 7',
    'Somatomotor Cortex 8',
    'Somatomotor Cortex 9',
    'Somatomotor Cortex 10',
    'Extra-Striate Cortex',
    'Extra-Striate Cortex 2',
    'Extra-Striate Cortex 3',
    'Extra-Striate Cortex 4',
    'Extra-Striate Cortex 5',
    'Primary Cortex',
    'Superior Parietal Lobule 6',
    'Extra-Striate Inferior',
    'Extra-Striate Inferior 2',
    'Extra-Striate Superior',
    'Extra-Striate Superior 2',
    'Striate Cortex',
    'Extra-Striate',
    'Extra-Striate 2',
    'Striate Cortex 2']
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


  