from brainiak.utils.fmrisim import _double_gamma_hrf as hrf_func
from data.schaeferparcel_kong2022_17network import SchaeferParcel_Kong2022_17Network
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

def get_color_palette():
    """
    Returns a list of hex color codes representing different brain regions.

    Returns:
        list: A list of hex color codes.
    """
    return [
        '#3C5B6F',  # Frontal
        '#FFC55A',  # Temporal
        '#378CE7',  # Parietal
        '#A91D3A',  # Occipital
        '#d5bdaf',  # Somatomotor
        '#9AC8CD',  # Cingulate
        '#002379',  # Insular
    ]
    
def get_region_labels():
    """
    Returns a list of labels for different brain regions.

    Returns:
        list: A list of region labels.
    """
    return [
        'Frontal',
        'Temporal',
        'Parietal',
        'Occipital',
        'Somatomotor',
        'Cingulate',
        'Insular', 
    ]

def get_main_anatomical_region(name):
    # Mapping specific labels in the parcellation names to broader anatomical categories
    region_mapping = {
        'FPole': 'Frontal',
        'IPL': 'Parietal',
        'PCC': 'Cingulate',
        'PFCd': 'Frontal',
        'PFCm': 'Frontal',
        'PHC': 'Temporal',
        'Temp': 'Temporal',
        'TempPole': 'Temporal',
        'pCun': 'Parietal',
        'IFG': 'Frontal',
        'Ins': 'Insular',
        'Cingm': 'Cingulate',
        'IPS': 'Parietal',
        'OFC': 'Frontal',
        'PFCl': 'Frontal',
        'RSC': 'Cingulate',
        'FrMed': 'Frontal',
        'FrOper': 'Frontal',
        'SPL': 'Parietal',
        'PrCv': 'Parietal',
        'ParOper': 'Parietal',
        'PostC': 'Parietal',
        'PrCd': 'Parietal',
        'Aud': 'Temporal',
        'SomMot': 'Somatomotor',
        'Visual': 'Occipital',
        'ExStr': 'Occipital',
        'Striate': 'Occipital'
    }
    
    # Default category if not found
    default_category = 'Unknown'

    # Extracting part of the label that corresponds to the anatomical region
    for key in region_mapping:
        if key in name: 
            return region_mapping[key]

    return default_category  # return the default category if no key matches


def get_main_anatomical_region_schaefer(name):
    # Mapping specific labels in the parcellation names to broader anatomical categories
    region_mapping = {
        'FPole': 'Frontal',
        'IPL': 'Parietal',
        'PCC': 'Cingulate',
        'PFCd': 'Frontal',
        'PFCm': 'Frontal',
        'PHC': 'Temporal',
        'Temp': 'Temporal',
        'TempPole': 'Temporal',
        'pCun': 'Parietal',
        'IFG': 'Frontal',
        'Ins': 'Insular',
        'Cingm': 'Cingulate',
        'IPS': 'Parietal',
        'OFC': 'Frontal',
        'PFCl': 'Frontal',
        'RSC': 'Cingulate',
        'FrMed': 'Frontal',
        'FrOper': 'Frontal',
        'SPL': 'Parietal',
        'PrCv': 'Parietal',
        'ParOper': 'Parietal',
        'PostC': 'Parietal',
        'PrCd': 'Parietal',
        'Aud': 'Temporal',
        'SomMot': 'Somatomotor',
        'Visual': 'Occipital',
        'ExStr': 'Occipital',
        'Striate': 'Occipital'
    }
    
    # # Default category if not found
    # default_category = 'Unknown'

    # Extracting part of the label that corresponds to the anatomical region
    for key in region_mapping.keys():
        if key in name.decode('utf-8'):
            return region_mapping[key]
        
    
def schaefer_to_anatomical(n_nodes):   
    labels, parcels_names, ctabs = SchaeferParcel_Kong2022_17Network(parcel_number = n_nodes) 
    parcels_names.pop(0)
    parcels_names.pop(250)
    schaefer200_anatomical_parcels_names = []
    # schaefer200_anatomical_labels= []  
    for parcel_i in parcels_names:
        schaefer200_anatomical_parcels_names.append(get_main_anatomical_region_schaefer(parcel_i))
 
     
    anatomical_main_regions = ['Frontal', 'Temporal', 'Parietal', 'Occipital', 'Cingulate', 'Insular', 'Somatomotor']#, 'Unknown']
    return schaefer200_anatomical_parcels_names, anatomical_main_regions 