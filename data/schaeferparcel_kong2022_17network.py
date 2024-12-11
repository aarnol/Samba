import nibabel as nib
import numpy as np


def sort_names_by_labels(names, labels):
    # Pair each name with its corresponding label and sort the pairs
    sorted_pairs = sorted(zip(labels, names))

    # Separate the sorted pairs back into two lists
    sorted_labels, sorted_names = zip(*sorted_pairs)

    return list(sorted_names), list(sorted_labels)

def SchaeferParcel_Kong2022_17Network(source_path, parcel_number = 200):
 
    lh_file_path = source_path+'lh.Schaefer2018_'+str(parcel_number)+'Parcels_Kong2022_17Networks_order.annot'
    rh_file_path = source_path+'rh.Schaefer2018_'+str(parcel_number)+'Parcels_Kong2022_17Networks_order.annot'

    lh_labels, lh_ctab, lh_names = nib.freesurfer.io.read_annot(lh_file_path)
    rh_labels, rh_ctab, rh_names = nib.freesurfer.io.read_annot(rh_file_path)
    
    labels = lh_labels.tolist()
    for item in rh_labels.tolist():
        labels.append(item+int(parcel_number/2)) 
        
    ctabs = lh_ctab.tolist()
    for item in rh_ctab.tolist():
        ctabs.append(item) 
        
    names = lh_names 
    for item in rh_names: 
        names.append(item) 
   
    return labels, names, ctabs
   