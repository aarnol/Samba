import os   
import matplotlib.pyplot as plt     
from matplotlib import cm 
import numpy as np
from auxiliary.visualizations.hrf.group.utils import anatomical_parcel_names


def replace_all_min_with_mean(lst):
    if len(lst) == 0:
        return "List is empty"
    mean_value = sum(lst) / len(lst) 
    min_value = min(lst) 
    for i, val in enumerate(lst):
        if val == min_value:
            lst[i] = mean_value*0.95 
    return lst

def convert_to_ms(data):
    return data*5 
        
def normalize_vector(v):
    min_val = np.min(v)
    max_val = np.max(v)
    normalized_v = ((v - min_val) / (max_val - min_val)) * 2 - 1
    return normalized_v

def sort_based_on_y(x, y, x_std, color):   
    combined = list(zip(x, y, x_std, color)) 
    sorted_combined = sorted(combined, key=lambda x: x[0]) #, reverse=True) 
    sorted_x, sorted_y, sorted_x_std, sorted_color = zip(*sorted_combined) 
    return list(sorted_x), list(sorted_y), list(sorted_x_std), list(sorted_color)
 

def xy_scatterplot(x, y, x_std, color_list, title_color, plt_label, convert_ms=True):  
    y_lim_coef = 0.01
    # if convert_ms:
    #     x = convert_to_ms(x)   
    #     y_lim_coef = 0.02
    # else:
    #     y_lim_coef = 0.1
         
    x_sorted, y_sorted, x_std_sorted, sorted_color = sort_based_on_y(x, y, x_std, color_list)    
    
    sorted_color_set = list(set(sorted_color))
    x_sorted_set, y_sorted_set, x_std_sorted_set = [], [], []
    for color_i in sorted_color_set:
        set_mask = [color == color_i for color in sorted_color]   
        x_sorted_set.append(sum([x_sorted[i] for i in range(len(x_sorted)) if set_mask[i]])/sum(set_mask))  
        # y_sorted_set.append(sum([x_sorted[i] for i in range(len(y_sorted)) if set_mask[i]])/sum(set_mask))  
        x_std_sorted_set.append(sum([x_std_sorted[i] for i in range(len(x_std_sorted)) if set_mask[i]])*0.5/sum(set_mask))
 
    plt.scatter(range(len(x_sorted_set)), x_sorted_set, color=sorted_color_set, alpha=0.7, s=50)  
    plt.errorbar(range(len(x_sorted_set)), x_sorted_set, yerr=x_std_sorted_set, fmt='.', color='gray', alpha=0.4, capsize=5, linestyle="None")
     
    # labels_list = ['Default', 'Language', 'Cont', 'SalVenAttn', 'DorsAttn', 'Aud', 'SomMot', 'Visual']
    labels_list = ['Frontal', 'Temporal', 'Parietal', 'Occipital', 'Cingulate', 'Insular', 'Somatomotor']
    
    labels = plt.xticks(range(len(labels_list)), labels_list, rotation=45, fontsize=8) 
    for i, label in enumerate(labels[1]):
        label.set_color('black') 
        # label.set_color(sorted_color_set[i])
    plt.ylim(min(x_sorted_set)-y_lim_coef*min(x_sorted_set), max(x_sorted_set)+y_lim_coef*max(x_sorted_set)) 
    plt.gca().tick_params(axis='y', labelsize=10) 
    
    if title_color == 'red':
        title_color_forground = 'black'
    else:
        title_color_forground = 'black'
      
    x_position, y_position = 0.70, 0.07  
    # x_position, y_position = 0.07, 0.7
    plt.annotate(
        f"{plt_label}",
        xy=(x_position, y_position),  # Decreasing y position for each label
        xycoords='axes fraction',  # Coordinates are given in fraction of axes dimensions
        color=title_color_forground,
        fontsize=7,
        # fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.3',
            edgecolor=title_color,
            facecolor=title_color,  # Setting the face color same as text color
            alpha=0.0  # Adjusting the transparency
        )
    )
      
      
      
      
   
      
def list_mean(data):
    return sum(data) / len(data)

def list_sem(data): 
    # Standard Error (SE):
    s = np.std(data, ddof=1) 
    # Compute SEM
    n = len(data)
    sem = s / np.sqrt(n)*0.5
    return sem
        
def hrf_histrogram_anatomical(model):   
    response_delay, undershoot_delay, response_dispersion, undershoot_dispersion, response_scale, undershoot_scale = [], [], [], [], [], []
    response_delay_std, undershoot_delay_std, response_dispersion_std, undershoot_dispersion_std, response_scale_std, undershoot_scale_std = [], [], [], [], [], []
    
    
    # hrfs = []
    # for hrfs in model.hrfs:
    #     parcels_name, anatomical_main_regions, hrfs_i, right_lob_index, left_lob_index = anatomical_parcel_names(model.hrfs[0])
    #     hrfs.append(hrfs_i)
    
    
     
    for p in range(200):    
        d_r, d_u, s_r, s_u, c_r, c_u = [], [], [], [], [], []  
        for s in range(len(model.hrfs)): 
            d_r_s, d_u_s, s_r_s, s_u_s, c_r_s, c_u_s = model.hrfs[s][p].parameter_estimation()  
            d_r.append(d_r_s.item())
            d_u.append(d_u_s.item())
            s_r.append(s_r_s.item())
            s_u.append(s_u_s.item())
            c_r.append(c_r_s.item())
            c_u.append(c_u_s.item()) 
         
        response_delay.append(list_mean(d_r)) 
        undershoot_delay.append(list_mean(d_u)) 
        response_dispersion.append(list_mean(s_r))
        undershoot_dispersion.append(list_mean(s_u)) 
        response_scale.append(list_mean(c_r))
        undershoot_scale.append(list_mean(c_u))  
        
        response_delay_std.append(list_sem(d_r)) 
        undershoot_delay_std.append(list_sem(d_u)) 
        response_dispersion_std.append(list_sem(s_r))
        undershoot_dispersion_std.append(list_sem(s_u)) 
        response_scale_std.append(list_sem(c_r))
        undershoot_scale_std.append(list_sem(c_u))  
    
     
    parcels_name, anatomical_main_regions, response_delay_std, right_lob_index, left_lob_index = anatomical_parcel_names(response_delay_std)
    parcels_name, anatomical_main_regions, undershoot_delay_std, right_lob_index, left_lob_index = anatomical_parcel_names(undershoot_delay_std)
    parcels_name, anatomical_main_regions, response_dispersion_std, right_lob_index, left_lob_index = anatomical_parcel_names(response_dispersion_std)
    parcels_name, anatomical_main_regions, undershoot_dispersion_std, right_lob_index, left_lob_index = anatomical_parcel_names(undershoot_dispersion_std)
    parcels_name, anatomical_main_regions, response_scale_std, right_lob_index, left_lob_index = anatomical_parcel_names(response_scale_std)
    parcels_name, anatomical_main_regions, undershoot_scale_std, right_lob_index, left_lob_index = anatomical_parcel_names(undershoot_scale_std) 
 
    model.args.rh_lob_index = right_lob_index
    model.args.lh_lob_index = left_lob_index 
    model.args.lh_rh_lob_names = anatomical_main_regions
    model.args.parcels_name = parcels_name   
    # model.args.parcels_name[100] = 'Unknown'   
    # color_names = ['green', 'brown', 'orange', 'gray', 'purple', 'navy', 'black', 'maroon'] 
    color_names = [
        '#F77F00',    # Default
        '#EAE2B7',    # Language
        '#2a9d8f',    # Cont
        '#fee08b',    # SalVenAttn 
        '#d5bdaf',    # DorsAttn
        '#abdda4',    # Aud
        '#FCBF49',    # SomMot
        '#D62828'     # Visual
    ]
    hem_length = 93
    
    lh_color_list, color_conter = [], 0
    for i in range(hem_length, hem_length*2):  
        if i == (model.args.lh_lob_index[color_conter]): 
            color_i = color_names[color_conter]
            color_conter += 1 
        lh_color_list.append(color_i) 
   
    rh_color_list, color_conter = [], 0
    for i in range(0, hem_length):  
        if i == (model.args.rh_lob_index[color_conter]): 
            color_i = color_names[color_conter]   
            color_conter += 1 
        rh_color_list.append(color_i) 
     
    
    plt.figure(figsize=(16, 6))
    plt.subplot(2, 6, 1) 
    xy_scatterplot(x=response_delay[:hem_length], y=model.args.parcels_name[:hem_length], x_std=response_delay_std[:hem_length], color_list=lh_color_list, title_color='red', plt_label='Response\nDelay\n(LH)')
    
    plt.subplot(2, 6, 2)
    xy_scatterplot(x=response_dispersion[:hem_length], y=model.args.parcels_name[:hem_length], x_std=response_dispersion_std[:hem_length], color_list=lh_color_list, title_color='red', plt_label='Response\nDispersion\n(LH)')
    
    plt.subplot(2, 6, 3)
    xy_scatterplot(x=response_scale[:hem_length], y=model.args.parcels_name[:hem_length], x_std=response_scale_std[:hem_length], color_list=lh_color_list, title_color='red', plt_label='Response\nScale\n(LH)', convert_ms=False)
    
    plt.subplot(2, 6, 4)
    xy_scatterplot(x=undershoot_delay[:hem_length], y=model.args.parcels_name[:hem_length], x_std=undershoot_delay_std[:hem_length], color_list=lh_color_list, title_color='red', plt_label='Undershoot\nDelay\n(LH)')
    
    plt.subplot(2, 6, 5)
    xy_scatterplot(x=undershoot_dispersion[:hem_length], y=model.args.parcels_name[:hem_length], x_std=undershoot_dispersion_std[:hem_length], color_list=lh_color_list, title_color='red', plt_label='Undershoot\nDispersion\n(LH)')
    
    plt.subplot(2, 6, 6)
    xy_scatterplot(x=undershoot_scale[:hem_length], y=model.args.parcels_name[:hem_length], x_std=undershoot_scale_std[:hem_length], color_list=lh_color_list, title_color='red', plt_label='Undershoot\nScale\n(LH)', convert_ms=False)
    
    plt.subplot(2, 6, 7)
    xy_scatterplot(x=response_delay[hem_length:], y=model.args.parcels_name[hem_length:], x_std=response_delay_std[hem_length:], color_list=rh_color_list, title_color='blue', plt_label='Response\nDelay\n(RH)')
    
    plt.subplot(2, 6, 8)
    xy_scatterplot(x=response_dispersion[hem_length:], y=model.args.parcels_name[hem_length:], x_std=response_dispersion_std[hem_length:], color_list=rh_color_list, title_color='blue', plt_label='Response\nDispersion\n(RH)')
         
    plt.subplot(2, 6, 9)
    xy_scatterplot(x=response_scale[hem_length:], y=model.args.parcels_name[hem_length:], x_std=response_scale_std[hem_length:], color_list=rh_color_list, title_color='blue', plt_label='Response\nScale\n(RH)', convert_ms=False)
         
    plt.subplot(2, 6, 10)
    xy_scatterplot(x=undershoot_delay[hem_length:], y=model.args.parcels_name[hem_length:], x_std=undershoot_delay_std[hem_length:], color_list=rh_color_list,  title_color='blue', plt_label='Undershoot\nDelay\n(RH)')
          
    plt.subplot(2, 6, 11)
    xy_scatterplot(x=undershoot_dispersion[hem_length:], y=model.args.parcels_name[hem_length:], x_std=undershoot_dispersion_std[hem_length:], color_list=rh_color_list, title_color='blue', plt_label='Undershoot\nDispersion\n(RH)')
          
    plt.subplot(2, 6, 12)
    xy_scatterplot(x=undershoot_scale[hem_length:], y=model.args.parcels_name[hem_length:], x_std=undershoot_scale_std[hem_length:], color_list=rh_color_list, title_color='blue', plt_label='Undershoot\nScale\n(RH)',  convert_ms=False)
           
    plt.tight_layout()             
    plt.savefig(model.save_path+'group_parcells_histo_lobes.png', bbox_inches='tight', dpi=600)  
     
    plt.close()      
    
    


















 