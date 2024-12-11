import os   
import matplotlib.pyplot as plt     
from matplotlib import cm 
import numpy as np

def replace_all_min_with_mean(lst):
    if len(lst) == 0:
        return "List is empty"
    
    # Calculate the mean of the list
    mean_value = sum(lst) / len(lst)
    
    # Find the minimum value in the list
    min_value = min(lst)
    
    # Replace all occurrences of the minimum value with the mean value
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
    # x = replace_all_min_with_mean(x) 
    
    if convert_ms:
        x = convert_to_ms(x)   
        y_lim_coef = 0.02
    else:
        y_lim_coef = 0.1
        
    x_sorted, y_sorted, x_std_sorted, sorted_color = sort_based_on_y(x, y, x_std, color_list)    

    plt.scatter(range(len(x_sorted)), x_sorted, color=sorted_color, alpha=0.7, s=40)  
    plt.errorbar(range(len(x_sorted)), x_sorted, yerr=x_std_sorted, fmt='.', color='gray', alpha=0.4, capsize=5, linestyle="None")
    # for i, x in enumerate(x_sorted):
    #     plt.vlines(i, 0, x, colors=sorted_color[i], alpha=0.5) 
    
    
    # # plt.bar(range(len(x_sorted)), x_sorted, color=sorted_color, alpha=0.5) 
    # # plt.bar(range(len(loss_sorted)), loss_sorted, bottom=x_sorted, color='black', alpha=0.5) 
    
    # plt.bar(range(len(x_sorted)), loss_sorted, color=sorted_color, alpha=0.4)  
    # # plt.xticks(range(len(y_sorted)), y_sorted, color=sorted_color, rotation=90, fontsize=4)   
       
    labels = plt.xticks(range(len(x_sorted)), y_sorted, rotation=90, fontsize=4) 
    for i, label in enumerate(labels[1]):
        label.set_color(sorted_color[i])
        
    # plt.title(plt_label)
    # plt.ylabel('')
    # plt.yticks([]) 
    plt.ylim(min(x_sorted)-y_lim_coef*min(x_sorted), max(x_sorted)+y_lim_coef*max(x_sorted))
    
    
    if title_color == 'red':
        title_color_forground = 'black'
    else:
        title_color_forground = 'black'
      
    x_position, y_position = 0.63, 0.10  
    plt.annotate(
        f"{plt_label}",
        xy=(x_position, y_position),  # Decreasing y position for each label
        xycoords='axes fraction',  # Coordinates are given in fraction of axes dimensions
        color=title_color_forground,
        fontsize=12,
        fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.3',
            edgecolor=title_color,
            facecolor=title_color,  # Setting the face color same as text color
            alpha=0.25  # Adjusting the transparency
        )
    )
    
    x_position, y_position = 0.02, 0.90
    colors = ['green', 'brown', 'orange', 'gray']
    labels = ['Default      ', 'Language  ', 'Cont          ', 'SalVenAttn']
    for i, (color, label) in enumerate(zip(colors, labels)):
        plt.annotate(
        f"{label}",
        xy=(x_position, y_position - i * 0.11),  # Decreasing y position for each label
        xycoords='axes fraction',  # Coordinates are given in fraction of axes dimensions
        color='black',
        fontsize=10,
        bbox=dict(
            boxstyle='round,pad=0.3',
            edgecolor='white',
            facecolor=color,  # Setting the face color same as text color
            alpha=0.2  # Adjusting the transparency
        )
    )
    
    x_position, y_position = 0.12, 0.90  
    colors = ['purple', 'navy', 'black', 'maroon']
    labels = ['DorsAttn  ', 'Aud          ', 'SomMot   ', 'Visual      ']
    for i, (color, label) in enumerate(zip(colors, labels)):
        plt.annotate(
        f"{label}",
        xy=(x_position, y_position - i * 0.11),  # Decreasing y position for each label
        xycoords='axes fraction',  # Coordinates are given in fraction of axes dimensions
        color='black',
        fontsize=10,
        bbox=dict(
            boxstyle='round,pad=0.3',
            edgecolor='white',
            facecolor=color,  # Setting the face color same as text color
            alpha=0.2  # Adjusting the transparency
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
        
def hrf_histrogram(model):   
    response_delay, undershoot_delay, response_dispersion, undershoot_dispersion, response_scale, undershoot_scale = [], [], [], [], [], []
    response_delay_std, undershoot_delay_std, response_dispersion_std, undershoot_dispersion_std, response_scale_std, undershoot_scale_std = [], [], [], [], [], []
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
    
    model.args.parcels_name[100] = 'Unknown'  
    # cm_list = ['Greens', 'hot', 'Oranges', 'Greys', 'Purples', 'Blues', 'bone', 'Reds']  
    color_names = ['green', 'brown', 'orange', 'gray', 'purple', 'navy', 'black', 'maroon'] 
    
    lh_color_list, color_conter = [], 0
    for i in range(101, 200):  
        if i == (model.args.lh_lob_index[color_conter]): 
            color_i = color_names[color_conter]
            color_conter += 1  
        lh_color_list.append(color_i) 
        
    rh_color_list, color_conter = [], 0
    for i in range(0, 100):  
        if i == (model.args.rh_lob_index[color_conter]): 
            color_i = color_names[color_conter]   
            color_conter += 1 
        rh_color_list.append(color_i) 
       
    plt.figure(figsize=(18, 20))
    plt.subplot(6, 2, 1) 
    xy_scatterplot(x=response_delay[:99], y=model.args.parcels_name[:99], x_std=response_delay_std[:99], color_list=lh_color_list, title_color='red', plt_label='LH Response Delay')
    plt.subplot(6, 2, 2)
    xy_scatterplot(x=response_delay[99:], y=model.args.parcels_name[99:], x_std=response_delay_std[99:], color_list=rh_color_list, title_color='blue', plt_label='RH Response Delay')
    
    plt.subplot(6, 2, 3)
    xy_scatterplot(x=undershoot_delay[:99], y=model.args.parcels_name[:99], x_std=undershoot_delay_std[:99], color_list=lh_color_list, title_color='red', plt_label='LH Undershoot Delay')
    plt.subplot(6, 2, 4)
    xy_scatterplot(x=undershoot_delay[99:], y=model.args.parcels_name[99:], x_std=undershoot_delay_std[99:], color_list=rh_color_list,  title_color='blue', plt_label='RH Undershoot Delay')
        
    plt.subplot(6, 2, 5)
    xy_scatterplot(x=response_dispersion[:99], y=model.args.parcels_name[:99], x_std=response_dispersion_std[:99], color_list=lh_color_list, title_color='red', plt_label='LH Response Dispersion')
    plt.subplot(6, 2, 6)
    xy_scatterplot(x=response_dispersion[99:], y=model.args.parcels_name[99:], x_std=response_dispersion_std[99:], color_list=rh_color_list, title_color='blue', plt_label='RH Response Dispersion')
        
    plt.subplot(6, 2, 7)
    xy_scatterplot(x=undershoot_dispersion[:99], y=model.args.parcels_name[:99], x_std=undershoot_dispersion_std[:99], color_list=lh_color_list, title_color='red', plt_label='LH Undershoot Dispersion')
    plt.subplot(6, 2, 8)
    xy_scatterplot(x=undershoot_dispersion[99:], y=model.args.parcels_name[99:], x_std=undershoot_dispersion_std[99:], color_list=rh_color_list, title_color='blue', plt_label='RH Undershoot Dispersion')
        
    plt.subplot(6, 2, 9)
    xy_scatterplot(x=response_scale[:99], y=model.args.parcels_name[:99], x_std=response_scale_std[:99], color_list=lh_color_list, title_color='red', plt_label='LH Response Scatter', convert_ms=False)
    plt.subplot(6, 2, 10)
    xy_scatterplot(x=response_scale[99:], y=model.args.parcels_name[99:], x_std=response_scale_std[99:], color_list=rh_color_list, title_color='blue', plt_label='RH Response Scatter', convert_ms=False)
        
    plt.subplot(6, 2, 11)
    xy_scatterplot(x=undershoot_scale[:99], y=model.args.parcels_name[:99], x_std=undershoot_scale_std[:99], color_list=lh_color_list, title_color='red', plt_label='LH Undershoot Scatter', convert_ms=False)
    plt.subplot(6, 2, 12)
    xy_scatterplot(x=undershoot_scale[99:], y=model.args.parcels_name[99:], x_std=undershoot_scale_std[99:], color_list=rh_color_list, title_color='blue', plt_label='RH Undershoot Scatter',  convert_ms=False)
          
    # plt.suptitle('Subject-'+str(ym_batch[-2:])+' (inferred parameters of HRFs)', fontsize = 6,
    #              bbox=dict(facecolor='yellow', edgecolor='none', alpha=0.7))
    
    plt.tight_layout()            
    # save_path = model.args.output_dir + '/hrf/parcells_histo/'
    # if not os.path.exists(save_path): 
    #     os.mkdir(save_path)    
    plt.savefig(model.save_path+'group_parcells_histo_.png', bbox_inches='tight', dpi=300)  
     
    plt.close()      
    
    




































# import os  
# # from brainiak.utils.fmrisim import _double_gamma_hrf as hrf_func
# # import matplotlib.image as mpimg   
# import matplotlib.pyplot as plt    
# # from nilearn.datasets import fetch_surf_fsaverage
# # from nilearn import plotting
# # import numpy as np
# # import brainplotlib as bpl 
# # import matplotlib.pyplot as plt
# # from skimage.metrics import structural_similarity as ssim
# # from skimage.metrics import mean_squared_error 
# # from einops import rearrange   
# # from nilearn.plotting import plot_glass_brain  
# # from matplotlib.colors import LinearSegmentedColormap
# # import matplotlib.colors as mcolors
# from matplotlib import cm 


# def xy_scatter(x, y, label, color='blue'):
#     sorted_pairs = sorted(zip(x, y))
#     x_sorted = [pair[0] for pair in sorted_pairs]
#     y_sorted = [pair[1] for pair in sorted_pairs]
 
#     plt.scatter(range(len(x_sorted)), x_sorted, c=color) 
#     plt.xticks(range(len(y_sorted)), y_sorted, c=color, rotation=90, fontsize=4)   
#     plt.ylabel(label)
#     plt.title(label)   

# def hrf_histo(model, ym_batch, iteration):
#     response_delay, undershoot_delay, response_dispersion, undershoot_dispersion, response_scale, undershoot_scale = [], [], [], [], [], []
#     for p in range(200):  
#         sub_index = model.meg_sub_list.index(ym_batch[-2:]) 
#         d_r, d_u, s_r, s_u, c_r, c_u = model.hrfs[sub_index][p].parameter_estimation()  
#         response_delay.append(d_r.item()) 
#         undershoot_delay.append(d_u.item()) 
#         response_dispersion.append(s_r.item())
#         undershoot_dispersion.append(s_u.item()) 
#         response_scale.append(c_r.item())
#         undershoot_scale.append(c_u.item())  
            
#     plt.figure(figsize=(18, 20))
#     plt.subplot(6, 2, 1)
#     xy_scatter(x=response_delay[:99], y=model.args.parcels_name[:99], label='LH Response Delay', color='blue')
#     plt.subplot(6, 2, 2)
#     xy_scatter(x=response_delay[99:], y=model.args.parcels_name[99:], label='RH Response Delay', color='red')
    
#     plt.subplot(6, 2, 3)
#     xy_scatter(x=undershoot_delay[:99], y=model.args.parcels_name[:99], label='LH Undershoot Delay', color='blue')
#     plt.subplot(6, 2, 4)
#     xy_scatter(x=undershoot_delay[99:], y=model.args.parcels_name[99:], label='RH Undershoot Delay', color='red')
        
#     plt.subplot(6, 2, 5)
#     xy_scatter(x=response_dispersion[:99], y=model.args.parcels_name[:99], label='LH Response Dispersion', color='blue')
#     plt.subplot(6, 2, 6)
#     xy_scatter(x=response_dispersion[99:], y=model.args.parcels_name[99:], label='RH Response Dispersion', color='red')
        
#     plt.subplot(6, 2, 7)
#     xy_scatter(x=undershoot_dispersion[:99], y=model.args.parcels_name[:99], label='LH Undershoot Dispersion', color='blue')
#     plt.subplot(6, 2, 8)
#     xy_scatter(x=undershoot_dispersion[99:], y=model.args.parcels_name[99:], label='RH Undershoot Dispersion', color='red')
        
#     plt.subplot(6, 2, 9)
#     xy_scatter(x=response_scale[:99], y=model.args.parcels_name[:99], label='LH Response Scale', color='blue')
#     plt.subplot(6, 2, 10)
#     xy_scatter(x=response_scale[99:], y=model.args.parcels_name[99:], label='RH Response Scale', color='red')
        
#     plt.subplot(6, 2, 11)
#     xy_scatter(x=undershoot_scale[:99], y=model.args.parcels_name[:99], label='LH Undershoot Scale', color='blue')
#     plt.subplot(6, 2, 12)
#     xy_scatter(x=undershoot_scale[99:], y=model.args.parcels_name[99:], label='RH Undershoot Scale', color='red')
        
#     plt.tight_layout()            
#     save_path = model.args.output_dir + '/hrf/hrf_parcells_scale/'
#     if not os.path.exists(save_path): 
#         os.mkdir(save_path)    
#     plt.savefig(save_path+'hrf_parcells_scale_'+str(iteration)+'.png', bbox_inches='tight', dpi=400)  
#     plt.close() 
    
# def xy_scatter(x, y, label, color='blue'):
#     sorted_pairs = sorted(zip(x, y))
#     x_sorted = [pair[0] for pair in sorted_pairs]
#     y_sorted = [pair[1] for pair in sorted_pairs]
#     # Create scatter plot 
#     plt.scatter(range(len(x_sorted)), x_sorted, c=color) 
#     plt.xticks(range(len(y_sorted)), y_sorted, c=color, rotation=90, fontsize=4)   
#     plt.ylabel(label)
#     plt.title(label)  
