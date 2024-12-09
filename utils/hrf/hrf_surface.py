import os  
import matplotlib.image as mpimg   
import matplotlib.pyplot as plt    
from nilearn.datasets import fetch_surf_fsaverage
from nilearn import plotting
import numpy as np
from io import BytesIO
import matplotlib as mpl

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

def list_mean(data):
    return sum(data) / len(data)

def denoise_with_moving_average(data, window_size=5):
    """ Denoise a 1D array using a simple moving average.
    
    Args:
        data (np.array): The input 1D array to denoise.
        window_size (int): The size of the moving average window.
        
    Returns:
        np.array: The denoised array.
    """
    # Ensure window_size is odd to have a symmetric window
    if window_size % 2 == 0:
        window_size += 1
    
    # Initialize the array for the denoised data
    denoised_array = np.zeros_like(data)
    
    # Calculate the half window size for indexing
    half_window = window_size // 2
    
    # Perform moving average
    for i in range(len(data)):
        # Define the window boundaries
        start_index = max(i - half_window, 0)
        end_index = min(i + half_window + 1, len(data))
        
        # Compute the average within the window
        denoised_array[i] = np.mean(data[start_index:end_index])
    
    return np.array(denoised_array)

# def normalize_vector(v):  
#     min_val = np.min(v) 
#     max_val = np.max(v) 
#     normalized_v = ((v - min_val) / (max_val - min_val)) * 2 - 1
#     return normalized_v

def normalize_vector(numbers, lower_percentile=10, upper_percentile=90):
    # import pdb;pdb.set_trace()
    numbers = np.array(numbers)
    # Calculate the mean of the list
    mean_value = np.mean(numbers)
    
    # Determine the lower and upper threshold based on percentiles
    lower_threshold = np.percentile(numbers, lower_percentile)
    upper_threshold = np.percentile(numbers, upper_percentile)
    
    # Replace values below the lower threshold or above the upper threshold with the mean
    replaced_numbers = [x if lower_threshold <= x <= upper_threshold else mean_value for x in numbers]
    
    return denoise_with_moving_average(np.array(replaced_numbers))
 
def hrf_parcells_surf_innerloop(model, ym_batch, cmap = 'seismic'):
    
    
    response_delay = np.zeros_like(model.args.parcel_name_vertex)+0.000000001
    undershoot_delay = np.zeros_like(model.args.parcel_name_vertex)+0.000000001 
    
    response_dispersion = np.zeros_like(model.args.parcel_name_vertex)+0.000000001
    undershoot_dispersion = np.zeros_like(model.args.parcel_name_vertex)+0.000000001
    
    response_scale = np.zeros_like(model.args.parcel_name_vertex)+0.000000001
    undershoot_scale = np.zeros_like(model.args.parcel_name_vertex)+0.000000001 
    model.args.parcel_name_vertex = np.array(model.args.parcel_name_vertex)
    for parcel_index in range(200):    
        d_r, d_u, s_r, s_u, c_r, c_u = [], [], [], [], [], [] 
        for s in range(len(model.hrfs)):
            d_r_s, d_u_s, s_r_s, s_u_s, c_r_s, c_u_s = model.hrfs[s][parcel_index].parameter_estimation()  
            d_r.append(d_r_s.item())
            d_u.append(d_u_s.item())
            s_r.append(s_r_s.item())
            s_u.append(s_u_s.item())
            c_r.append(c_r_s.item())
            c_u.append(c_u_s.item()) 
         
        d_r = list_mean(d_r) 
        d_u = list_mean(d_u) 
        s_r = list_mean(s_r)
        s_u = list_mean(s_u) 
        c_r = list_mean(c_r)
        c_u = list_mean(c_u)  
        
        parcel_indecies = np.where(model.args.parcel_name_vertex==parcel_index)[0]
        
        response_delay[parcel_indecies] = d_r 
        response_dispersion[parcel_indecies] = s_r 
        response_scale[parcel_indecies] = c_r 
        
        undershoot_delay[parcel_indecies] = d_u
        undershoot_dispersion[parcel_indecies] = s_u
        undershoot_scale[parcel_indecies] = c_u 
         
         
    fig = plt.figure(figsize=(16, 4))
    axes = [fig.add_subplot(2, 6, i+1, projection='3d') for i in range(12)] 
    
    fsaverage_surface = fetch_surf_fsaverage(mesh='fsaverage5')
    plt.tight_layout(pad=0) 
    alpha = 0.99
    
    # response_delay = replace_extreme_values(response_delay[:10242])
    # response_dispersion = replace_extreme_values(response_dispersion[:10242])
    # response_scale = replace_extreme_values(response_scale[:10242])
    # undershoot_delay = replace_extreme_values(undershoot_delay[:10242])
    # undershoot_dispersion = replace_extreme_values(undershoot_dispersion[:10242])
    # undershoot_scale = replace_extreme_values(undershoot_scale[:10242]) 
    
    # response_delay = replace_extreme_values(response_delay[10242:])
    # response_dispersion = replace_extreme_values(response_dispersion[10242:])
    # response_scale = replace_extreme_values(response_scale[10242:])
    # undershoot_delay = replace_extreme_values(undershoot_delay[10242:])
    # undershoot_dispersion = replace_extreme_values(undershoot_dispersion[10242:])
    # undershoot_scale = replace_extreme_values(undershoot_scale[10242:])
    
    plotting.plot_surf_stat_map(fsaverage_surface["infl_left"], normalize_vector(response_delay[:10242]),  hemi='left',  view='lateral',  bg_map=fsaverage_surface['sulc_left'], cmap=cmap, alpha=alpha, axes=axes[0], colorbar=False)   
    plotting.plot_surf_stat_map(fsaverage_surface["infl_left"], normalize_vector(response_dispersion[:10242]),  hemi='left',  view='lateral',  bg_map=fsaverage_surface['sulc_left'], cmap=cmap, alpha=alpha, axes=axes[1], colorbar=False)  
    plotting.plot_surf_stat_map(fsaverage_surface["infl_left"], normalize_vector(response_scale[:10242]),  hemi='left',  view='lateral',  bg_map=fsaverage_surface['sulc_left'], cmap=cmap, alpha=alpha, axes=axes[2], colorbar=False)  
    plotting.plot_surf_stat_map(fsaverage_surface["infl_left"], normalize_vector(undershoot_delay[:10242]),  hemi='left',  view='lateral',  bg_map=fsaverage_surface['sulc_left'], cmap=cmap, alpha=alpha, axes=axes[3], colorbar=False)   
    plotting.plot_surf_stat_map(fsaverage_surface["infl_left"], normalize_vector(undershoot_dispersion[:10242]),  hemi='left',  view='lateral',  bg_map=fsaverage_surface['sulc_left'], cmap=cmap, alpha=alpha, axes=axes[4], colorbar=False)   
    plotting.plot_surf_stat_map(fsaverage_surface["infl_left"], normalize_vector(undershoot_scale[:10242]),  hemi='left',  view='lateral',  bg_map=fsaverage_surface['sulc_left'], cmap=cmap, alpha=alpha, axes=axes[5], colorbar=False)    
    plt.tight_layout(pad=0) 
    
    # import pdb;pdb.set_trace() 
    response_delay = normalize_vector(replace_all_min_with_mean(normalize_vector(replace_all_min_with_mean(response_delay[10242:]))))     
    response_dispersion = normalize_vector(replace_all_min_with_mean(response_dispersion[10242:])) 
    response_scale = normalize_vector(replace_all_min_with_mean(response_scale[10242:]))  
    undershoot_delay = normalize_vector(replace_all_min_with_mean(normalize_vector(replace_all_min_with_mean(undershoot_delay[10242:])))) 
    undershoot_dispersion = normalize_vector(replace_all_min_with_mean(undershoot_dispersion[10242:])) 
    undershoot_scale = normalize_vector(replace_all_min_with_mean(undershoot_scale[10242:]))
    # import pdb;pdb.set_trace() 
     
    plotting.plot_surf_stat_map(fsaverage_surface["infl_right"], response_delay,  hemi='right',  view='lateral',  bg_map=fsaverage_surface['sulc_left'], cmap=cmap, alpha=alpha, axes=axes[6], colorbar=False) 
    plotting.plot_surf_stat_map(fsaverage_surface["infl_right"], response_dispersion,  hemi='right',  view='lateral',  bg_map=fsaverage_surface['sulc_left'], cmap=cmap, alpha=alpha, axes=axes[7], colorbar=False) 
    plotting.plot_surf_stat_map(fsaverage_surface["infl_right"], response_scale,  hemi='right',  view='lateral',  bg_map=fsaverage_surface['sulc_left'], cmap=cmap, alpha=alpha, axes=axes[8], colorbar=False)  
    plotting.plot_surf_stat_map(fsaverage_surface["infl_right"], undershoot_delay,  hemi='right',  view='lateral',  bg_map=fsaverage_surface['sulc_left'], cmap=cmap, alpha=alpha, axes=axes[9], colorbar=False)  
    plotting.plot_surf_stat_map(fsaverage_surface["infl_right"], undershoot_dispersion,  hemi='right',  view='lateral',  bg_map=fsaverage_surface['sulc_left'], cmap=cmap, alpha=alpha,  axes=axes[10], colorbar=False)  
    plotting.plot_surf_stat_map(fsaverage_surface["infl_right"], undershoot_scale,  hemi='right',  view='lateral',  bg_map=fsaverage_surface['sulc_left'], cmap=cmap, alpha=alpha, axes=axes[11], colorbar=False)  
    plt.tight_layout(pad=0) 
        
    fontsize = 10
    x, y = 0.0, 0.75
    axes[0].annotate('Response\nDelay\n(LH)', xy=(x, y), xycoords='axes fraction', ha='left', va='center', fontsize=fontsize)
    axes[1].annotate('Response\nDispersion\n(LH)', xy=(x-0.1, y), xycoords='axes fraction', ha='left', va='center', fontsize=fontsize) 
    axes[2].annotate('Response\nScale\n(LH)', xy=(x, y), xycoords='axes fraction', ha='left', va='center', fontsize=fontsize)  
    axes[3].annotate('Undershoot\nDelay\n(LH)', xy=(x, y), xycoords='axes fraction', ha='left', va='center', fontsize=fontsize)
    axes[4].annotate('Undershoot\nDispersion\n(LH)', xy=(x-0.1, y), xycoords='axes fraction', ha='left', va='center', fontsize=fontsize) 
    axes[5].annotate('Undershoot\nScale\n(LH)', xy=(x, y), xycoords='axes fraction', ha='left', va='center', fontsize=fontsize) 
    
    axes[6].annotate('Response\nDelay\n(RH)', xy=(x, y), xycoords='axes fraction', ha='left', va='center', fontsize=fontsize)
    axes[7].annotate('Response\nDispersion\n(RH)', xy=(x-0.1, y), xycoords='axes fraction', ha='left', va='center', fontsize=fontsize) 
    axes[8].annotate('Response\nScale\n(RH)', xy=(x, y), xycoords='axes fraction', ha='left', va='center', fontsize=fontsize)  
    axes[9].annotate('Undershoot\nDelay\n(RH)', xy=(x, y+0.015), xycoords='axes fraction', ha='left', va='center', fontsize=fontsize)
    axes[10].annotate('Undershoot\nDispersion\n(RH)', xy=(x-0.1, y), xycoords='axes fraction', ha='left', va='center', fontsize=fontsize) 
    axes[11].annotate('Undershoot\nScale\n(RH)', xy=(x, y), xycoords='axes fraction', ha='left', va='center', fontsize=fontsize) 
       
    # fig.text(0.025, 0.5, 'Subject-'+str(ym_batch), 
    #      rotation='vertical',  
    #      ha='right', 
    #      va='center', 
    #      fontsize=25, 
    #      bbox=dict(facecolor='gray', edgecolor='none', alpha=0.2))
    
    plt.axis('off')  # Turn off the axis.
    plt.tight_layout(pad=0)   
    
    cbar_ax = fig.add_axes([0.935, 0.25, 0.01, 0.5]) #  [left, bottom, width, height]  
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1, clip=False),cmap=cmap),
                        cax=cbar_ax,
                        orientation='vertical')

    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Medium', 'High']) 
    
    return fig

def fig_2_axes(fig_to_capture, axes):
    # Save it to a BytesIO object
    buf = BytesIO()
    fig_to_capture.savefig(buf, format="png", dpi=400)
    plt.close(fig_to_capture)  # Close the generated figure
    # Load the image into a numpy array and display it in your target axes
    buf.seek(0)
    img_arr = mpimg.imread(buf)
    axes.imshow(img_arr)
    axes.set_xlabel('')
    axes.set_ylabel('') 
    axes.set_xticks([])
    axes.set_yticks([]) 
    axes.spines['top'].set_alpha(0.5)
    axes.spines['right'].set_alpha(0.5)
    axes.spines['bottom'].set_alpha(0.5)
    axes.spines['left'].set_alpha(0.5)

    
def hrf_surface_draw(model, cmap = 'seismic'): 
    meg_sub_list = model.args.meg_sub_list 
    fig, ax = plt.subplots(figsize=(16, 4)) 
    fig = hrf_parcells_surf_innerloop(model, meg_sub_list, cmap=cmap)  
    ax.set_xlabel('')
    ax.set_ylabel('') 
    ax.set_xticks([])
    ax.set_yticks([]) 
    ax.spines['top'].set_alpha(0.5)
    ax.spines['right'].set_alpha(0.5)
    ax.spines['bottom'].set_alpha(0.5)
    ax.spines['left'].set_alpha(0.5)
     
    # save_path = model.args.output_dir + '/hrf/parcells_surf/' 
    # if not os.path.exists(save_path): 
    #     os.mkdir(save_path)       
    plt.savefig(model.save_path+'group_parcells_surf.png', dpi=600)   
    fig.subplots_adjust(left=0)
     
    plt.close() 
    
    
    
    