 
from args import make_directroy 
from model.model_registry import str2model   
from torch.optim import Adam   
from args import params_fn 
import torch.nn as nn
import numpy as np
import torch
import time 
import os 
from utils.graph_bipartite_plot import bipartite_plot
from utils.graph_chord_plot import graph_chord_draw 


from utils.hrf.hrf_histo import hrf_histrogram

from utils.hrf.hrf_histrogram_lobe_anatomical import hrf_histrogram_anatomical 
from utils.hrf.hrf_shapes_mean_anatomical import group_hrf_shapes_mean 

from utils.hrf.hrf_surface import hrf_surface_draw
 
# from auxiliary.visualizations.hrf.group.region_histo_anatomical import group_hrf_histo_lobes
# from auxiliary.visualizations.hrf.group.hrf_shapes import group_hrf_shapes 
# from auxiliary.visualizations.hrf.group.hrf_shapes_mean_anatomical import group_hrf_shapes_mean 
# from auxiliary.visualizations.hrf.group.hrf_histo_anatomical import group_hrf_histo
# from auxiliary.visualizations.hrf.group.hrf_surface import group_hrf_parcells_surf
# from auxiliary.visualizations.attention.graph_attention_circular_anatomical import group_single_attention 
# from auxiliary.visualizations.attention.wavelet_attention import alphas_barplot
# from auxiliary.visualizations.attention.skip_lossplot import skip_lossplot
# from auxiliary.visualizations.attention.bipartite_viz_anatomical import bipartite_plot

# from data_eegfmri.schaeferparcel_kong2022_17network import SchaeferParcel_Kong2022_17Network


def validation(args, model, valid_dataloader, iteration=0): 
    first_time = True
    best_valoss = np.inf 
    valoss = 0
    for it_eval, ((xm_i, xf_i, y_meta, y_batch), [miniute_index, sub_f, sub_m]) in enumerate(valid_dataloader):        
        with torch.no_grad():
            model.eval()    
            [valoss_i, loss_mse_parcells_i], vizi_i = model.loss(xm_i, xf_i, sub_m, sub_f, iteration, 
                                                                 train_mode=False, visualization=True) 
 
            if first_time:
                first_time = False
                xf = vizi_i[1][:10, :]
                xfm_hat = vizi_i[3][:10, :] 
            else:
                xf = torch.cat((xf, vizi_i[1][5:20, :]), dim=0)
                xfm_hat = torch.cat((xfm_hat, vizi_i[3][5:20, :]), dim=0)
            
            if valoss_i < best_valoss:
                best_valoss = valoss_i
                loss_mse_parcells = loss_mse_parcells_i
                vizi = vizi_i
                
            valoss += valoss_i
                 
        if it_eval >= 1:
            break
        
    vizi_i[1] = xf
    vizi_i[3] = xfm_hat
    args.output_dir = make_directroy(args)  
    
    h_att = vizi[5].cpu().numpy()  
    alphas = vizi[6].cpu().numpy()     
    return h_att, alphas
 
  
def list_folders(directory): 
    return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
   

def dataloader_return(args, source_sub, target_sub):
    
    train_dataloader = NumpyDataset(
        meg_dir = args.meg_dir,  
        fmri_dir =  args.fmri_dir,  
        split = 'train',
        n_way = args.n_way,
        single_subj=args.single_subj, 
        single_subj_list=None
    )   
    
    train_dataloader.meg_subs = source_sub
    train_dataloader.fmri_subs = target_sub   
    
    test_dataloader = NumpyDataset(
        meg_dir = args.meg_dir,  
        fmri_dir = args.fmri_dir,  
        split = 'train',
        n_way = 20,  
        single_subj=train_dataloader.single_subj, 
        single_subj_list=[train_dataloader.meg_subs, train_dataloader.fmri_subs]
    ) 
    
    adj_mat_path = '/home/aa2793/gibbs/arman/datasets/NEUROSCIENCE/MEGfMRI/forrest_gump/' 
    args.fmri_adjacency_matrix_dir = adj_mat_path + 'graphs/arman/fmri-'+str(n_fmri_parcel)+'parcels/'+train_dataloader.fmri_subs[0]+'.npy'
    args.meg_adjacency_matrix_dir = adj_mat_path + 'graphs/arman/meg-'+str(n_meg_parcel)+'parcels/'+train_dataloader.meg_subs[0]+'.npy'
    # print_gpu_info(args, train_dataloader)
    return args, train_dataloader, test_dataloader   

def hrf_forpaper(args, save_path, path_folder = 'tst_m2f_v1'): 
    directory_path = '../outputs/' + path_folder + '/'
    model_folders = list_folders(directory_path) 
    hrfs = [] 
    for i, model_folder_i in enumerate(model_folders):  
        loaded_model = torch.load('../outputs/'+path_folder+'/'+model_folder_i+'/xmodel/best_model.pth')   
        print('uploaded: '+model_folder_i)
        model = loaded_model['model']
        model.to(args.device)  
        args = model.args  
        hrfs.append(model.hrfnet.hrfs) 
    
    proto_model = str2model(args.model)
    model = proto_model(args).to(args.device) 
    model.hrfs = hrfs 
    
    save_path = './xoutputs/hrfs/'
    if not os.path.exists(save_path): 
        os.mkdir(save_path)  
    model.save_path = save_path
    hrf_histrogram(model)
    group_hrf_shapes_mean(model) 
    hrf_surface_draw(model, cmap='RdBu_r') 
    hrf_histrogram_anatomical(model) 
    
       
    
def wavelet_skiploss(args):
    model_folder_i = 'tst_m2f_v1-20240421-113756--(sub-02 -> sub-20)'
    loaded_model = torch.load('../outputs/'+args.model+'/'+model_folder_i+'/xmodel/best_model.pth') 
    model = loaded_model['model']
    model.to(args.device)  
    args = model.args
    result_list = model.result_list
    res_list = []
    for result_list_i in result_list:  
        res_list.append(result_list_i[40:])
           
    # # import pdb;pdb.set_trace()
    # with open('res_list.txt', 'w') as file:
    #     for item in res_list:
    #         file.write(f"{item}\n") 
    skip_lossplot(res_list, x1_var=0.036, x2_var=0.04)
   
if __name__ == "__main__":  
    
    args = params_fn() 
    n_fmri_parcel=500 
    n_meg_parcel=200
    
    datasets = ['megfmri', 'eegfmri_translation']  
    args = params_fn(server_mode='mccleary', dataset=datasets[0]) 
     
    # args.model = 'tst_m2f_v1'
    args.single_subj = True   
    from data.dataloader import  NumpyBatchDataset as NumpyDataset 
  
    save_path = '../outputs/figures/'
    if not os.path.exists(save_path): 
        os.mkdir(save_path) 
    save_path = save_path + 'group/'
    if not os.path.exists(save_path): 
        os.mkdir(save_path) 
     
    path_folder = 'tst_m2f_v1-archive'
    directory_path = '../outputs/' + path_folder + '/'
    model_folders = list_folders(directory_path) 
    
    
    # GRAPH VISUALISATION
    hrf_forpaper(args, save_path, path_folder = 'tst_m2f_v1-archive')
    
    
    
    
    
    # # ## saving attentions for paper
    # alphas, h_att = [], []
    # for i, model_folder_i in enumerate(model_folders): 
    #     loaded_model = torch.load('../outputs/'+path_folder+'/'+model_folder_i+'/xmodel/best_model.pth')  
    #     print('uploaded: '+model_folder_i)
    #     model = loaded_model['model']
    #     model.to(args.device)  
    #     break
    #     args = model.args
    #     source_sub = [model_folder_i[29:-1][:6]]
    #     target_sub = [model_folder_i[29:-1][10:]]
    #     args, train_dataloader, test_dataloader =  dataloader_return(args, source_sub, target_sub)
        
    #     h_att_i, alphas_i = validation(args, model, test_dataloader)
    #     h_att.append(h_att_i.mean(0)[:, :, -1]) 
    #     alphas.append(alphas_i)
        
    #     if i==10:
    #         break
    
    
    # # # visualization of the attention  
    # h_att_sum = 0
    # for h_att_i in h_att: 
    #     h_att_sum += h_att_i
    # h_att = h_att_sum/len(h_att) 
    # np.save('h_att.npy', h_att)
    h_att = np.load('h_att.npy')
    
    
    
    
    # # GRAPH VISUALISATION
    # graph_chord_draw(model, h_att, save_name='graph_chord'+path_folder) 
    # bipartite_plot(h_att, model, brain_sides='left_heme', k=5, n_regions_show=7)
    # bipartite_plot(h_att, model, brain_sides='right_heme', k=5, n_regions_show=7)
    
    # bipartite_plot(h_att, model, brain_sides='left_heme', k=5, n_regions_show=5)
    # bipartite_plot(h_att, model, brain_sides='right_heme', k=5, n_regions_show=5)
     
  
    # # WAVELET VISUALISATION
    # wavelet_skiploss(args)
    # alphas_barplot(alphas) 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # alphas = []
    # h_att, model_names = [], []   
    # for i, model_folder_i in enumerate(model_folders): 
    #     # model_names.append(model_folder_i[29:-1])  
    #     loaded_model = torch.load('../outputs/'+path_folder+'/'+model_folder_i+'/xmodel/best_model.pth') 
    #     # model_folder_i = 'tst_m2f_v1-20240421-113756--(sub-02 -> sub-20)'
    #     # loaded_model = torch.load('../outputs/'+args.model+'/'+model_folder_i+'/xmodel/best_model.pth') 
 
    #     print('uploaded: '+model_folder_i)
    #     model = loaded_model['model']
    #     model.to(args.device)  
    #     args = model.args
        
        # result_list = model.result_list
        # res_list = []
        # for result_list_i in result_list: 
        #     res_list.append(result_list_i[40:])
            
        # skip_lossplot(res_list, x1_var=0.036, x2_var=0.04)
        # # import pdb;pdb.set_trace()
        # with open('res_list.txt', 'w') as file:
        #     for item in res_list:
        #         file.write(f"{item}\n")
        # import pdb;pdb.set_trace()
         
        # source_sub = [model_folder_i[29:-1][:6]]
        # target_sub = [model_folder_i[29:-1][10:]]
        # args, train_dataloader, test_dataloader =  dataloader_return(args, source_sub, target_sub)
        
        
        # hrfs.append(model.hrfnet.hrfs) 
        
 
    # proto_model = str2model(args.model)
    # model = proto_model(args).to(args.device) 
    # model.hrfs = hrfs
    # model.model_names = model_names
     
     
     
    # 
    # import pdb;pdb.set_trace() 
    
     
    
      
    
    

      
    
    
 
  