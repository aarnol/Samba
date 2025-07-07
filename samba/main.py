from model.model_registry import str2model  
from args import params_fn, print_gpu_info 
from args import make_directroy, meg2List 
from torch.optim import Adam    
import torch.nn as nn
import numpy as np
import torch
import time 
import os
from utils.viz_corr import correlation_r2_plot 
from einops import rearrange

## training
def training(args, model, train_dataloader, valid_dataloader):  
    optimizer = Adam(model.parameters(), args.lr)   
    best_valoss = np.inf
    
    for iteration, ((xm, xf, y_meta, y_batch), [t, sub_f, sub_m]) in enumerate(train_dataloader):  
        
        #move tensors to device
        xm = xm.to(args.device)
        xf = xf.to(args.device)
        

        model.train()   
        loss = model.loss(xm, xf, sub_m, sub_f, iteration)   
        
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  
        
        # validate and save model 
        if iteration == 0  or iteration % args.validation_iteration == 0:  
            valoss_i = validation(args, model, valid_dataloader, iteration) 
            
            if valoss_i < best_valoss:
                best_valoss = valoss_i
                torch.save(
                    {'iteration': iteration, 
                    'model': model.cpu(),  
                    'result_list': model.result_list},
                    args.output_dir + '/xmodel/best_model.pth' 
                ) 
                
            torch.save(
                {'iteration': iteration, 
                'model': model,  
                'result_list': model.result_list},
                args.output_dir + '/xmodel/current_model.pth' 
            ) 
            
            model.print_results(iteration)   
            meg2List(args.output_dir + '/xmodel/results.txt', model.result_list)  
            model.to(args.device) 
        if iteration == 1500: break     # founded by validation set
 
## validation 
def validation(args, model, valid_dataloader, iteration): 
    first_time = True
    best_valoss = np.inf 
    valoss = 0
    for it_eval, ((xm_i, xf_i, y_meta, y_batch), [miniute_index, sub_f, sub_m]) in enumerate(valid_dataloader):        
        with torch.no_grad():
            model.eval()     
            valoss_i, x_i, x_hat_i, [zm_hrf_i, h_att_i, alphas_i] = model.loss(xm_i, xf_i, sub_m, sub_f, iteration) 
       
            if first_time:
                first_time = False
                x = x_i[:5, :]
                x_hat = x_hat_i[:5, :]
                if args.mc_probabilistic:
                    x_hat_std = x_hemo_hat_std_i[:5, :]
            else:
                x = torch.cat((x, x_i[5:10, :]), dim=0)
                x_hat = torch.cat((x_hat, x_hat_i[5:10, :]), dim=0)
                if args.mc_probabilistic:
                    x_hat_std = torch.cat((x_hat_std, x_hemo_hat_std_i[5:10, :]), dim=0)
            
            if valoss_i < best_valoss:
                best_valoss = valoss_i
                loss_mse_parcells = valoss_i 
                
            valoss += valoss_i
                 
        if it_eval >= 10:
            break 
        
    args.output_dir = make_directroy(args)   
    correlation_r2_plot(args, iteration, x=x, y=x_hat, title = args.model, folder_extention='')  

    return valoss_i
  
if __name__ == "__main__":  
    import os, time
    print(f"[{time.strftime('%H:%M:%S')}] Script running with PID: {os.getpid()}", flush=True) 
    print("SambaEleToHemo main.py", flush = True)
    args = params_fn(server_mode='mccleary', dataset='eegfmri_translation') 
    args.model = 'SambaEleToHemo'   
    args.single_subj = True  
     
    sub_ele = args.ele_sub_list[-1]
    sub_hemo = args.hemo_sub_list[-1] 
    if args.single_subj:
        args.ele_sub_list = [sub_ele]
        args.hemo_sub_list = [sub_hemo]
        args.save_prefix = 'eleSub' + sub_ele + '_hemoSub' + sub_hemo
    else:
        args.save_prefix = 'allSubs'
         
    

    from data.dataloader_second import  NumpyBatchDatasetHCP as NumpyDataset 
    
    import importlib
    import data.dataloader_second as d
    importlib.reload(d)  # force reload of the module


    cwd = os.getcwd()
    fmri_data_path = os.path.join(cwd, 'samba', 'data', 'fmri')
    fnirs_data_path = os.path.join(cwd, 'samba', 'data', 'fnirs', 'fnirsHbC.save')  # Assuming fnirsHbC.save is the correct file name
    if not os.path.exists(fmri_data_path):
        raise FileNotFoundError(f"Directory not found: {fmri_data_path}. Please check the path.")
    if not os.path.exists(fnirs_data_path):
        raise FileNotFoundError(f"File not found: {fnirs_data_path}. Please check the path and file name.")
    
    args.ele_dir = fnirs_data_path
    args.hemo_dir = fmri_data_path 
    fnirs_subs = torch.load(fnirs_data_path, weights_only=False)
    fnirs_subs = [x['pheno']['subjectId'] for x in fnirs_subs]
    fmri_subs = os.listdir(fmri_data_path)
    fmri_subs = [x[:6] for x in fmri_subs]
     
    train_dataloader = NumpyDataset(
        fnirs_file=args.ele_dir,
        fmri_folder=args.hemo_dir,
        split='train',
        n_way=args.n_way,
        fmri_sub_list=fmri_subs,
        fnirs_sub_list=fnirs_subs,
        single_subj=args.single_subj
    )

    test_dataloader = NumpyDataset(
        fnirs_file=args.ele_dir,
        fmri_folder=args.hemo_dir,
        split='test',  # this should be set to 'valid' we use test here for demo
        n_way=args.n_way,
        fmri_sub_list=fmri_subs,
        fnirs_sub_list=fnirs_subs,
        single_subj=args.single_subj
    )
      
     
    
    # build model   
    proto_model = str2model(args.model) 
    model = proto_model(args).to(args.device)  
    args.output_key = time.strftime(args.model+"--"+args.save_prefix+"--%Y%m%d-%H%M%S--" + args.dataset) 
    print_gpu_info(args)  
    #print the number of parameters
    print(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    training(
        args, 
        model, 
        train_dataloader, 
        test_dataloader,  
    )   
  
  
  
   
