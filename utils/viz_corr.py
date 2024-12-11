import os 
import torch
import numpy as np
from scipy import stats
import matplotlib.cm as cm 
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, median_absolute_error

 
def correlation_r2_plot(args, iteration, x, y, heat_map = False, pwd=None, title = 'Real (fMRI) vs Predicted (MEG-to-fMRI)', folder_extention=''):
    plt.figure(figsize=(10, 10))  
    
    
    zero_indices = (x == 0) 
     
    x = x[~zero_indices]
    y = y[~zero_indices]    
    
    rmse = np.sqrt(mean_squared_error(x, y))
    rmse = round(rmse, 4) 
    
    medae = median_absolute_error(x, y)
    medae = round(medae, 4) 
       
    spearman_corr, _ = stats.spearmanr(x.flatten(), y.flatten())  
    spearman_corr = round(spearman_corr, 4)
    lm = LinearRegression() 
    lm.fit(x.flatten().reshape(-1, 1), y.flatten().reshape(-1, 1)) 
    y_pred = lm.predict(x.flatten().reshape(-1, 1)) 
    r2 = r2_score(y.flatten().reshape(-1, 1), y_pred)
    r2 = round(np.mean(r2), 4) 
  
  
    plt.scatter(x, y, color='blue', marker='.', s=10,  alpha=0.07) 

    plt.text(0.05, 0.95, 'spearman corr: ' + str(spearman_corr) +'\nr2: ' + str(r2) + '\nMSE: ' + str(rmse) +'\nMedSE: ' + str(medae),  
            fontsize=15,
            horizontalalignment='left',
            verticalalignment='top',
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='yellow', alpha=0.2))  # yellow highlight
     
    plt.title(title, fontsize=15)
    plt.xlabel('Real', fontsize=15)
    plt.ylabel('Prediction', fontsize=15)    
    # plt.xticks([])  
    # plt.yticks([])  
     
 
    save_path = args.output_dir + '/corr/correlation_r2_plot_'+folder_extention+'/'
    if not os.path.exists(save_path): 
        os.mkdir(save_path)    
    
    plt.savefig(save_path+'correlation_r2_plot_'+str(iteration)+'.png', dpi=400)   
    plt.close()
    
    
    
def twoD_plot(x, x_hat, pwd=None, viz_method='phate', title = 'phate visualization of fMRI in cricle and MEG-to-fMRI in star '): 
    # plt.figure(figsize=(10, 10)) 
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))  
    # plt.scatter(x[:, :], x_hat[:,:], color='red',  alpha=0.5)
    colormap = cm.get_cmap("inferno", x.shape[1])  
    colors = [colormap(i) for i in range(colormap.N)]
      
    x, x_hat = torch.tensor(x), torch.tensor(x_hat)
    x = rearrange(x, 'd b -> b d')
    x_hat = rearrange(x_hat, 'd b -> b d')
    batch_size = x.shape[0]
    x_viz = torch.cat((x, x_hat), dim=0).numpy()
    
    if viz_method == 'phate': 
        z_viz = phate.PHATE(verbose = False).fit_transform(x_viz)
    elif viz_method == 'pca': 
        z_viz = PCA(2).fit_transform(x_viz)  
    
    viz_x = z_viz[:batch_size, :]
    viz_x_hat = z_viz[batch_size:, :] 
    
    for i in range(viz_x.shape[0]): 
        axs[0].scatter(viz_x[i, 0], viz_x[i, 1], marker="o", s=150, color=colors[i], alpha=0.6, linewidth=1, edgecolor='black')  
        axs[1].scatter(viz_x_hat[i, 0], viz_x_hat[i, 1], marker="*", s=220, color=colors[i], alpha=0.6, linewidth=1, edgecolor='black')  
     
    axs[0].set_title('Phate: fMRI')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('') 
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    
    axs[1].set_title('Phate: MEG-to-fMRI')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('') 
    axs[1].set_xticks([])
    axs[1].set_yticks([]) 
     
    path = os.path.join(pwd)
    plt.savefig(path, dpi=200) 
    plt.close()
     
     
def psi_plot(psi, pwd=None, title = 'Mother Wavelet (Ψ) | mean: '): 
    plt.figure(figsize=(15, 10)) 
    psi = psi[0,0,:].cpu().detach().numpy()       
    plt.plot(psi, linewidth=2, color='black') 
    rounded_number = round(np.mean(psi), 3)
    plt.title(title + str(rounded_number), fontsize=40)
    # plt.xlabel()
    # plt.ylabel()    
    plt.xticks([])  
    plt.yticks([])  
    path = os.path.join(pwd)
    plt.savefig(path, dpi=100) 
    plt.close()