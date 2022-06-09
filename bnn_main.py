# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 17:32:07 2022

@author: jhazelde
"""

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import os
import yaml

from lfads import LFADS_Net
from utils import read_data, load_parameters, save_parameters


from config import CFG
from bnn import HH_Gap
from torchviz import make_dot


def shutoff_stuff():
    def shutoff_test():
        CFG.sim_t = 4000
        bnn = HH_Gap(1).cuda()
        w = torch.ones(()) * 1.0
        w = w.cuda()
        w.requires_grad = True
        optim = torch.optim.Adam([w], lr=0.1)
        I = torch.ones((1, CFG.sim_t, 1)).cuda()
        for epoch in range(100):
            optim.zero_grad()
            if epoch > 0:
                print(w.grad.cpu().item())
            z = I * w
            out = bnn(z).squeeze()
            plt.plot(out.detach().cpu())
            plt.title(f'{w.cpu().item()}')
            plt.show()
            loss = torch.mean(out)
            loss.backward()
     #       make_dot(loss).render("loss", format="png")
    
            optim.step()
            print(epoch, w.item(), w.grad.cpu().item())
            
    def shutoff_sweep():
        CFG.sim_t = 4000
        bnn = HH_Gap(1).cuda()
        CFG.plot = False
        ws = np.linspace(0.9000104069709778, 0.9000104069709778+0.0001, 20)
        I = torch.ones((1, CFG.sim_t, 1)).cuda()
        losses = np.zeros(len(ws))
        for idx, w in enumerate(ws):
            print(idx)
            z = I * w
            out = bnn(z).squeeze()
            # plt.plot(out.detach().cpu())
            # plt.title(f'{w.cpu().item()}')
            # plt.show()
            loss = torch.mean(out)
            losses[idx] = loss
            if idx > 0:
                print((losses[idx] - losses[idx-1]) / (ws[idx] - ws[idx-1]))
        plt.plot(ws, losses)
        plt.show()        
    shutoff_sweep()

# plt.style.use('dark_background')

device = 'cuda' if torch.cuda.is_available() else 'cpu'; print('Using device: %s'%device)
Ncells = 50
if False and os.path.exists('./synth_data/chaotic_rnn_300'):
    data_dict = read_data('./synth_data/chaotic_rnn_300')
else:
    if not os.path.isdir('./synth_data'):
        os.mkdir('./synth_data/')
    
    from synth_data_chaotic_rnn import generate_data
    data_dict = generate_data(T= 0.3, dt_rnn= 0.01, dt_cal= 0.01,
                              Ninits= 400, Ntrial= 10, Ncells=Ncells, trainp= 0.8,
                              tau=0.025, gamma=1.5, maxRate=30, B=20,
                              seed=300, save=True)

train_data = torch.Tensor(data_dict['train_spikes']).to(device)
valid_data = torch.Tensor(data_dict['valid_spikes']).to(device)

print(train_data.shape)

train_truth = torch.Tensor(data_dict['train_rates']).to(device)
valid_truth = torch.tensor(data_dict['valid_rates']).to(device)

train_ds      = torch.utils.data.TensorDataset(train_data)
valid_ds      = torch.utils.data.TensorDataset(valid_data)

num_trials, num_steps, num_cells = train_data.shape
print(train_data.shape)

plt.figure(figsize = (12,12))
plt.imshow(data_dict['train_spikes'][0].T, cmap=plt.cm.Greys)
plt.xticks(np.linspace(0, 100, 6), ['%.1f'%i for i in np.linspace(0, 1, 6)])
plt.xlabel('Time (s)')
plt.ylabel('Cell #')
plt.colorbar(orientation='horizontal', label='# Spikes in 0.01 s time bin')
plt.title('Example trial')
plt.show()

plt.figure(figsize = (12,12))
plt.imshow(data_dict['train_rates'][0].T, cmap=plt.cm.plasma)
plt.xticks(np.linspace(0, 100, 6), ['%.1f'%i for i in np.linspace(0, 1, 6)])
plt.xlabel('Time (s)')
plt.ylabel('Cell #')
plt.colorbar(orientation='horizontal', label='Firing Rate (Hz)')
plt.title('Example trial')
plt.show()

T = 0.3

def fitting_code():
    def compare_rates(r, out, k, plot=False):
        true_rates = torch.mean(r, 1).cuda().float()
        true_rates = (true_rates - torch.mean(true_rates)) / torch.std(true_rates)
        
        # spike_duration = 10 # Assume spikes take 0.5 ms
        # pred_rates = torch.sum(out[0, :, :], 0).float() / spike_duration
        # pred_rates = pred_rates / T
        
        pred_rates = torch.sum(out[0, :, :], 0).float()
        pred_rates = (pred_rates - torch.mean(pred_rates)) / torch.std(pred_rates)
        
        loss_fun = torch.nn.MSELoss()
        loss = loss_fun(pred_rates, true_rates) / 1000
        
    #    loss = torch.mean(out**2)
            
        if plot:
            plt.figure(dpi=150)
            plt.plot(true_rates.cpu().detach().numpy(), zorder=5, color='black', linewidth=1)
            plt.plot(pred_rates.cpu().detach().numpy(), zorder=5, color='red', linewidth=1)
            plt.title(f'k={k:.5f}, loss={loss.cpu().item():.5f}')
            
        return loss
    
    r = torch.from_numpy(data_dict['train_rates']).cuda()
    interp = torchvision.transforms.Resize((CFG.sim_t, Ncells))
    r = interp(r)
    
    def k_grad_sweep():
        CFG.plot = False
        bnn = HH_Gap(Ncells)
        
        k = torch.ones(()).float() * 0.1
        k.requires_grad = True
        
        ks = np.zeros(100)
        losses = np.zeros(len(ks))
        optim = torch.optim.SGD([k], lr=0.01)
        grads = np.zeros(len(ks))
        
        for idx in range(len(ks)):
            optim.zero_grad()
            CFG.dt = 0.1
            CFG.sim_t = int((T * 1000) / CFG.dt)
    
            r_sub = r[:10, :, :]
            z = torch.clamp(k * r_sub, 0.0, 1.8)
        
            out = bnn(z)
            # plt.imshow(out[0, :, :].cpu().detach().numpy(), aspect='auto')
            # plt.show()
            
            plt.plot(out[-1, :, 0].cpu().detach().numpy())
            plt.show()
        
            loss = compare_rates(r_sub, out, k, True)
            plt.show()    
            losses[idx] = loss.cpu().item()
            ks[idx] = k.cpu().item()
            
            loss.backward()
            grad = k.grad
            grads[idx] = grad.cpu().item()
            print(idx, losses[idx], k.item(), grad.item())
            optim.step()
            
        #plt.plot(ks, losses)
        #plt.savefig(f'loss_gradient_find_2.pdf')
        #plt.show()
    k_grad_sweep()
        
    def k_simple_sweep(a, b, N=10):
        CFG.plot = False
        bnn = HH_Gap(Ncells)
    
        ks = np.linspace(a, b, N)
        losses = np.zeros(len(ks))
        
        slopes = []
        for idx, k in enumerate(ks):
            CFG.dt = 0.1
            CFG.sim_t = int((T * 1000) / CFG.dt)
            z = torch.clamp(k * r, 0.0, 1.8)
            out = bnn(z)
            
            plt.plot(out[-1, :, 0].cpu().detach().numpy())
            plt.show()
            
            loss = compare_rates(r, out, k, True)
            plt.show()
            losses[idx] = loss.cpu().item()
    
            print(idx, losses[idx], k)
            if idx > 0:
                slope = (losses[idx] - losses[idx-1]) / (ks[idx] - ks[idx-1])
                print(f'slope k={k}, {slope:.3f}')
                slopes.append(slope)
            
        plt.plot(ks, losses)
        for i in range(len(slopes)):
            plt.text((ks[i] + ks[i+1]) * 0.5, (losses[i] + losses[i+1]) * 0.5, f'{slopes[i]:.1f}', size='small')
            
        plt.ylabel('Loss')
        plt.title('Noisy Loss Landscape Smoother')
        plt.xlabel('k')
        plt.savefig(f'loss_simple_noise_sweep_{a}_{b}_{N}.pdf')
        plt.show()  
        return ks, losses
        
    k = 0.0001
    def compare_rates_old(r, out, k, plot=False):
        true_rates = torch.mean(r[0, :, :], 0).cuda().float()
        true_rates = (true_rates - torch.mean(true_rates)) / torch.std(true_rates)
        
        pred_rates = torch.sum(out[0, :, :], 0).float()
        pred_rates = (pred_rates - torch.mean(pred_rates)) / torch.std(pred_rates)
            
        plt.figure(dpi=150)
        plt.plot(true_rates.cpu().detach().numpy(), zorder=5, color='black', linewidth=1, label='True Rates')
        plt.plot(pred_rates.cpu().detach().numpy(), zorder=5, color='red', linewidth=1, label='BNN Output Rates')
        plt.title('Before Training')
        plt.xlabel('Cell #')
        plt.yticks([])
        plt.legend()
        
    CFG.dt = 0.1
    CFG.sim_t = int((T * 1000) / CFG.dt)
    bnn = HH_Gap(Ncells).cuda()
    z = torch.clamp(k * r, 0.0, 1.8)
    out = bnn(z)
    compare_rates_old(r, out, k, True)
    plt.show()
fitting_code()

