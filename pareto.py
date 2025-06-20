import pandas as pd
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
import random
import argparse
import os

import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import preprocessing


import utils
from models import * 
from Experiments import *
from informer import *

from scipy import stats

def set_seeds(seed): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 


def parse_args():    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--dataset', type=str, help='Choice of "traffic", "tourism", "labor", "m5"')
    
    args = parser.parse_args()
    return args.dataset

def load_config(dataset, config_file='config.yaml'):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config.get("params", {})

def plot(results, metrics, get, model_name, color="black"):
    mean_res = get_mean(results, metrics)    
    plt.plot(range(mean_res.shape[0]), mean_res[get], label=model_name, color=color)

def plot_results(GET):

    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('{} by hierarchy level'.format(GET), fontsize=16)
    plt.xlabel('Hierarchy level', fontsize=14)
    plt.ylabel(GET, fontsize=14)

    plot(base_results, metrics, GET, "base model", color='red')
    plot(projection_results, metrics, GET, "projection model", color='blue')
    plot(coherency_results, metrics, GET, "coherency loss model", color='green')
    plot(profhit_results, metrics, GET, "profhit model", color='orange')

    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/{}_{}.png'.format(dataset_name, GET))

def plot_all_results(): 
    for get in metrics:
        plot_results(get)
    
def mean_mse(df): 
    return (df.values[0] + df.values[1] * 10) / 11
    
if __name__ == "__main__":
    
    dataset_name = parse_args()
    data, base_agg_mat = utils.load_data(dataset_name)
    params = load_config(dataset_name)
    
    # model_type = InformerModel
    model_type = BaseModel
    
    # print("Base model")
    # base_results, metrics, base_losses       = repeat_exp(model_type, base_agg_mat, data, params)
    # mean_base = get_mean(base_results, metrics)
    # print(mean_base)
    # plt.scatter(mean_base["Coherency"].mean(), mean_base["MSE"].mean(), color='black', label='base')
    # plt.scatter(mean_base["Coherency"].mean(), mean_mse(mean_base["MSE"]), color='black', label='base')

    # print("Projection model")
    # projection_results, _, projection_losses = repeat_exp(model_type, base_agg_mat, data, params, project=True)
    # mean_proj = get_mean(projection_results, metrics)
    # # plt.scatter(mean_proj["Coherency"].mean(), mean_proj["MSE"].mean(), color='red', label='proj')
    # plt.scatter(mean_proj["Coherency"].mean(), mean_mse(mean_proj["MSE"]), color='red', label='proj')

    coh_weights = np.concatenate((np.arange(0, 1e-4, 2e-5), np.arange(0, 1e-3, 2e-4), np.arange(0, 1e-2, 2e-3), np.arange(0, 1e-1, 2e-2))) 
    all_coherency_results = []
    for i, w in enumerate(coh_weights):
        set_seeds(0)
        
        print("CoRE model", w)
        params['coherency_weight'] = w
        coherency_results, metrics, coherency_losses  = repeat_exp(model_type, base_agg_mat, data, params, coherency_loss=True, cv_=False)
        mean_core = get_mean(coherency_results, metrics)
        

        if i == 0:
            plt.scatter(mean_core["Coherency"].mean(), mean_core["MSE"].mean(), color='green', label='Core')
            plt.scatter(mean_core["val_Coherency"].mean(), mean_core["val_MSE"].mean(), color='red', alpha=0.5, label='(val) Core')
        else:
            plt.scatter(mean_core["val_Coherency"].mean(), mean_core["val_MSE"].mean(), color='red', alpha=0.5)
            plt.scatter(mean_core["Coherency"].mean(), mean_core["MSE"].mean(), color='green')# alpha=i/len(coh_weights))
        # plt.scatter(mean_core["Coherency"].mean(), mean_core["MSE"].mean(), color='green')
        
    prof_weights = np.concatenate((np.arange(0, 1e-4, 2e-5), np.arange(0, 1e-3, 2e-4), np.arange(0, 1e-2, 2e-3), np.arange(0, 1e-1, 2e-2)))
    all_profhit_results = []
    params['lr'] = 1e-3
    for i, w in enumerate(prof_weights):
        set_seeds(0)
        
        print("PROFHiT model", w)
        params['coherency_weight'] = w
        profhit_results, metrics, profhit_losses    = repeat_exp(model_type, base_agg_mat, data, params, profhit_loss=True, cv_=False)
        mean_profhit = get_mean(profhit_results, metrics)
        
        if i == 0:
            plt.scatter(mean_profhit["Coherency"].mean(), mean_profhit["MSE"].mean(), color='blue', label='PROFHiT')
            plt.scatter(mean_profhit["val_Coherency"].mean(), mean_profhit["val_MSE"].mean(), color='orange', label='(val) PROFHiT')
        else:
            plt.scatter(mean_profhit["Coherency"].mean(), mean_profhit["MSE"].mean(), color='blue')
            plt.scatter(mean_profhit["val_Coherency"].mean(), mean_profhit["val_MSE"].mean(), color='orange')
            # plt.scatter(mean_profhit["Coherency"].mean(), mean_profhit["MSE"].mean(), color='blue', alpha=i/len(prof_weights))
    
    plt.xlabel('Coherency')
    plt.ylabel('MSE')
    plt.title("MSE vs. Coherency Pareto Frontier ({})".format(dataset_name))
    plt.legend()
    
    plt.savefig('plots/pareto_{}.png'.format(dataset_name))