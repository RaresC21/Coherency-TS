import pandas as pd
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
import seaborn as sns
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


def parse_args():    
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('-d', '--dataset', type=str, help='Choice of "traffic", "tourism", "labor"')
    
    args = parser.parse_args()
    return args.dataset

def load_config(dataset, config_file='distribution_config.yaml'):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config.get("params", {})

def plot(results, metrics, get, model_name, color="black"):
    mean_res = get_mean(results, metrics)    
    plt.plot(range(mean_res.shape[0]), mean_res[get], label=model_name, color=color)

def plot_results(GET, base_results, coherency_results, modeltype):
    # DROPOUT

    # Set plot style
    sns.set(style="whitegrid", palette="pastel")

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Define line styles and markers
    plot_params = {
        'base': {'color': '#E24A33', 'linestyle': '-', 'marker': 'o', 'markersize': 4},
        'coherency': {'color': 'green', 'linestyle': '--', 'marker': 's', 'markersize': 4}
    }

    # First plot
    base_line1 = ax1.plot(base_results[:, :, 0].T, **plot_params['base'])
    coherency_line1 = ax1.plot(coherency_results[:, :, 0].T, **plot_params['coherency'])
    ax1.set_title(metrics[0], fontsize=12, pad=15)
    ax1.set_xlabel("Hierarchy Level", fontsize=10)
    # ax1.set_ylabel("Value", fontsize=10)

    # Second plot
    ax2.plot(base_results[:, :, GET].T, **plot_params['base'])
    ax2.plot(coherency_results[:, :, GET].T, **plot_params['coherency'])
    ax2.set_title(metrics[GET], fontsize=12, pad=15)
    ax2.set_xlabel("Hierarchy Level", fontsize=10)
    # ax2.set_ylabel("Value", fontsize=10)

    # Global adjustments
    plt.suptitle("Variance across training runs", y=1.02, fontsize=14)
    plt.tight_layout()

    # Add legend outside the subplots
    plt.legend(handles=[base_line1[0], coherency_line1[0]], labels=['Base Results', 'Coherency Results'], 
               loc='lower center', bbox_to_anchor=(-0.1, 1.1), ncol=2, frameon=True)

    # Adjust the layout to make room for the legend
    plt.subplots_adjust(bottom=0.2)
    # plt.legend()
    plt.savefig('plots/{}_{}_variance.png'.format(metrics[GET], dataset_name))
    # plt.show()

if __name__ == "__main__":
    
    
    dataset_name = parse_args()
    data, base_agg_mat = utils.load_data(dataset_name)
    params = load_config(dataset_name)
    
    base_dropout, metrics, base_drop_losses       = repeat_exp(DropoutDistribution, base_agg_mat, data, params)
    base_vae,           _, base_vae_losses        = repeat_exp(VAEDistribution, base_agg_mat, data, params)
    
    coherency_dropout,  _, coherency_drop_losses  = repeat_exp(DropoutDistribution, base_agg_mat, data, params, coherency_loss=True)
    coherency_vae,      _, coherency_vae_losses   = repeat_exp(VAEDistribution, base_agg_mat, data, params, coherency_loss=True)
    
    for i in range(1,4):
        plot_results(i, base_dropout, coherency_dropout, "Dropout")
        plot_results(i, base_vae, coherency_vae, "VAE")