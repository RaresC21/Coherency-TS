import torch
import torch.nn as nn
from coherency import *
from constants import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class RNN(nn.Module):
    def __init__(self, aggregate_mat, params):
        super(RNN, self).__init__()
        
        self.input_size = params['n_series']
        self.aggregate_mat = aggregate_mat
        self.hidden_size = params['hidden_size']
        self.params = params 

        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        self.batch_norm = nn.BatchNorm1d(self.hidden_size)
        self.last_layer = nn.Linear(self.hidden_size, self.input_size)
        
        self.dropout = nn.Dropout(p=0.2)

        if self.params.get('project', False):
            self.projector = Projection(aggregate_mat)

    def forward(self, x, get_hidden=False, test=False):
        batch_size, _, sequence_length = x.size()
        hidden = torch.zeros(batch_size, self.hidden_size).float().to(device)

        outputs = []
        for t in range(sequence_length):
            input_t = self.input_layer(x[:, :, t])
            embedding = torch.cat((input_t, hidden), 1)
            hidden = torch.tanh(self.hidden_layer(embedding))
            
            if 'dropout' in self.params:
                hidden = self.dropout(hidden)
        
        hidden = self.batch_norm(hidden)
        output = self.last_layer(hidden)    
        
        if self.params.get('project', False):
            return self.projector.project(output)
        
        if self.params.get('aggregate', False): 
            return self.aggregate(output)
        
        if get_hidden: 
            return hidden
        
        return output

    def aggregate(self, x): 
        return utils.aggregate(x, self.aggregate_mat)
    

class CNNTimeSeriesForecaster(nn.Module):
    def __init__(self, aggregate_mat, params):
    # def __init__(self, input_channels, seq_length, output_steps, k_series):
        super().__init__()
        input_channels = params['n_series']
        k_series = params['n_series']
        seq_length = params['context_window']
        output_steps = 1
        
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=16, 
            kernel_size=3, 
            padding=1  # Preserve sequence length
        )
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        # Calculate flattened dimension after convolutions and pooling
        reduced_length = seq_length // 4  # Two pooling layers (div by 2 each)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * reduced_length, 256)
        self.fc2 = nn.Linear(256, output_steps * k_series)
        self.output_steps = output_steps
        self.k_series = k_series

    def forward(self, x):
#         # Input shape: (batch_size, seq_length, input_channels)
#         x = x.permute(0, 2, 1)  # → (batch, channels, sequence)
        
        print(x)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Reshape to (batch, output_steps, k_series)
        return x.view(-1, self.output_steps, self.k_series)

    
    
# make gaussian forecast
class DistForecast(nn.Module):
    def __init__(self, aggregate_mat, params, project=False):
        super(DistForecast, self).__init__()
        self.project = project
        self.latent_dim = params['latent_dim']
        input_dim = params['n_series']
        hidden_dim = params['hidden_dim']        

        self.encoder_rnn = RNN(input_dim, hidden_dim, aggregate_mat, params)
        self.hidden = nn.Linear(self.latent_dim, hidden_dim)
        self.fc1 = nn.Linear(self.latent_dim, input_dim * 2)
    
        if self.project:
            self.projector = Projection(aggregate_mat)
    
    def reparameterize(self, mean, logvar):
        std = 0.5 * logvar.exp()
        eps = torch.randn_like(std)
        return eps * std + mean
    
    def forward(self, x, test=False):
        h = self.encoder_rnn(x, get_hidden=True)
        out = F.relu(self.hidden(h))
        out = self.fc1(out)
        
        mu, logvar = torch.chunk(out, 2, dim=-1)
        sample = self.reparameterize(F.relu(mu), logvar)
        
        if self.project:
            return self.projector.project(sample) 
        
        if test: return sample
        return sample, F.relu(mu), logvar

# generate distribution by dropout
class DropForecast(nn.Module):
    def __init__(self, aggregate_mat, params):
        super(DropForecast, self).__init__()

        params['dropout'] = True
        self.latent_dim = params['latent_dim']
        input_dim = params['n_series']
        hidden_dim = params['hidden_dim']        

        self.encoder_rnn = RNN(aggregate_mat, params)
        self.hidden = nn.Linear(self.latent_dim, hidden_dim)
        self.last_layer = nn.Linear(self.latent_dim, input_dim)
    
    def reparameterize(self, mean, logvar):
        std = 0.5 * logvar.exp()
        eps = torch.randn_like(std)
        return eps * std + mean
    
    def forward(self, x, test=False):
        h = self.encoder_rnn(x, get_hidden=True)
        out = F.relu(self.hidden(h))
        out = self.last_layer(out)
        return out
    
class VAEForecast(nn.Module):
    def __init__(self, aggregate_mat, params):
        super(VAEForecast, self).__init__()

        params['dropout'] = False
        self.latent_dim = params['latent_dim']
        input_dim = params['n_series']
        hidden_dim = params['hidden_dim']        

        self.encoder_rnn = RNN(aggregate_mat, params)
        self.hidden = nn.Linear(self.latent_dim, hidden_dim)
        self.last_layer = nn.Linear(self.latent_dim, input_dim)
    
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
    
    def reparameterize(self, mean):
        eps = torch.randn_like(mean)
        return eps + mean
    
    def forward(self, x, test=False):
        h = self.encoder_rnn(x, get_hidden=True)        
        out = self.reparameterize(h)
        
        out = F.relu(self.hidden(out))
        out = self.batch_norm(out) 
        
        return self.last_layer(out)