import torch
import torch.nn as nn
from coherency import *
from constants import *

import torch
import torch.nn as nn
import utils

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, aggregate_mat, params):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.aggregate_mat = aggregate_mat
        self.params = params 

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size * 2, hidden_size)
        
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.last_layer = nn.Linear(hidden_size, input_size)

        if self.params['project']:
            self.projector = Projection(aggregate_mat)

    def forward(self, x, get_hidden=False):
        batch_size, _, sequence_length = x.size()
        hidden = torch.zeros(batch_size, self.hidden_size).float().to(device)

        outputs = []
        for t in range(sequence_length):
            input_t = self.input_layer(x[:, :, t])
            embedding = torch.cat((input_t, hidden), 1)
            hidden = torch.tanh(self.hidden_layer(embedding))
        
        hidden = self.batch_norm(hidden)
        output = self.last_layer(hidden)    
        
        if self.params['project']:
            return self.projector.project(output)
        
        if self.params['aggregate']: 
            return self.aggregate(output)
        
        if get_hidden: 
            return hidden
        
        return output

    def aggregate(self, x): 
        # x = utils.get_bottom_level(x, self.aggregate_mat)
        return utils.aggregate(x, self.aggregate_mat)