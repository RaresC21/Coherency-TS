from abc import ABC, abstractmethod
import coherency
from utils import * 
from constants import * 
from models import *

import pandas as pd

from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import tqdm


def validation_loss(model, X_val, y_val):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculations
        pred = model(X_val)
        return calculate_wmape(y_val, pred).item()

class Metrics:
    def __init__(self, aggregation_mat): 
        self.aggregation_mat = aggregation_mat
        self.metrics = []
        self.names = []
    def run_metrics(self, network, X, y): 
        # network.eval()
        pred = network(X)
        
        results = pd.DataFrame()
        metrics = [getattr(self, method) for method in dir(self) if callable(getattr(self, method)) and not method.startswith("__") and not method.startswith("run_metrics")]
        for metric in metrics:
            res, name = metric(network, pred, y)
            results[name] = res
        return results

    def mse_by_level(self, _, pred, y): 
        subtree_size = self.aggregation_mat.sum(dim=1)
        sizes = torch.unique(subtree_size)
        mse_by_level = []
        for s in sizes: 
            keep = subtree_size == s
            err = mse_loss(y.to(device)[:, keep].float(), pred[:, keep].to(device).float())
            mse_by_level.append(err.item())
        return mse_by_level, "MSE"
    
    def wmape_by_level(self, _, pred, y): 
        subtree_size = self.aggregation_mat.sum(dim=1)
        sizes = torch.unique(subtree_size)
        wmape_by_level = []
        for s in sizes: 
            keep = subtree_size == s
            err = calculate_wmape(y.to(device)[:, keep].float(), pred[:, keep].to(device).float())
            wmape_by_level.append(err.item())
        return wmape_by_level, "WMAPE"
    
    def coherency_levels(self, _, pred, y): 
        coherency_loss = torch.abs(coherency_metric(pred, self.aggregation_mat)).mean(dim=1)
        subtree_size = self.aggregation_mat.sum(dim=1)
        sizes = torch.unique(subtree_size)
        coh = []
        for s in sizes: 
            keep = subtree_size == s
            c = coherency_loss[keep].mean()
            coh.append(c.item())
        return coh, "Coherency"
    
def train(network, data_loader, X_val, y_val, params, aggregation_mat=None):
    network.train()
    early_stopping = EarlyStopping(patience=100, verbose=False)

    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=params['lr'])

    n_epochs = params['n_epochs']
    losses = [1]
    c_losses = [1]
    val_losses = []
    for epoch in tqdm.tqdm(range(n_epochs)):  # loop over the dataset multiple times
        inputs, targets = next(iter(data_loader))

        optimizer.zero_grad()
        network.train()

        inputs = inputs
        targets = targets
        
        pred = network(inputs.float())
        loss = mse_loss(pred, targets.float())
        
        if params['coherency_loss']:
            c_loss = coherency.coherency_loss(network, aggregation_mat)
        elif params['profhit']:
            c_loss = coherency.coherency_data_loss(pred, aggregation_mat)
        else: 
            c_loss = torch.tensor([0])
            
        total_loss = loss * np.mean(c_losses) / np.mean(losses) + c_loss * params['coherency_weight']

        total_loss.backward() 
        optimizer.step()
        
        losses.append(loss.item())
        c_losses.append(c_loss.item())
        
        val_loss = validation_loss(network, X_val, y_val)
        early_stopping(val_loss, network)
  
        val_losses.append(val_loss)
    
        # print(epoch, loss.item(), val_loss, c_loss.item())

    # network.load_state_dict(torch.load('checkpoint.pt'))
    return losses, c_losses, val_losses

#     if params['coherency_loss'] or params['profhit']:
#         early_stopping = EarlyStopping(patience=100, verbose=False)
#         c_losses = [1]
#         for epoch in tqdm.tqdm(range(n_epochs)):
#             inputs, targets = next(iter(data_loader))

#             optimizer.zero_grad()

#             inputs = inputs.to(device) 
#             targets = targets.to(device) 

#             pred = network(inputs.float())
#             loss = mse_loss(pred, targets.float())
            
#             if params['coherency_loss']:
#                 c_loss = coherency.coherency_loss(network, aggregation_mat.to(device).float())
#             elif params['profhit']:
#                 c_loss = coherency.coherency_data_loss(pred, aggregation_mat.to(device).float())
                        
#             total_loss = loss * np.mean(c_losses) / np.mean(losses) + c_loss * params['coherency_weight']
#             total_loss.backward() 
#             optimizer.step()

#             losses.append(loss.item())
#             c_losses.append(c_loss.item())
            
#             val_loss = validation_loss(network, X_val, y_val)

#             early_stopping(val_loss, network)
#             val_losses.append(val_loss)
# #             if early_stopping.early_stop:
# #                 print("Early stopping")
# #                 network.load_state_dict(torch.load('checkpoint.pt'))
# #                 return
        
#         network.load_state_dict(torch.load('checkpoint.pt'))
#         return losses, c_losses, val_losses
#     else:
#         return losses, val_losses

class Experiment(ABC): 
    def __init__(self, aggregation_mat, params = {'lr':1e-3, 'n_epochs':400}):
        self.aggregation_mat = torch.tensor(aggregation_mat).float().to(device)
        self.full_agg = format_aggregation_matrix(aggregation_mat).float().to(device)
        self.params = params
        self.train_split = self.params['train_split']
        self.val_split = self.params['val_split']
        self.context_window = self.params['context_window']
        self.batch_size = self.params['batch_size']
        
        self.network = None 
        
    def run(self, data):
        X_train, y_train, X_val, y_val, X_test, y_test = self.make_data(data)
        
        train_dataloader = DataLoader(TensorDataset(X_train.to(device).float(), y_train.to(device).float()), batch_size=self.batch_size, shuffle=True)
        val_dataloader   = DataLoader(TensorDataset(X_val.to(device).float()  , y_val.to(device).float()  ), batch_size=self.batch_size, shuffle=True)
        test_dataloader  = DataLoader(TensorDataset(X_test.to(device).float() , y_test.to(device).float() ), batch_size=self.batch_size, shuffle=True)

        losses = train(self.network, train_dataloader, X_val.to(device).float(), y_val.to(device).float(), self.params, self.full_agg)

        metrics = Metrics(self.full_agg)
        return metrics.run_metrics(self.network, X_test.to(device).float(), y_test.to(device).float()), losses
        # return metrics.run_metrics(self.network, X_train.to(device).float(), y_train.to(device).float()), losses
        
        
    @abstractmethod
    def make_data(self, data): 
        pass
        
        
        
class BaseModel(Experiment): 
    def __init__(self, aggregation_mat, params): 
        super().__init__(aggregation_mat, params)
        
        network = RNN(params['n_series'], params['hidden_size'], self.aggregation_mat.to(device), self.params)
        self.network = network.to(device)
        
    def make_data(self, data): 
        return get_data(data, self.train_split, self.val_split, self.context_window)

def repeat_noise_exp(model_class, base_agg_mat, data, params, n_runs):
    results = []
    for i in range(n_runs):
        np.random.seed(i)
        print("run", i)
        model = model_class(base_agg_mat, params)
        res, losses = model.run(utils.add_noise(data.copy(), params['noise']))
        results.append(res.values)
    return np.array(results), res.columns, losses
    
def repeat_exp(model_class, base_agg_mat, data, params, n_runs):
    results = []
    for i in range(n_runs):
        print("run", i)
        model = model_class(base_agg_mat, params)
        res, losses = model.run(data)
        results.append(res.values)
    return np.array(results), res.columns, losses

def get_mean(results, names): 
    return pd.DataFrame(columns=names, data=results.mean(axis=0))
def get_std(results, names): 
    return pd.DataFrame(columns=names, data=results.std(axis=0))