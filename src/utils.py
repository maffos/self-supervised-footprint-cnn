import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import GraphNorm
from torch_geometric.nn.conv import GCNConv,SAGEConv,SplineConv
import os
import shutil
import csv
import yaml

def get_norm_layer(norm,num_channels, num_groups = 32, channels_first = True):
    if norm == 'batch':
        return nn.BatchNorm1d(num_channels)
    elif norm == 'group':
        if num_groups is None:
            raise ValueError("num_groups must be specified for group normalization.")
        return nn.GroupNorm(num_groups, num_channels)
    elif norm is None:
        return nn.Identity()
    else:
        raise NotImplementedError('Only GroupNorm and BatchNorm are supported.')

def get_graph_norm_layer(norm,num_channels):
    if norm == 'graph':
        return GraphNorm(num_channels)
    elif norm is None:
        return nn.Identity()
    else:
        raise NotImplementedError('Only GraphNorm is supported for Graphs so far.')

def get_graph_conv_layer(operator, in_channels, out_channels, **kwargs):
    if operator == 'gcn':
        return GCNConv(in_channels, out_channels,**kwargs)
    elif operator == 'spline_conv':
        return SplineConv(in_channels, out_channels,**kwargs)
    elif operator == 'sage_conv':
        return SAGEConv(in_channels,out_channels,**kwargs)
    else:
        raise NotImplementedError(f"Unknown operator {operator}")

def get_activation(activation, relu_slope=0.1):
    activations = {
        'leakyrelu': nn.LeakyReLU(relu_slope, inplace=True),
        'relu': nn.ReLU(inplace=True),
        'tanh': nn.Tanh()
    }
    try:
        return activations[activation.lower()] if activation else nn.Identity()
    except KeyError:
        raise NotImplementedError('Invalid activation function.')

def setup_directories(*dirs):
    """Ensure necessary directories for logs and outputs exist."""
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)

def save_config(out_dir, config):
    """Save the configuration parameters to a YAML file for reproducibility."""
    config_path = os.path.join(out_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

def save_checkpoint(epoch, model, optimizer, file_path, scheduler=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **({'scheduler_state_dict': scheduler.state_dict()} if scheduler else {}),
    }
    torch.save(checkpoint, file_path)

def save_results_to_csv(file_path, results):

    with open(file_path, "w") as f:
        writer = csv.writer(f)
        for key, value in results.items():
            if key != 'residuals':
                writer.writerow([key, value])

def initialize(model, params, tag, model_name, run = None):
    optimizer = Adam(
        params=model.parameters(),
        lr=params['lr'], weight_decay=params['weight_decay']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=params['num_epochs'],
                                  eta_min=params['lr_min'])
    base_dir = os.path.join('runs', tag, model_name)
    if run is not None:
        out_dir = os.path.join(base_dir, 'run_%d' % run)
    else:
        out_dir = base_dir
    log_dir = os.path.join(out_dir, 'logs')

    # delete previous model log directory to start fresh
    if not params.get('resume_training', False) and os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    return optimizer,scheduler,out_dir,log_dir,base_dir