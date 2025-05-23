import torch
from torch.nn import MSELoss,L1Loss
import pandas as pd

class ScaledMSELoss(torch.nn.Module):
    def __init__(self, std):
        super(ScaledMSELoss, self).__init__()
        self.sigma = std

    def forward(self, preds, targets):
        return .5*torch.mean(((preds - targets)/self.sigma.to(preds.device)) ** 2)

class ScaledMAELoss(torch.nn.Module):
    def __init__(self, std):
        super(ScaledMAELoss, self).__init__()
        self.sigma = std

    def forward(self, preds, targets):
        return torch.mean(torch.abs(preds-targets)/self.sigma.to(preds.device))

def get_loss_fn(loss_fn):
    std = read_stats(fname='data/TriangleFeatures/target_statistics_train.csv', stat='Std', set_angles_1=True)
    loss_functions = {
        'MSE': MSELoss(reduction='mean'),
        'MAE': L1Loss(reduction='mean'),
        'ScaledMAE': ScaledMAELoss(torch.tensor(std)),
        'ScaledMSE': ScaledMSELoss(torch.tensor(std)),
    }
    try:
        return loss_functions[loss_fn]
    except KeyError:
        raise NotImplementedError('Invalid loss function.')

def read_stats(fname, stat,set_angles_1=True):
    df = pd.read_csv(fname)
    matching_row = df[df['Statistic'] == stat]

    if matching_row.empty:
        print(f"No row with Statistic='{stat}' found in {fname}")
        return None

    result = matching_row.iloc[0].drop('Statistic').to_dict()
    if set_angles_1:
        result['CosAngle+1'] = 1.
        result['SinAngle+1'] = 1.

    return list(result.values())