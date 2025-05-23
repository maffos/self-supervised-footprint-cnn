import torch
from torch.nn import MSELoss,L1Loss, CrossEntropyLoss
import numpy as np
class ClassWeightedMAELoss(torch.nn.Module):
    def __init__(self, class_weight=3.5813,label=2):
        super(ClassWeightedMAELoss, self).__init__()
        self.weight = class_weight
        self.class_label = label
    def forward(self, preds, targets,labels):
        error = torch.abs(preds-targets)
        label_mask = torch.where(labels==self.class_label,self.weight,1)
        return torch.mean(error*label_mask)

def get_loss_fn(loss_fn, device = None, **kwargs):
    class_weights_inverse = torch.tensor([0.4602,0.1912,0.3486])
    loss_functions = {
        'MSE': MSELoss(reduction='mean'),
        'MAE': L1Loss(reduction='mean'),
        'CrossEntropy': CrossEntropyLoss(**kwargs),
        'WeightedCrossEntropy': CrossEntropyLoss(weight=class_weights_inverse.to(device)),
        'ClassWeightedMAE': ClassWeightedMAELoss(),
    }
    try:
        return loss_functions[loss_fn]
    except KeyError:
        raise NotImplementedError('Invalid loss function.')

def automatic_weight(weights, task_loss):
    """
    It is adapted from https://github.com/Mikoto10032/AutomaticWeightedLoss.git
    The orginal paper is: Auxiliary tasks in multi-task learning
    """

    total_loss = 0
    for i in range(len(task_loss)):

        total_loss += 0.5 / (weights[i] ** 2) * task_loss[i] + torch.log1p(weights[i] ** 2)
    return total_loss

def print_results(results):
    """
    Prints a results dictionary in a neat, readable format.
    Supports nested dictionaries.
    """
    def print_nested(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print(" " * indent + f"{key}:")
                print_nested(value, indent + 4)
            else:
                print(" " * indent + f"{key}: {value}")

    print("Results:")
    print_nested(results)

def convert_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):  # Handles numpy scalars like np.float64
        return obj.item()
    else:
        return obj

def initialize_weights(model, weights_fname, device):
    chkpt = torch.load(weights_fname, map_location=device)
    weights = chkpt['model_state_dict']
    try:
        model.triangle_embedding.load_state_dict(weights, strict=False)
    except:
        with torch.no_grad():
            model.triangle_embedding.out_layer.weight.copy_(weights["mlp.8.weight"])
            if model.triangle_embedding.out_layer.bias is not None:
                model.triangle_embedding.out_layer.bias.copy_(weights["mlp.8.bias"])
    return model
