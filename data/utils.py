import torch
from torch.functional import F
import numpy as np

def swap_channels(tensor: torch.tensor) -> torch.tensor:

    if tensor.ndim < 2:
        raise ValueError("Tensor must have at least two dimensions to swap the last two.")

        # Generate a permutation order for the dimensions
    perm_order = list(range(tensor.ndim - 2)) + [tensor.ndim - 1, tensor.ndim - 2]

    # Permute the tensor
    return tensor.permute(*perm_order)

def zero_padding(tensor, target_shape):

    assert tensor.ndim == len(target_shape) and tensor.ndim <= 3 and tensor.ndim >= 2,\
        'Either target shape and tensor have different dimensions or their dimension is not in [2,3]'

    # Calculate padding sizes for the last two dimensions
    pad_height = target_shape[-2] - tensor.shape[-2]
    pad_width = target_shape[-1] - tensor.shape[-1]

    if pad_height < 0 or pad_width < 0:
        raise ValueError("Target shape must be greater than or equal to the current tensor shape in padded dimensions.")
    padding = (0, pad_width, 0, pad_height)  # Pad width first, then height

    return F.pad(tensor, padding)

def zero_out_no_move(y):
    y[torch.where(y[...,0]!=2)][...,1:3] = 0
    return y

def compute_distance_weights(x,edge_index):

    src, tgt = edge_index  # (2, num_edges)
    edge_weight = 1/torch.norm(x[src] - x[tgt], p=2, dim=1)
    return edge_weight

def compute_pseudo_coords(x,edge_index, x_min,x_max, method = 'midpoint', clamp_outliers = True):
    x_norm = (x - x_min) / (x_max - x_min + 1e-6)  # Avoid division by zero
    if method.lower() == 'midpoint':
        pseudo_coordinates = (x_norm[edge_index[0]] + x_norm[edge_index[1]]) / 2
    elif method.lower() == 'diff':
        pseudo_coordinates = torch.abs((x_norm[edge_index[0]] - x_norm[edge_index[1]]))
    else:
        raise NotImplementedError("Unknown method {}".format(method))
    pseudo_coordinates = torch.clamp(pseudo_coordinates, 0, 1) if clamp_outliers else pseudo_coordinates
    return pseudo_coordinates

def get_edge_coords(x,edge_index):

    prev_x = torch.roll(x, shifts=1, dims=-1)
    out = torch.cat([x,prev_x], dim=-2)
    return out

def get_edge_coords_self_loop(x,edge_index):
    prev_x = torch.roll(x, shifts=1, dims=-1)
    out = torch.cat([x, prev_x], dim=-2)
    self_loop = torch.cat([x,x], dim=-2)
    out = torch.cat([out,self_loop], dim=-1)
    return out

def get_edge_diffs(x,edge_index):
    prev_x = torch.roll(x, shifts=1, dims=-1)
    out = torch.cat([x-prev_x, x-x], dim=-1)
    return  out

def get_edge_coords_with_diffs(x,edge_index):
    prev_x = torch.roll(x, shifts=1, dims=-1)
    out = torch.cat([x,prev_x,x-prev_x], dim=-2)
    self_loop = torch.cat([x,x, x-x], dim=-2)
    out = torch.cat([out,self_loop], dim=-1)
    return out

def create_polygon_edge_index(num_vertices):
    source_nodes = []
    target_nodes = []

    for i in range(num_vertices):
        prev_node = (i - 1) % num_vertices  # Previous neighbor
        next_node = (i + 1) % num_vertices  # Next neighbor

        source_nodes.append(i)
        target_nodes.append(prev_node)

        source_nodes.append(i)
        target_nodes.append(next_node)

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    return edge_index

def polygon_min_max_std(polygons):
    min_max_values = []

    for polygon in polygons:
        if torch.is_tensor(polygon):
            polygon = polygon.numpy()
        if polygon.shape[-1] != 2:
            raise ValueError("Each polygon must have shape (n,2)")
        if polygon.ndim >2:
            polygon.squeeze()

        centroid = np.mean(polygon, axis=0)

        # Center the polygon
        centered_polygon = polygon - centroid

        # Get min and max across both dimensions
        min_val = np.min(centered_polygon)
        max_val = np.max(centered_polygon)

        min_max_values.extend([min_val, max_val])

    # Compute standard deviation
    std_dev = np.std(min_max_values)

    return std_dev

def normalize(tensor, std = 17.551151):
    assert tensor.shape[-1] == 2, 'Assumes xy coords are in last dimension'
    assert tensor.ndim == 2, f'Assumes no batch dimension. Tensor has {tensor.ndim} dimensions.'
    return (tensor - tensor.mean(dim=0))/std

def polygons_to_arrays(geo_series):
    array_list = []
    for geom in geo_series:
        # Extract x and y coordinates
        x_coords = list(geom.exterior.coords.xy[0])
        y_coords = list(geom.exterior.coords.xy[1])

        # Combine x and y into a tensor with shape (n_vertices, 2)
        coords_array = np.array(list(zip(x_coords, y_coords)), dtype=np.float32)

        # Append to the list
        array_list.append(coords_array)

    return array_list

def delete_last_vertices(polygons):
    polygons = [np.delete(polygon, -1, axis=0) for polygon in polygons]
    return polygons
