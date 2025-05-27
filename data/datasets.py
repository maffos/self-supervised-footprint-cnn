import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import DataLoader as PyGDataLoader
from torch_geometric.data import Data as GraphData
import os
import geopandas as gpd
import pandas as pd
import copy
from collections import defaultdict
from data.utils import *
from classification.DPCN.new_pre import get_shape_mbr, get_shape_normalize_final, get_shape_simplify, get_node_features, reset_start_point, get_inter_features, get_neat_features, get_line_features_final

class TriangleFeatureDataset(Dataset):

    def __init__(self, data_dir: str,
                 split: str,
                 device: torch.device,
                 load_raw_coords = False,
                 transform: list = None,
                 target_transform: list = None,
                 bucketize: bool = False,
                 batch_size: int = 64):
        """
        Initializes the dataset with the given parameters.
        :param data_dir: Directory where data files are located.
        :param split: The dataset split to load ('train', 'valid', 'test').
        :param load_raw_coords: Whether to load raw features directly (True) or processed features (False).
        :param transform: A list of transformations to apply to the input data. Default is None.
        :param target_transform: A list of transformations to apply to the target labels. Default is None.
        :param bucketize: Whether to bucketize the data by size for more efficient batching. Default is False.
        :param batch_size: The batch size for training or inference. Only required when bucketize is set to True. Default is 64.
        """

        super(TriangleFeatureDataset, self).__init__()
        # Determine the root directory (one level above src/)
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Set default data_dir to the 'data/' folder in the root directory
        self.data_dir = os.path.join(root_dir, data_dir) if data_dir else os.path.join(root_dir, "data")
        self.split = split
        self.device = device
        self.transform = self._initialize_transforms(transform)
        self.target_transform = self._initialize_transforms(target_transform)
        self.spatial_dim = -1 if target_transform is not None and 'swap_channels' in target_transform else -2

        # Data files corresponding to each split
        feature_fname = f'x_{split}.bin' if load_raw_coords else f'xFully_{split}.bin'

        self.data_files = {
            'x': feature_fname,
            'y': f'y_{split}.bin'
        }

        # Load data from files
        self.data = {key: self.from_binary(os.path.join(self.data_dir, filename))
                     for key, filename in self.data_files.items()}

        if split != 'train':
            bucketize = False #only bucket the training set

        if bucketize and not batch_size:
            raise ValueError('You must specify a batch size when bucketize is set to True.')
        elif not bucketize and batch_size:
            batch_size = None

        self.bucketize = bucketize
        self.data = self.bucketize_and_batch(batch_size) if bucketize else self.data

        self.max_extent = self._get_max_extent('x')

    def __len__(self):
        return len(self.data['x'])

    def __getitem__(self, idx):
        # Retrieve the x, y, and adj_matrix for the specified index
        x = self.data['x'][idx]
        y = self.data['y'][idx]

        # Convert to torch tensors if needed
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        x_tensor_transformed = copy.deepcopy(x_tensor)
        y_tensor_transformed = copy.deepcopy(y_tensor)
        if self.transform:
            x_tensor_transformed = self.apply_transforms(x_tensor_transformed,self.transform)
        if self.target_transform:
            y_tensor_transformed = self.apply_transforms(y_tensor_transformed,self.target_transform, x_tensor)

        return {'x':x_tensor_transformed.to(self.device), 'y':y_tensor_transformed.to(self.device)}

    def _get_max_extent(self, key):
        max_extent = 0
        for i in range(len(self.data[key])):
            if self.data[key][i].shape[self.spatial_dim] > max_extent:
                max_extent = self.data[key][i].shape[self.spatial_dim]

        return max_extent

    def _get_matrix_column_size(self, filename):

        with open(filename, "rb") as file:
            file.seek(0, os.SEEK_SET)
            num_fields = np.fromfile(file, dtype=np.intc, count=1)
            _ = np.fromfile(file, dtype=np.intc, count=1)  # we only need to read this but don't use the value
            memory_count = np.fromfile(file, dtype=np.intc, count=num_fields[0] + 1)
            file.seek(memory_count[0], os.SEEK_SET)
            matrix_shape = np.fromfile(file, dtype=np.intc, count=2)
        return matrix_shape[1]

    def _initialize_transforms(self, transforms):
        if transforms is None:
            return None
        transform_map = {
            'swap_channels': swap_channels,
            'zero_padding': zero_padding,
            'normalize': normalize,
        }
        transform_list = []
        for trans in transforms:
            if trans in transform_map:
                transform_list.append(transform_map[trans])
        return transform_list


    def get_feature_names(self, filename="data_features.txt"):
        filein = os.path.join(self.data_dir, filename)
        data = [line.strip() for line in open(filein, 'r')]
        columns = data[0].split()

        return columns

    def from_binary(self, filename: str, index_init: int = -1, index_end: int = -1) -> list:

        with (open(filename, "rb") as file):
            file.seek(0, os.SEEK_SET)
            num_fields = np.fromfile(file, dtype=np.intc, count=1)
            value_type = np.fromfile(file, dtype=np.intc, count=1)

            if index_init < 0:
                n_fields = num_fields[0]
                index_init = 0
            elif index_end < 0:
                n_fields = 1
            else:
                if index_init > index_end:
                    index_init, index_end = index_end, index_init
                n_fields = min(num_fields[0] - 1, index_end) - max(0, index_init) + 1

            memory_count = np.fromfile(file, dtype=np.intc, count=num_fields[0] + 1)
            file.seek(memory_count[index_init], os.SEEK_SET)

            matrices = []

            def _dtype():

                if value_type[0] == 0:
                    return np.intc
                elif value_type[0] == 1:
                    return np.longlong
                elif value_type[0] == 2:
                    return np.single
                else:
                    return np.double

            for _ in range(n_fields):
                matrix_shape = np.fromfile(file, dtype=np.intc, count=2)
                matrix = np.fromfile(file, dtype=_dtype(), count=matrix_shape[0] * matrix_shape[1]).reshape(
                    matrix_shape[0],
                    matrix_shape[1])
                matrices.append(matrix)
        return matrices

    def get_data_feature_dim(self):
        """
        Get the feature dimension of the x data.
        """
        return self._get_matrix_column_size(os.path.join(self.data_dir, self.data_files['x']))

    def get_data_output_dim(self):
        """
        Get the output dimension of the y data.
        """
        return self._get_matrix_column_size(os.path.join(self.data_dir, self.data_files['y']))

    def bucketize_and_batch(self, batch_size):
        """
        Bucketize a dataset based on the first dimension of the xs and create batches.

        Parameters:
            batch_size (int): The maximum number of items in a batch.

        Returns:
            list of tuples: A list of batches, where each batch is a tuple (batch_xs, batch_ys).
                            batch_xs and batch_ys are numpy arrays of shapes [batch_size, ?, 2] and [batch_size, ?, 7].
        """
        # Create buckets based on the first dimension of x
        buckets = defaultdict(list)
        for x, y in zip(self.data['x'], self.data['y']):
            bucket_key = x.shape[0]  # Group by the length of the first dimension of x
            buckets[bucket_key].append((x, y))

        # Create batches from each bucket
        batches = {'x': [], 'y': []}
        for bucket_key, items in buckets.items():
            # Split the bucket into batches
            for i in range(0, len(items), batch_size):
                batch_items = items[i:i + batch_size]
                batch_xs = [x for x, y in batch_items]
                batch_ys = [y for x, y in batch_items]

                batches['x'].append(np.array(batch_xs))
                batches['y'].append(np.array(batch_ys))

        return batches

    def apply_transforms(self, sample, transforms, *args):
        for transform in transforms:
            if transform == zero_padding:
                target_shape = []
                for i in range(sample.ndim):
                    target_shape.append(sample.shape[i])
                target_shape[-2] = self.max_extent
                sample = transform(sample, target_shape)
            elif transform == normalize:
                std = self.std()
                sample = transform(sample,std)
            else:
                sample = transform(sample, *args)

        return sample

    def get_data_loader(self, batch_size):

        if self._is_valid_dataloader(batch_size):
            return DataLoader(self, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            raise ValueError('DataLoader is not valid.')

    def _is_valid_dataloader(self, batch_size):
        return zero_padding in self.transform or self.bucketize or batch_size == 1

    def target_statistics(self):
        y = self.data['y']
        mean = np.zeros(7)
        std = np.zeros(7)
        max_vals = np.zeros(7) - np.inf
        min_vals = np.ones(7) * np.inf

        for idx in range(len(y)):
            footprint = np.array(y[idx])
            mean += footprint.mean(axis=0)

            footprint_max = footprint.max(axis=0)
            footprint_min = footprint.min(axis=0)

            max_vals = np.maximum(max_vals, footprint_max)
            min_vals = np.minimum(min_vals, footprint_min)

        mean /= len(y)

        for idx in range(len(y)):
            footprint = np.array(y[idx])
            std += np.mean((footprint - mean) ** 2, axis=0)
        std /= len(y)
        std = np.sqrt(std)
        statistics = pd.DataFrame(np.stack([mean,min_vals,max_vals,std], axis=0),index=['mean','min','max','std'],
                                  columns=self.get_feature_names())


        return statistics

    def footprint_std(self):
        return polygon_min_max_std(self.data['x'])

class TriangleFeatureGraphDataset(TriangleFeatureDataset):

    def __init__(self, *args,  compute_edge_weights = False, **kwargs):

        super(TriangleFeatureGraphDataset, self).__init__(*args, **kwargs)
        self.compute_edge_weights = compute_edge_weights

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        edge_index = create_polygon_edge_index(batch['x'].shape[0]).to(batch['x'].device)
        edge_weights = compute_distance_weights(batch['x'],edge_index) if self.compute_edge_weights else torch.ones(edge_index.shape[1])
        data = GraphData(x=batch['x'], edge_index=edge_index, y=batch['y'], edge_attr=edge_weights)

        return data

class ClassificationDataset(Dataset):
    """
    This code is taken and modified from https://github.com/Huyaohui122/DPCN
    """
    def __init__(self, split='train', data_dir = './data/', transforms = None, raw_coordinates = False, no_val_set = False):

        super(ClassificationDataset, self).__init__()
        self.bu_filename = 'test_5010'
        self.hparams = {
            'if_simplify': True
            , 'tor_dist': 0.1
            , 'tor_cos': 0.99
            , 'if_scale_y': False

            , 'scale_type': 'final'
            , 'seq_length': 16
            , 'rotate_type': 'equal'
            , 'rotate_length': 1
            , 'norm_type': 'minmax'
        }
        train_size = 4000
        self.raw_coordinates = raw_coordinates
        if no_val_set:
            label = np.genfromtxt(os.path.join(data_dir, "label.content"), dtype=np.int64)
            label_train = label[:train_size]
            label_train = label_train.reshape(-1, 1)
            label_test = label[train_size:]
            label_test = label_test.reshape(-1, 1)

            bu_shape = gpd.read_file(data_dir + self.bu_filename + '.shp', encode='utf-8')
            bu_use = copy.deepcopy(bu_shape)
            bu_mbr, bu_use = get_shape_mbr(bu_use)
            bu_use = get_shape_normalize_final(bu_use, self.hparams['if_scale_y'])

            if self.hparams['if_simplify']:
                bu_use = get_shape_simplify(bu_use, self.hparams['tor_dist'], self.hparams['tor_cos'], simplify_type=0)
            if raw_coordinates:
                all_polygons = polygons_to_arrays(bu_use.geometry)
                all_polygons = delete_last_vertices(all_polygons)
                if split == 'train':
                    modelnet_data = all_polygons[:train_size]
                    modelnet_label = label_train
                elif split == 'test':
                    modelnet_data = all_polygons[train_size:]
                    modelnet_label = label_test
                else:
                    raise ValueError('split must be train or test')
            else:
                bu_use = reset_start_point(bu_use)
                bu_node = get_node_features(bu_use)
                bu_line = get_line_features_final(bu_node, self.hparams['seq_length'])
                bu_detail = get_inter_features(bu_line)
                bu_detail = get_neat_features(bu_detail, self.hparams['seq_length'], self.hparams['rotate_length'])

                all_data_x = np.array(bu_detail['xs'])
                train_x = all_data_x[:train_size * self.hparams['seq_length']]
                train_x = train_x.reshape(train_size * self.hparams['seq_length'], 1)
                test_x = all_data_x[train_size * self.hparams['seq_length']:]
                test_x = test_x.reshape(-1, 1)

                all_data_y = np.array(bu_detail['ys'])
                train_y = all_data_y[:train_size * self.hparams['seq_length']]
                train_y = train_y.reshape(train_size * self.hparams['seq_length'], 1)
                test_y = all_data_y[train_size * self.hparams['seq_length']:]
                test_y = test_y.reshape(-1, 1)

                train_x_y = np.concatenate((train_x, train_y), axis=1)
                test_x_y = np.concatenate((test_x, test_y), axis=1)
                train_xy_reshape = train_x_y.reshape(train_size, self.hparams['seq_length'], 2)
                test_xy_reshape = test_x_y.reshape(-1, self.hparams['seq_length'], 2)

                if split == 'train':
                    modelnet_data = np.zeros([0, self.hparams['seq_length'], 2], dtype=np.float64)
                    modelnet_label = np.zeros([0, 1], np.float64)
                    modelnet_data = np.concatenate([modelnet_data, train_xy_reshape], axis=0)
                    modelnet_label = np.concatenate([modelnet_label, label_train], axis=0)
                elif split == 'test':
                    modelnet_data = np.zeros([0, self.hparams['seq_length'], 2], dtype=np.float64)
                    modelnet_label = np.zeros([0, 1], np.float64)
                    modelnet_data = np.concatenate([modelnet_data, test_xy_reshape], axis=0)
                    modelnet_label = np.concatenate([modelnet_label, label_test], axis=0)
                else:
                    raise ValueError('split must be train or test')
        else:
            val_size = 505

            label = np.genfromtxt(os.path.join(data_dir, "label.content"), dtype=np.int64)
            label_train = label[:train_size]
            label_train = label_train.reshape(-1, 1)
            label_val = label[train_size:train_size + val_size]
            label_val = label_val.reshape(-1, 1)
            label_test = label[train_size + val_size:]
            label_test = label_test.reshape(-1, 1)

            bu_shape = gpd.read_file(data_dir + self.bu_filename + '.shp', encode='utf-8')
            bu_use = copy.deepcopy(bu_shape)
            bu_mbr, bu_use = get_shape_mbr(bu_use)
            bu_use = get_shape_normalize_final(bu_use, self.hparams['if_scale_y'])
            if self.hparams['if_simplify']:
                bu_use = get_shape_simplify(bu_use, self.hparams['tor_dist'], self.hparams['tor_cos'], simplify_type=0)
            if raw_coordinates:
                all_polygons = polygons_to_arrays(bu_use.geometry)
                all_polygons = delete_last_vertices(all_polygons)
                if split == 'train':
                    modelnet_data = all_polygons[:train_size]
                    modelnet_label = label_train
                elif split == 'test':
                    modelnet_data = all_polygons[train_size:]
                    modelnet_label = label_test
                else:
                    raise ValueError('split must be train or test')
            else:
                bu_use = reset_start_point(bu_use)
                bu_node = get_node_features(bu_use)
                bu_line = get_line_features_final(bu_node, self.hparams['seq_length'])
                bu_detail = get_inter_features(bu_line)
                bu_detail = get_neat_features(bu_detail, self.hparams['seq_length'], self.hparams['rotate_length'])

                all_data_x = np.array(bu_detail['xs'])
                train_x = all_data_x[:train_size * self.hparams['seq_length']]
                train_x = train_x.reshape(train_size * self.hparams['seq_length'], 1)
                val_x = all_data_x[train_size * self.hparams['seq_length']:(train_size + val_size) * self.hparams['seq_length']]
                val_x = val_x.reshape(-1, 1)
                test_x = all_data_x[(train_size + val_size) * self.hparams['seq_length']:]
                test_x = test_x.reshape(-1, 1)

                all_data_y = np.array(bu_detail['ys'])
                train_y = all_data_y[:train_size * self.hparams['seq_length']]
                train_y = train_y.reshape(train_size * self.hparams['seq_length'], 1)
                val_y = all_data_y[train_size * self.hparams['seq_length']:(train_size + val_size) * self.hparams['seq_length']]
                val_y = val_y.reshape(-1, 1)
                test_y = all_data_y[(train_size + val_size) * self.hparams['seq_length']:]
                test_y = test_y.reshape(-1, 1)

                train_x_y = np.concatenate((train_x, train_y), axis=1)
                val_x_y = np.concatenate((val_x, val_y), axis=1)
                test_x_y = np.concatenate((test_x, test_y), axis=1)
                train_xy_reshape = train_x_y.reshape(train_size, self.hparams['seq_length'], 2)
                test_xy_reshape = test_x_y.reshape(val_size, self.hparams['seq_length'], 2)
                val_xy_reshape = val_x_y.reshape(val_size, self.hparams['seq_length'], 2)

                if split == 'train':
                    modelnet_data = np.zeros([0, self.hparams['seq_length'], 2], dtype=np.float64)
                    modelnet_label = np.zeros([0, 1], np.float64)
                    modelnet_data = np.concatenate([modelnet_data, train_xy_reshape], axis=0)
                    modelnet_label = np.concatenate([modelnet_label, label_train], axis=0)
                elif split == 'test':
                    modelnet_data = np.zeros([0, self.hparams['seq_length'], 2], dtype=np.float64)
                    modelnet_label = np.zeros([0, 1], np.float64)
                    modelnet_data = np.concatenate([modelnet_data, test_xy_reshape], axis=0)
                    modelnet_label = np.concatenate([modelnet_label, label_test], axis=0)
                elif split == 'val':
                    modelnet_data = np.zeros([0, self.hparams['seq_length'], 2], dtype=np.float64)
                    modelnet_label = np.zeros([0, 1], np.float64)
                    modelnet_data = np.concatenate([modelnet_data, val_xy_reshape], axis=0)
                    modelnet_label = np.concatenate([modelnet_label, label_val], axis=0)
                else:
                    raise ValueError('split must be train, val or test')

        self.point_cloud = modelnet_data
        self.label = modelnet_label
        self.transforms = self._initialize_transforms(transforms)

    def _initialize_transforms(self, transforms):
        """Initialize and validate transformation list."""
        if transforms is None:
            return None
        transform_map = {
            'swap_channels': swap_channels,
            'zero_padding': zero_padding,
            'normalize': normalize
        }
        transform_list = []
        for trans in transforms:
            if trans in transform_map:
                transform_list.append(transform_map[trans])
        return transform_list

    def __getitem__(self, item):
        x = torch.tensor(self.point_cloud[item], dtype = torch.float32)
        for transform in self.transforms or []:
            if transform == normalize:
                std = self.std()
                x = transform(x,std)
            else:
                x = transform(x)
        y = torch.tensor(self.label[item], dtype = torch.int64)
        return x,y

    def __len__(self):
        return self.label.shape[0]

    def std(self):
        polygon_min_max_std(self.point_cloud)

class BuildingSimplificationDataset(Dataset):
    def __init__(self, data_dir, split, transforms=None, bucketize = False, target_transform = None, batch_size = 1):
        super(BuildingSimplificationDataset, self).__init__()
        data = np.load('{}/vertex_{}.npy'.format(data_dir, split), allow_pickle=True)
        self.coords = [footprint[:,4:6] for footprint in data]
        self.metadata = [footprint[:,:4] for footprint in data] if split == 'test' else None
        self.labels = [footprint[:,-3:] for footprint in data]
        self.data = self.bucketize_and_batch(self.coords,self.labels,batch_size) if bucketize else list(zip(self.coords,self.labels))
        self.transforms = self._initialize_transforms(transforms)
        self.target_transform = self._initialize_transforms(target_transform)

    def _initialize_transforms(self, transforms):
        if transforms is None:
            return None
        transform_map = {
            'swap_channels': swap_channels,
            'zero_padding': zero_padding,
            'normalize': normalize,
            'zero_out_no_move': zero_out_no_move
        }
        transform_list = []
        for trans in transforms:
            if trans in transform_map:
                transform_list.append(transform_map[trans])
        return transform_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        x,y = self.data[idx]
        meta_data = self.metadata[idx] if self.metadata is not None else None
        x = torch.from_numpy(x).to(torch.float32)
        y = torch.from_numpy(y).to(torch.float32)
        for transform in self.transforms or []:
            if transform == normalize:
                std = self.std()
                x = transform(x, std)
            else:
                x = transform(x)

        for transform in self.target_transform or []:
            y = transform(y)
        out_dict = {'x': x, 'y': y, 'meta_data': meta_data} if meta_data is not None else {'x': x, 'y': y}
        return out_dict

    def footprint_std(self):
        return polygon_min_max_std(self.coords)

    def get_min_max(self):
        min_value = np.inf
        max_value = -np.inf
        for coords in self.coords:
            min_value = min(min_value, np.min(coords))
            max_value = max(max_value, np.max(coords))
        return min_value, max_value

    def bucketize_and_batch(self, coords, labels, batch_size):
        """
        Bucketize a dataset based on the first dimension of the xs and create batches.

        Parameters:
            batch_size (int): The maximum number of items in a batch.

        Returns:
            list of tuples: A list of batches, where each batch is a tuple (batch_xs, batch_ys).
                            batch_xs and batch_ys are numpy arrays of shapes [batch_size, ?, 2] and [batch_size, ?, 7].
        """
        # Create buckets based on the first dimension of x
        buckets = defaultdict(list)
        for x, y in zip(coords, labels):
            bucket_key = x.shape[0]  # Group by the length of the first dimension of x
            buckets[bucket_key].append((x, y))

        # Create batches from each bucket
        xs,ys = [],[]
        for bucket_key, items in buckets.items():
            # Split the bucket into batches
            for i in range(0, len(items), batch_size):
                batch_items = items[i:i + batch_size]
                batch_xs = [x for x, y in batch_items]
                batch_ys = [y for x, y in batch_items]

                xs.append(np.array(batch_xs))
                ys.append(np.array(batch_ys))

        return list(zip(xs,ys))

class BuildingSimplificationGraphDataset(BuildingSimplificationDataset):

    def __init__(self, *args, edge_attr_type = 'distance_weights', edge_attr_params = {}, **kwargs):
        super(BuildingSimplificationGraphDataset, self).__init__(*args, **kwargs)
        self.edge_attr_fn = self._get_edge_attr_fn(edge_attr_type)
        self.edge_attr_params = self._init_edge_attr_params(edge_attr_params,edge_attr_type)

    def _init_edge_attr_params(self, edge_attr_params, edge_attr_type):
        if edge_attr_type in ['pseudo_coordinates', 'pseudo_coords']:
            x_min,x_max = self.get_min_max()
            edge_attr_params.update({'x_min': x_min, 'x_max': x_max})
        return edge_attr_params

    def _get_edge_attr_fn(self, edge_attr_type):

        edge_attr_types = {'distance_weights': compute_distance_weights,
                           'pseudo_coords': compute_pseudo_coords,
                           'ones': lambda x, y: torch.ones(y.shape[1]),
                           'coords': get_edge_coords,
                           'coords_self_loops': get_edge_coords_self_loop,
                           'diffs': get_edge_diffs,
                           'coords_diffs': get_edge_coords_with_diffs}
        return edge_attr_types.get(edge_attr_type, None)

    def __getitem__(self, idx):
        item_dict = super().__getitem__(idx)
        edge_index = create_polygon_edge_index(item_dict['x'].shape[0]).to(item_dict['x'].device)
        edge_attr = self.edge_attr_fn(item_dict['x'], edge_index, **self.edge_attr_params) if self.edge_attr_fn is not None else None
        data = GraphData(x=item_dict['x'], edge_index=edge_index, y=item_dict['y'], edge_attr=edge_attr)
        return data


def get_dataloaders(model_type, data_dir, batch_size, bucketize, **kwargs):
    if model_type == 'gnn':
        train_set = BuildingSimplificationGraphDataset(data_dir,
                                                       split='train',
                                                       batch_size=batch_size,
                                                       bucketize=bucketize,
                                                       **kwargs)
        val_set = BuildingSimplificationGraphDataset(data_dir,
                                                     split='valid',
                                                     batch_size=batch_size,
                                                     bucketize=bucketize,
                                                     **kwargs)
        test_set = BuildingSimplificationGraphDataset(data_dir,
                                                      split='test',
                                                      batch_size=batch_size,
                                                      bucketize=bucketize,
                                                      **kwargs)
        train_loader = PyGDataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = PyGDataLoader(val_set, batch_size=1, shuffle=False, drop_last=True)
        test_loader = PyGDataLoader(test_set, batch_size=1, shuffle=False, drop_last=True)
    else:
        train_set = BuildingSimplificationDataset(data_dir,
                                                  split='train',
                                                  batch_size=batch_size,
                                                  bucketize=bucketize,
                                                  **kwargs)
        val_set = BuildingSimplificationDataset(data_dir,
                                                split='valid',
                                                batch_size=batch_size,
                                                bucketize=bucketize,
                                                **kwargs)
        test_set = BuildingSimplificationDataset(data_dir,
                                                 split='test',
                                                 batch_size=batch_size,
                                                 bucketize=bucketize,
                                                 **kwargs)
        batch_size = 1 if bucketize else batch_size
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader