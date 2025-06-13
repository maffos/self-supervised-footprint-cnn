from .utils import get_activation,get_graph_conv_layer,get_graph_norm_layer,get_norm_layer
from torch import nn
import torch
from torch_geometric.nn import MessagePassing

class TriangleConv(nn.Module):
    def __init__(self):
        super(TriangleConv, self).__init__()

    def forward(self, x):

        # Circular shift for previous (-1) and next (+1) neighbors
        prev_x = torch.roll(x, shifts=1, dims=-1)
        next_x = torch.roll(x, shifts=-1, dims=-1)

        # Compute differences
        diff_prev = x - prev_x
        diff_next = x - next_x
        diff_triangle = prev_x-next_x

        # Concatenate along feature dimension
        output = torch.cat([x, diff_prev, diff_next,diff_triangle], dim=1)

        return output

class TriangleFeatureExtractor(nn.Module):

    def __init__(self, in_channels,out_channels,
                 norm='batch',
                 stride = 1,
                 hidden_dims=[64, 256, 256, 256],
                 activation='relu',
                 num_groups=None,
                 dropout=None,
                 conv_type = 'standard',
                 mlp_norm = None):

        super(TriangleFeatureExtractor, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_type = norm
        hidden_dims[0] = self.in_channels*4 if conv_type == 'diffencoder' else hidden_dims[0]
        self.hidden_dims = hidden_dims
        self.num_layers = len(self.hidden_dims)
        self.stride = stride
        self.activation = get_activation(activation)
        self.bias = norm is None
        self.num_groups = num_groups
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

        if conv_type == 'standard':
            self.conv_layer = nn.Sequential(nn.Conv1d(self.in_channels, self.hidden_dims[0], kernel_size=3, padding=1, stride = self.stride, padding_mode='circular', bias=self.bias),
                                get_norm_layer(norm, self.hidden_dims[0],num_groups),
                                self.activation)
        elif conv_type == 'triangleconv':
            self.conv_layer = TriangleConv()
        else:
            raise ValueError("conv_type must be 'standard' or 'triangleconv'")

        self.mlp = self._make_mlp(hidden_dims, mlp_norm)
        self.out_layer = nn.Conv1d(self.hidden_dims[-1], self.out_channels, kernel_size=1, bias=True)

    def _make_mlp(self,hidden_dims, mlp_norm):

        layers = []
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Conv1d(hidden_dims[i], hidden_dims[i+1], kernel_size=1, bias=self.bias))
            layers.append(get_norm_layer(mlp_norm, hidden_dims[i+1], self.num_groups))
            layers.append(self.activation)
            layers.append(self.dropout)

        return nn.Sequential(*layers)

    def forward(self, x, *inputs):

        x = self.conv_layer(x,*inputs)
        x = self.mlp(x)
        x = self.out_layer(x)
        return x

class TriangleFeatureBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 norm='batch',
                 mlp_norm = None,
                 activation='relu',
                 upproject=None,
                 residual_connection=True,
                 num_groups = None,
                 dropout = None,
                 bottleneck = True,
                 stride= 1,
                 conv_type = 'standard',
                 append_coords = False,
                 d_pos = 0,
                 project_coords = True):
        super(TriangleFeatureBlock, self).__init__()

        has_bias = norm is None
        self.append_coords = append_coords
        self.residual_connection = residual_connection
        d_pos = d_pos if d_pos > 0 else in_channels
        self.coords_input_dim = d_pos if self.append_coords else 0
        self.conv_type = conv_type
        self.stride = stride
        self.d_pos = 2*self.coords_input_dim if project_coords else self.coords_input_dim
        self.coord_proj = nn.Sequential(nn.Conv1d(self.coords_input_dim, self.d_pos, kernel_size=3, padding=1,
                                    padding_mode='circular'),
                                   # get_norm_layer(norm,self.d_pos, self.d_pos//2),
                                    nn.Conv1d(self.d_pos,self.d_pos,kernel_size = 1)) if project_coords else nn.Identity()
        if bottleneck:
            bottleneck_size = in_channels // 4 if in_channels > 64 else 64
        else:
            bottleneck_size = in_channels

        self.conv1 = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, bias=has_bias)
        assert (not self.append_coords or d_pos != 0), 'append coords => d_pos !=0'
        if conv_type == 'standard':
            self.conv2 = nn.Conv1d(bottleneck_size+self.d_pos, bottleneck_size, kernel_size=3, padding=1, stride = self.stride, padding_mode='circular', bias=has_bias)
        elif conv_type == 'triangleconv':
            self.conv2 = TriangleConv()
        else:
            raise NotImplementedError("Unknown conv type '{}'".format(conv_type))
        self.conv3 = nn.Conv1d((bottleneck_size+self.d_pos)*4, out_channels, kernel_size=1, bias=has_bias) if self.conv_type == 'diffencoder' else nn.Conv1d(bottleneck_size, out_channels, kernel_size=1, bias=has_bias)
        self.conv4 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=has_bias)
        self.activation = get_activation(activation)

        self.nl1 = get_norm_layer(norm, bottleneck_size,num_groups)
        self.nl2 = nn.Identity if conv_type == 'diffencoder' else get_norm_layer(norm, bottleneck_size,num_groups)
        self.nl3 = get_norm_layer(mlp_norm, out_channels,num_groups)
        self.nl4 = get_norm_layer(norm, out_channels,num_groups)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

        self.upproject = upproject
        self.residual_connection = residual_connection

    def forward(self, x, coords = None, *args):

        coords_proj = self.coord_proj(coords) if self.append_coords else None
        out = self.activation(self.nl1(self.conv1(x)))
        assert x.shape[-1] == coords_proj.shape[-1], f'wrong number of vertices, {x.shape}, {coords_proj.shape}, {self.coord_proj}'
        if self.append_coords:
            out = torch.cat([out,coords_proj], dim=1)
        out = self.conv2(out,*args) if self.conv_type == 'diffencoder' else self.activation(self.nl2(self.conv2(out, *args)))
        out = self.dropout(self.activation(self.nl3(self.conv3(out))))
        out = self.nl4(self.conv4(out))
        if self.residual_connection:
            identity = self.upproject(x) if self.upproject else x
            out += identity
            out = self.activation(out)
        else:
            out = self.dropout(self.activation(out))
        return out

class GraphTriangleEncoder(MessagePassing):
    def __init__(self, in_channels,
                 out_channels,
                 hidden_dims,
                 activation,
                 norm = 'graph',
                 mlp_norm = 'graph',
                 graph_operator = 'gcn',
                 projection = 'mlp',
                 conv_params = {},
                 dropout = 0,
                 num_groups = None):
        super(GraphTriangleEncoder, self).__init__()

        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.graph_operator = graph_operator
        conv_in = in_channels if projection is None else hidden_dims[0]
        self.graph_conv = get_graph_conv_layer(graph_operator, conv_in,hidden_dims[0], **conv_params)
        self.norm = get_norm_layer(norm, hidden_dims[0], num_groups = num_groups) if graph_operator == 'edge_conv' else get_graph_norm_layer(norm,hidden_dims[0])
        node_layers = []
        self.projection = nn.Sequential(nn.Linear(in_channels, hidden_dims[0]),
                                        self.activation,
                                        self.dropout,
                                        nn.Linear(hidden_dims[0], hidden_dims[0]),
                                        get_norm_layer(norm, hidden_dims[0], num_groups = num_groups),
                                        self.activation) if projection == 'mlp' else nn.Identity()

        for i in range(len(hidden_dims)-1):
            node_layers.append(nn.Conv1d(hidden_dims[i], hidden_dims[i+1], kernel_size = 1) if graph_operator == 'edge_conv' else nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            node_layers.append(get_norm_layer(mlp_norm, hidden_dims[i+1], num_groups = num_groups) if graph_operator == 'edge_conv' else get_graph_norm_layer(norm, hidden_dims[i+1]))
            node_layers.append(self.activation)
            node_layers.append(self.dropout)

        self.node_layers = nn.Sequential(*node_layers)
        self.output_layer = nn.Conv1d(hidden_dims[-1], out_channels, kernel_size = 1) if graph_operator == 'edge_conv' else nn.Linear(hidden_dims[-1], out_channels)

    def forward(self, x, edge_index, *edge_attr):
        if self.graph_operator == 'spline_conv':
            x = x.squeeze()
        x = self.projection(x)
        x = self.graph_conv(x, edge_index, *edge_attr)
        x = self.norm(x)
        x = self.activation(x)
        x = self.node_layers(x)
        x = self.output_layer(x)
        return x

class GraphTriangleBlock(MessagePassing):
    def __init__(self, in_channels,
                 out_channels,
                 activation,
                 norm='graph',
                 mlp_norm = 'graph',
                 graph_operator='gcn',
                 bottleneck = True,
                 conv_params = {},
                 num_groups = 0,
                 dropout = 0):
        super(GraphTriangleBlock, self).__init__()

        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        bottleneck_size = in_channels//4 if bottleneck and in_channels > 64 else in_channels
        self.downproject = nn.Conv1d(in_channels,bottleneck_size, kernel_size = 1) if graph_operator == 'edge_conv' else nn.Linear(in_channels, bottleneck_size)
        self.graph_conv = get_graph_conv_layer(graph_operator, bottleneck_size, bottleneck_size, **conv_params)
        self.uproject = nn.Conv1d(bottleneck_size, out_channels, kernel_size=1) if graph_operator == 'edge_conv' else nn.Linear(bottleneck_size, out_channels)
        self.out_layer = nn.Conv1d(out_channels,out_channels,kernel_size=1) if graph_operator == 'edge_conv' else nn.Linear(out_channels, out_channels)
        self.norm_layer1 = get_norm_layer(norm, bottleneck_size, num_groups) if graph_operator == 'edge_conv' else get_graph_norm_layer(norm, bottleneck_size)
        self.norm_layer2 = get_norm_layer(norm, bottleneck_size, num_groups) if graph_operator == 'edge_conv' else get_graph_norm_layer(norm, bottleneck_size)
        self.norm_layer3 = get_norm_layer(mlp_norm, out_channels, num_groups) if graph_operator == 'edge_conv' else get_graph_norm_layer(norm, out_channels)
        self.norm_layer4 = get_norm_layer(norm, out_channels, num_groups) if graph_operator == 'edge_conv' else get_graph_norm_layer(norm, out_channels)

    def forward(self, x, edge_index, *edge_attr):
        x = self.activation(self.norm_layer1(self.downproject(x)))
        x = self.activation(self.norm_layer2(self.graph_conv(x, edge_index, *edge_attr)))
        x = self.dropout(self.activation(self.norm_layer3(self.uproject(x))))
        x = self.activation(self.norm_layer4(self.out_layer(x)))
        return x


