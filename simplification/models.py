from src.models import GraphTriangleEncoder,GraphTriangleBlock,TriangleFeatureBlock,TriangleFeatureExtractor,TriangleConv
from src.utils import get_norm_layer,get_activation,get_graph_norm_layer,get_graph_conv_layer
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import Sequential as PyGSequential

class BuildingSimplificationGraphModel(MessagePassing):
    def __init__(self, triangle_embed_params,
                 block_params,
                 num_classes = 3,
                 num_reg_targets = 2,
                 in_channels=2,
                 hidden_dims = None,
                 num_reg_layers= 2,
                 num_cls_layers= 2,
                 task_dropout = 0.2,
                 multiply_labels = False,
                 out_dim = None):
        super(BuildingSimplificationGraphModel, self).__init__()

        self.activation = get_activation(block_params['activation'])
        self.d_model_triangle = triangle_embed_params['out_channels']
        self.graph_operator = block_params['graph_operator']
        if block_params['graph_operator'] == 'sage_conv':
            self.triangle_embedding = PyGSequential('x, edge_index', [
                (GraphTriangleEncoder(in_channels, **triangle_embed_params), 'x, edge_index -> x'),
                get_graph_norm_layer(triangle_embed_params['norm'], self.d_model_triangle),
                self.activation])
            layers = [(GraphTriangleBlock(self.d_model_triangle, hidden_dims[0], **block_params),
                       'x, edge_index -> x')]
            for i in range(len(hidden_dims) - 1):
                layers.append((GraphTriangleBlock(hidden_dims[i], hidden_dims[i + 1], **block_params),
                               'x, edge_index -> x'))
            self.hidden_layers = PyGSequential('x, edge_index', layers)
        else:
            self.triangle_embedding = PyGSequential('x, edge_index, edge_attr', [
                (GraphTriangleEncoder(in_channels, **triangle_embed_params),'x, edge_index, edge_attr -> x'),
                get_graph_norm_layer(triangle_embed_params['norm'], self.d_model_triangle),
                self.activation])
            layers = [(GraphTriangleBlock(self.d_model_triangle, hidden_dims[0], **block_params),
                       'x, edge_index, edge_attr -> x')]
            for i in range(len(hidden_dims) - 1):
                layers.append((GraphTriangleBlock(hidden_dims[i], hidden_dims[i + 1], **block_params),
                               'x, edge_index, edge_attr -> x'))
            self.hidden_layers = PyGSequential('x, edge_index, edge_attr', layers)
        self.multiply_labels = multiply_labels
        self.cls_head = self._initialise_task_head(num_cls_layers, self.d_model_triangle+hidden_dims[-1], out_dim, num_classes, task_dropout)
        self.reg_head = self._initialise_task_head(num_reg_layers, self.d_model_triangle+hidden_dims[-1], out_dim, num_reg_targets, task_dropout)
        self.loss_weights = torch.nn.Parameter(torch.ones(3))

    def _initialise_task_head(self, num_layers, in_channels, out_channels, num_targets, dropout):
        layers = [nn.Linear(in_channels, out_channels), self.activation, nn.Dropout(dropout)]
        for i in range(num_layers - 1):
            layers.append(nn.Linear(out_channels, out_channels))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(out_channels, num_targets))
        return nn.Sequential(*layers)

    def forward(self, x, edge_index, *edge_attr):

        if x.ndim > 2:
            x = x.squeeze()
        local_embeddings = self.triangle_embedding(x, edge_index, *edge_attr)
        global_embeddings = self.hidden_layers(local_embeddings, edge_index, *edge_attr)
        task_input = torch.cat([local_embeddings, global_embeddings], dim=-1)
        cls_out = self.cls_head(task_input)
        reg_out = self.reg_head(task_input)
        if self.multiply_labels:
            cls_labels = cls_out.argmax(-1)
            move_labels = torch.where(cls_labels==2,1,0)
            if x.ndim == 3:
                reg_out = reg_out*move_labels[:,:,None]
            else:
                reg_out = reg_out*move_labels[:,None]

        if x.ndim == 3:
            cls_out = cls_out.permute(0, 2, 1)

        return cls_out, reg_out[...,0], reg_out[...,1]

class BuildingSimplificationModel(nn.Module):
    def __init__(self, triangle_embed_params,
                 block_params,
                 num_classes=3,
                 norm='batch',
                 initial_num_groups=None,
                 hidden_dims=None,
                 activation='relu',
                 num_reg_layers=2,
                 num_cls_layers=2,
                 out_dim=None,
                 use_diff_encoder=False,
                 head_type='mlp',
                 num_layers = None,
                 task_dropout = 0.2,
                 blocks_per_layer=[1, 1, 1],
                 downsample = 'conv',
                 multiply_labels = False,
                 concat_local_features = False,
                 concat_class_probs = False,
                 ):
        super(BuildingSimplificationModel, self).__init__()

        self.activation = get_activation(activation)
        self.d_model_triangle = triangle_embed_params['out_channels']
        self.hidden_dims = hidden_dims if hidden_dims is not None else [self.d_model_triangle*2**i for i in range(num_layers)]
        self.num_layers = len(self.hidden_dims)
        self.diff_encoder = TriangleConv()
        self.use_diff_encoder = use_diff_encoder
        self.num_groups = [initial_num_groups*2**i for i in range(self.num_layers)]
        self.downsample = downsample
        self.blocks_per_layer = blocks_per_layer if isinstance(blocks_per_layer, list) else [blocks_per_layer for _ in range(self.num_layers)]
        self.multiply_labels = multiply_labels
        self.concat_local_features = concat_local_features
        self.dropout = nn.Dropout(task_dropout)
        self.concat_class_probs = concat_class_probs
        self.softmax = nn.Softmax(dim=1)
        input_channels = 8 if self.use_diff_encoder else 2
        self.d_pos = input_channels
        has_bias = (norm == None)
        self.block = TriangleFeatureBlock

        if block_params['append_coords']:
            self.coords_pathway = nn.ModuleList()
            self.coord_pooling_layers = nn.ModuleList()

            for i in range(self.num_layers-1):
                if self.downsample == 'maxpool':
                    self.coords_pathway.append(nn.Sequential(nn.Conv1d(self.d_pos*2**i, self.d_pos*2**(i+2), kernel_size=3,padding=1,padding_mode='circular'),
                                                             self.activation,
                                                             nn.Conv1d(self.d_pos*2**(i+2),self.d_pos*2**(i+1), kernel_size=1)))
                    self.coord_pooling_layers.append(nn.AdaptiveMaxPool1d(output_size = None))
                else:
                    self.coords_pathway.append(nn.Sequential(nn.Conv1d(self.d_pos*2**i, self.d_pos*2**(i+2), kernel_size=3,padding=1,stride=2,padding_mode='circular'),
                                                             self.activation,
                                                             nn.Conv1d(self.d_pos*2**(i+2), self.d_pos*2**(i+1), kernel_size=1)))
                    self.coord_pooling_layers.append(nn.Identity())
        else:
            self.coords_pathway = None

        #Encoder
        self.triangle_embedding = TriangleFeatureExtractor(in_channels=self.d_pos, **triangle_embed_params)
        self.norm_embed = get_norm_layer(triangle_embed_params['norm'], self.d_model_triangle)
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.encoder_blocks_layer1 = nn.ModuleList()
        assert self.d_model_triangle == self.hidden_dims[0], 'd_model_triangle and hidden_dims[0] are not the same'
        for i in range(self.blocks_per_layer[0]-1):
            self.encoder_blocks_layer1.append(
                self.block(
                   self.hidden_dims[0],
                    self.hidden_dims[0],
                    num_groups=self.num_groups[0],
                    d_pos = self.d_pos,
                    **block_params
                )
            )
        for level in range(1,self.num_layers):
            if downsample == 'conv':
                self.downsamples.append(nn.Conv1d(self.hidden_dims[level-1],
                                                self.hidden_dims[level],
                                                kernel_size=3,
                                                padding=1,
                                                padding_mode='circular',
                                                stride = 2)
                )
            elif downsample == 'maxpool':
                self.downsamples.append(nn.AdaptiveMaxPool1d(output_size=None))
            else:
                raise NotImplementedError('downsample needs to be either conv or maxpool')

            blocks_level = []
            if downsample == 'maxpool':
                blocks_level.append(self.block(
                        self.hidden_dims[level-1],
                        self.hidden_dims[level],
                        num_groups=self.num_groups[level],
                        d_pos = self.d_pos*2**level,
                        upproject = nn.Sequential(nn.Conv1d(self.hidden_dims[level-1], self.hidden_dims[level], kernel_size=1, bias=has_bias),
                                                  get_norm_layer(norm,self.hidden_dims[level], self.num_groups[level])),
                        **block_params
                    ))
                for i in range(self.blocks_per_layer[level]-1):
                    blocks_level.append(
                        self.block(
                            self.hidden_dims[level],
                            self.hidden_dims[level],
                            d_pos=self.d_pos * 2 ** level,
                            num_groups=self.num_groups[level],
                            **block_params
                        )
                    )
            else:
                for i in range(self.blocks_per_layer[level]):
                    blocks_level.append(
                        self.block(
                            self.hidden_dims[level],
                            self.hidden_dims[level],
                            d_pos=self.d_pos * 2 ** level,
                            num_groups=self.num_groups[level],
                            **block_params
                        )
                    )
            self.encoder_blocks.append(nn.ModuleList(blocks_level))

        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        #Decoder
        for level in range(len(self.hidden_dims) - 2, -1, -1):
            self.upsamples.append(
                nn.Upsample(mode='linear',align_corners=True)
            )

            blocks_level = [self.block(
                        self.hidden_dims[level]+self.hidden_dims[level+1],
                        self.hidden_dims[level],
                        d_pos=self.d_pos * 2 ** level,
                        num_groups=self.num_groups[level],
                        upproject = nn.Sequential(nn.Conv1d(self.hidden_dims[level]+self.hidden_dims[level+1],
                                                            self.hidden_dims[level],
                                                            kernel_size=1, bias=has_bias),
                                                  get_norm_layer(norm,self.hidden_dims[level], self.num_groups[level])),
                        **block_params
                    )]
            for i in range(self.blocks_per_layer[level]-1):
                blocks_level.append(
                    self.block(
                        self.hidden_dims[level],
                        self.hidden_dims[level],
                        d_pos=self.d_pos * 2 ** level,
                        num_groups=self.num_groups[level],
                        **block_params
                    )
                )
            self.decoder_blocks.append(nn.ModuleList(blocks_level))

        # Task Heads
        cls_input_dim = self.hidden_dims[0]*2 if self.concat_local_features else self.hidden_dims[0]
        reg_input_dim = cls_input_dim + num_classes if self.concat_class_probs else cls_input_dim
        self.cls_head = self._initialise_task_head(num_cls_layers, cls_input_dim, out_dim, num_classes,
                                                   dropout=task_dropout, head_type=head_type)
        self.reg_head = self._initialise_task_head(num_reg_layers, reg_input_dim, out_dim, num_targets=2, dropout=task_dropout,
                                                   head_type=head_type)
        self.loss_weights = torch.nn.Parameter(torch.ones(3))

    def _initialise_task_head(self, num_layers, in_channels, out_channels, num_targets, dropout, head_type='mlp'):
        if head_type == 'mlp':
            layers = [nn.Conv1d(in_channels, out_channels, kernel_size=1), self.activation, self.dropout]
            for i in range(num_layers - 1):
                layers.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))
                layers.append(self.activation),
                layers.append(self.dropout)
            layers.append(nn.Conv1d(out_channels, num_targets, kernel_size=1))
            return nn.Sequential(*layers)
        else:
            return self.block(in_channels, num_targets,
                              dropout=dropout, num_groups=self.num_groups[0], norm='group')

    def forward(self, x):

        input_dims = []
        coords = self.diff_encoder(x) if self.use_diff_encoder else x
        coord_list = [coords]
        for coords_down,pool in zip(self.coords_pathway,self.coord_pooling_layers) or []:
            if pool is not None:
                pool.output_size = coords.shape[-1]//2
            coords = pool(coords_down(coords))
            coord_list.append(coords)

        #Encoder
        x = self.activation(self.norm_embed(self.triangle_embedding(coord_list[0])))
        skip_connections = []
        for block in self.encoder_blocks_layer1:
            x = block(x,coord_list[0])
        skip_connections.append(x)
        input_dims.append(x.shape[-1])

        for idx, (blocks,downsample) in enumerate(zip(self.encoder_blocks,self.downsamples)):
            level = idx+1
            if self.downsample == 'maxpool':
                downsample.output_size = x.shape[-1]//2
            x = downsample(x)
            for block in blocks:
                coords = coord_list[level] if self.coords_pathway else None
                assert x.shape[-1] == coords.shape[-1], f'level: {level}, {x.shape}, {coords.shape}'
                x = block(x, coords)
            if level < self.num_layers-1:
                skip_connections.append(x)
                input_dims.append(x.shape[-1])

        #Decoder
        for i, (blocks, upsample) in enumerate(zip(self.decoder_blocks,self.upsamples)):
            level = self.num_layers - i - 2
            target_size = input_dims[level]
            upsample.size=target_size
            x = upsample(x)
            x = torch.cat([x, skip_connections[level]], dim=1)
            for block in blocks:
                coords = coord_list[level] if self.coords_pathway else None
                x = block(x,coords)


        task_input = torch.cat([x, skip_connections[0]], dim=1) if self.concat_local_features else x
        cls_out = self.cls_head(task_input)
        if self.concat_class_probs:
            probs = self.softmax(cls_out)
            task_input = torch.cat([task_input, probs], dim=1)
        reg_out = self.reg_head(task_input)

        if self.multiply_labels:
            cls_labels = cls_out.argmax(1)
            move_labels = torch.where(cls_labels==2,1,0)
            reg_out = reg_out*move_labels[:,None,:]

        return cls_out, reg_out[:, 0, :], reg_out[:, 1, :]