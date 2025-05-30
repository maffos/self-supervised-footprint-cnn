from torch import nn
import torch
from src.models import TriangleConv, TriangleFeatureExtractor, TriangleFeatureBlock
from src.utils import get_activation, get_norm_layer

class UNetTriangleModelRegression(nn.Module):
    def __init__(self, triangle_embed_params,
                 block_params,
                 in_channels = 2,
                 num_classes = 7,
                 norm='batch',
                 initial_num_groups=None,
                 hidden_dims=None,
                 activation='relu',
                 use_diff_encoder=False,
                 num_layers = None,
                 blocks_per_layer=[1, 1, 1],
                 downsample = 'conv',
                 ):
        super(UNetTriangleModelRegression, self).__init__()

        self.activation = get_activation(activation)
        self.d_model_triangle = triangle_embed_params['out_channels']
        self.hidden_dims = hidden_dims if hidden_dims is not None else [self.d_model_triangle*2**i for i in range(num_layers)]
        self.num_layers = len(self.hidden_dims)
        self.diff_encoder = TriangleConv()
        self.use_diff_encoder = use_diff_encoder
        self.num_groups = [initial_num_groups*2**i for i in range(self.num_layers)]
        self.downsample = downsample
        self.blocks_per_layer = blocks_per_layer if isinstance(blocks_per_layer, list) else [blocks_per_layer for _ in range(self.num_layers)]

        input_channels = 4*in_channels if self.use_diff_encoder else in_channels
        self.d_pos = input_channels
        has_bias = (norm == None)

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
                TriangleFeatureBlock(
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
                blocks_level.append(TriangleFeatureBlock(
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
                        TriangleFeatureBlock(
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
                        TriangleFeatureBlock(
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

            blocks_level = [TriangleFeatureBlock(
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
                    TriangleFeatureBlock(
                        self.hidden_dims[level],
                        self.hidden_dims[level],
                        d_pos=self.d_pos * 2 ** level,
                        num_groups=self.num_groups[level],
                        **block_params
                    )
                )
            self.decoder_blocks.append(nn.ModuleList(blocks_level))
        self.out_layer = nn.Conv1d(self.hidden_dims[0],num_classes,1)

    def _initialise_task_head(self, num_layers, in_channels, out_channels, num_targets, dropout, head_type='mlp'):
        if head_type == 'mlp':
            layers = [nn.Conv1d(in_channels, out_channels, kernel_size=1), self.activation, nn.Dropout(dropout)]
            for i in range(num_layers - 1):
                layers.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))
                layers.append(self.activation)
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Conv1d(out_channels, num_targets, kernel_size=1))
            return nn.Sequential(*layers)
        else:
            return TriangleFeatureBlock(in_channels, num_targets,
                              dropout=dropout, num_groups=self.num_groups[0], norm='group')

    def forward(self, x):

        input_dims = []
        coords = self.diff_encoder(x) if self.use_diff_encoder else x
        coord_list = [coords]
        for coords_down, pool in zip(self.coords_pathway, self.coord_pooling_layers) or []:
            if pool is not None:
                pool.output_size = coords.shape[-1] // 2
            coords = pool(coords_down(coords))
            coord_list.append(coords)

        # Encoder
        x = self.activation(self.norm_embed(self.triangle_embedding(coord_list[0])))
        skip_connections = []
        for block in self.encoder_blocks_layer1:
            x = block(x, coord_list[0])
        skip_connections.append(x)
        input_dims.append(x.shape[-1])

        # x = local_embeddings
        for idx, (blocks, downsample) in enumerate(zip(self.encoder_blocks, self.downsamples)):
            level = idx + 1
            if self.downsample == 'maxpool':
                downsample.output_size = x.shape[-1] // 2
            x = downsample(x)
            for block in blocks:
                coords = coord_list[level] if self.coords_pathway else None
                assert x.shape[-1] == coords.shape[-1], f'level: {level}, {x.shape}, {coords.shape}'
                x = block(x, coords)
            if level < self.num_layers - 1:
                skip_connections.append(x)
                input_dims.append(x.shape[-1])

        # Decoder
        for i, (blocks, upsample) in enumerate(zip(self.decoder_blocks, self.upsamples)):
            level = self.num_layers - i - 2
            target_size = input_dims[level]
            upsample.size = target_size
            x = upsample(x)
            x = torch.cat([x, skip_connections[level]], dim=1)
            for block in blocks:
                coords = coord_list[level] if self.coords_pathway else None
                x = block(x, coords)

        x = self.out_layer(x)
        return x