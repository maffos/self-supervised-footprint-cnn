from src.model_building_blocks import TriangleFeatureBlock, TriangleFeatureExtractor
from src.utils import get_norm_layer,get_activation
from torch import nn

class BuildingClassificationModel(nn.Module):

    def __init__(self,triangle_embed_params,
                 block_params,
                 num_classes=10,
                 in_channels=2,
                 blocks_per_layer = [1,1],
                 hidden_dims=None,
                 downsample = None
                 ):

        super(BuildingClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.blocks_per_layer = blocks_per_layer
        self.norm = block_params.get('norm', None)
        self.bias = self.norm is None
        self.downsample = downsample
        self.num_groups = block_params.get('num_groups', None)
        self.hidden_dims = hidden_dims if hidden_dims else [self.triangle_feature_dim * 2 ** i for i in range(len(blocks_per_layer) + 1)]
        self.triangle_feature_dim = triangle_embed_params['out_channels']
        assert self.hidden_dims[0] == self.triangle_feature_dim, 'First Hidden layer needs to have same size as triangle feature embedding'
        self.triangle_embedding = TriangleFeatureExtractor(self.in_channels,swap_channels=False,
                                                            input_downsample=False, **triangle_embed_params)

        self.triangle_norm = get_norm_layer(triangle_embed_params['norm'], self.triangle_feature_dim, triangle_embed_params['num_groups'])
        self.triangle_act = get_activation(triangle_embed_params['activation'])
        self.layer_list = nn.ModuleList()
        self.act = get_activation(block_params['activation'])

        #create layer list
        for layer in range(len(self.hidden_dims)-1):
            block_list = nn.ModuleList()
            for _ in range(blocks_per_layer[layer]-1):
                block_list.append(TriangleFeatureBlock(self.hidden_dims[layer], self.hidden_dims[layer],**block_params))
            #last block of layer
            block_list.append(TriangleFeatureBlock(self.hidden_dims[layer], self.hidden_dims[layer+1],**block_params))
            self.layer_list.append(block_list)

        # last layer
        if len(blocks_per_layer) == len(hidden_dims):
            layer = len(hidden_dims) - 1
            block_list = nn.ModuleList()
            for _ in range(blocks_per_layer[-1]):
                block_list.append(TriangleFeatureBlock(self.hidden_dims[layer], self.hidden_dims[layer],**block_params))

            self.layer_list.append(block_list)

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(self.hidden_dims[-1], num_classes)


    def forward(self, x):

        x = self.triangle_act(self.triangle_norm(self.triangle_embedding(x)))
        for layer in self.layer_list:
            for block in layer:
                x = block(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x