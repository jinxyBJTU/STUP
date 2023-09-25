import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

class GCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels,dropout=.0):
        super(GCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, F_in)
        :return: (batch_size, N, F_out)
        '''
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels)) # (b, n, t, f)-permute->(b, t, n, f)->(b*t,n,f_in)
        x = self.dropout(F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x))))
        x = x.reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2)
        
        return x

class SpatialPositionalEncoding(nn.Module):
    def __init__(self, d_model, num_of_vertices, dropout, gcn=None, smooth_layer_num=1):
        super(SpatialPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = torch.nn.Embedding(num_of_vertices, d_model)
        self.gcn_smooth_layers = None
        if (gcn is not None) and (smooth_layer_num > 0):
            self.gcn_smooth_layers = nn.ModuleList([gcn for _ in range(smooth_layer_num)])

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        batch, num_of_vertices, timestamps, _ = x.shape
        x_indexs = torch.LongTensor(torch.arange(num_of_vertices)).to(x.device)  # (N,)
        embed = self.embedding(x_indexs).unsqueeze(0)  # (N, d_model)->(1,N,d_model)
        if self.gcn_smooth_layers is not None:
            for _, l in enumerate(self.gcn_smooth_layers):
                embed = l(embed)  # (1,N,d_model) -> (1,N,d_model)
        x = x + embed.unsqueeze(2)  # (B, N, T, d_model)+(1, N, 1, d_model)
        return self.dropout(x)

class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len, lookup_index=None):
        super(TemporalPositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.lookup_index = lookup_index
        self.max_len = max_len
        # computing the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0).unsqueeze(0)  # (1, 1, T_max, d_model)
        self.register_buffer('pe', pe)
        # register_buffer:
        # Adds a persistent buffer to the module.
        # This is typically used to register a buffer that should not to be considered a model parameter.

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        if self.lookup_index is not None:
            x = x + self.pe[:, :, self.lookup_index, :]  # (batch_size, N, T, F_in) + (1,1,T,d_model)
        else:
            x = x + self.pe[:, :, :x.size(2), :]

        return self.dropout(x.detach())

class Temporal_CNN(nn.Module):
    def __init__(self, d_model,hidden_size,kernel_size=3, dropout=.0):
        super(Temporal_CNN, self).__init__()
        self.padding = (kernel_size - 1)//2
        self.linear1 = nn.Conv2d(d_model, hidden_size, (1, kernel_size), padding=(0, self.padding))
        self.linear2 = nn.Conv2d(hidden_size, d_model, (1, kernel_size), padding=(0, self.padding))
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
         # B N T D ——> permute B D N T ——> conv 
       
        module_output = x.permute(0,3,1,2)
        module_output = F.relu(self.linear2(F.relu(self.linear1(module_output))))
        module_output = module_output.permute(0,2,3,1)
        return self.dropout(x)

def embedding_layer(encoder_input_size,d_model,homo_adj,dropout):

    c = copy.deepcopy

    src_dense = nn.Linear(encoder_input_size, d_model)

    # encode_temporal_position = TemporalPositionalEncoding(d_model, dropout, max_len, en_lookup_index)  # decoder temporal position embedding

    temporal_position = Temporal_CNN(d_model,128,dropout = dropout)

    # spatial_position = SpatialPositionalEncoding(d_model, num_of_vertices, dropout, GCN(homo_adj, d_model, d_model)) 
    spatial_position = GCN(homo_adj, d_model, d_model,dropout = dropout)

    encoder_embedding = nn.Sequential(src_dense , c(temporal_position) , c(spatial_position))

    return encoder_embedding