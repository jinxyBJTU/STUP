"""
this is a pro version of clu
add interaction of uncertainty and prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
from lib.utils import norm_Adj,sym_norm_Adj
from model.positional_embedding import  SpatialPositionalEncoding,TemporalPositionalEncoding
from model.encoder import Encoder_with_Uncer,Encoder_wto_Uncer,Uncer_GRU_cell,Uncer_atten_cell
from model.decoder import  TCNN_output_layer,Gated_Output_layer
from model.ST_block import Spatial_Aggregation,Temporal_Aggregation
from torch.autograd import Variable

def positivity_constraint(sigma):

    sigma = F.softplus(sigma) + torch.tensor(1e-6).to(sigma.device)
    if (sigma<0).any():
        print('sigma <0')
        exit()
    return sigma

class ST_embedding_proj(nn.Module):
    def __init__(self,encoder_input_size,history_step,d_model,hidden_dim=32,kernel_size=3):
        super(ST_embedding_proj, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.tcnn = nn.ModuleList()

        self.layer = 2
        self.padding = (kernel_size - 1)//2

        for _ in range(self.layer):
            self.tcnn.append(nn.Conv2d(encoder_input_size, encoder_input_size, (1, kernel_size), padding=(0, self.padding)))

        self.linear1 = nn.Linear(history_step*encoder_input_size,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,d_model)

    def forward(self, source_flow ):
        '''
        src:  (batch_size, N, T_in, F_in)
        cluster_centers:  (C, d_cluster)
        trg: (batch, N, T_out, F_out)
        '''  
        # (B N T F) -> (B F N T) -> (B N T*F)            (B C N T) (B C N ) (B N C)  (N C)
        B,N,T,D = source_flow.shape
        module_output = source_flow.permute(0,3,1,2) 
        for i in range(self.layer):
            module_output = self.tcnn[i](module_output)
        
        module_output = module_output.permute(0,2,3,1).reshape(B,N,-1) 
        module_output = self.linear2(F.relu(self.linear1(module_output)))

        return module_output

class EncoderDecoder(nn.Module):
    def __init__(self, input_mapping , encoder , final_gate,
                stib, sampling, spatial, temporal,  
                num_of_vertices, num_of_timeslices, d_model, DEVICE):
        super(EncoderDecoder,self).__init__()
        self.input_mapping = input_mapping
        self.encoder = encoder
        self.final_gate = final_gate
        self.stib = stib
        self.sampling = sampling
        self.spatial_embedding = None
        self.temporal_embedding = None
        # hyper-para
        if spatial:
            self.spatial_embedding = torch.nn.Embedding(num_of_vertices, d_model)
        if temporal:
            self.temporal_embedding = torch.nn.Embedding(num_of_timeslices, d_model)

        if self.stib:
            self.linear_uncer_1 = nn.Linear(d_model,d_model)
            if self.sampling:
                self.linear_uncer_2 = nn.Linear(d_model,2*d_model) # 2*
            else:
                self.linear_uncer_2 = nn.Linear(d_model,d_model) # 2*

        self.num_of_vertices = num_of_vertices
        self.d_model = d_model
        self.device = DEVICE
        self.to(DEVICE)

    def get_stEmbeddings(self):
        return self.spatial_embedding , self.temporal_embedding

    def generate_init_uncertainty(self, timestamps):
        batch_size,history_step = timestamps.shape
        
        total_embeddings = 0
        spatial_embedding = None
        temporal_embedding = None
        if self.spatial_embedding:
            spatial_indexs = torch.LongTensor(torch.arange(self.num_of_vertices)).to(timestamps.device)  # (N,)
            spatial_embedding = self.spatial_embedding(spatial_indexs) # (N, d_model)->(1,N,1,d_model)
            total_embeddings += spatial_embedding.unsqueeze(0).unsqueeze(2).expand(batch_size,self.num_of_vertices, history_step, self.d_model)
        if self.temporal_embedding:
            temporal_embedding = self.temporal_embedding(timestamps)  # (B,T, d_model)->(B, 1, T, d_model)
            total_embeddings += temporal_embedding.unsqueeze(1).expand(batch_size,self.num_of_vertices, history_step, self.d_model)

        if self.stib:
            if self.sampling:
                miu_sigma = self.linear_uncer_2(F.relu(self.linear_uncer_1(total_embeddings)))
                miu, sigma = miu_sigma[:,:,:,:self.d_model], miu_sigma[:,:,:,self.d_model:]
                epsilon = torch.rand_like(sigma)
                uncertainty = miu + epsilon*sigma
            else:
                uncertainty = self.linear_uncer_2(F.relu(self.linear_uncer_1(total_embeddings)))
        else:

            uncertainty = torch.rand_like(total_embeddings)

        return uncertainty, spatial_embedding, temporal_embedding

    def forward(self,source_flow,timestamps): 
        '''
        src:  (batch_size, N, T_in, F_in)
        trg: (batch, N, T_out, F_out)
        '''  
        B,N,T,D = source_flow.shape
        
        hidden_embedding = self.input_mapping(source_flow)
        init_uncertainty, spatial_embedding, temporal_embedding = self.generate_init_uncertainty(timestamps)
        # print(init_uncertainty.shape)
        # exit()
        pre_hidden_state, uncer_hidden_state = self.encoder(hidden_embedding, init_uncertainty, spatial_embedding, temporal_embedding)

        output, sigma = self.final_gate(pre_hidden_state,uncer_hidden_state)
        return output, positivity_constraint(sigma)   
        
def make_model(DEVICE, adj_mx, adj_with_self_loop, num_of_vertices,history_step,num_for_predict,
            sharing_paras, stib, sampling, spatial_embedding,temporal_embedding, mid_gate, output_gate,
            encoder_input_size, encoder_output_size, d_model, nb_head, num_of_timeslices,
            num_layers=4, dropout=.0,kernel_size=3):

    c = copy.deepcopy
    sym_norm_adj_loop , sym_norm_adj_wto_loop ,  W_loop , W_wto_loop = sym_norm_Adj(adj_mx , adj_with_self_loop)
    self_loop_adj = torch.from_numpy(W_loop).type(torch.FloatTensor).to(DEVICE)
    sym_norm_adj_loop = torch.from_numpy(sym_norm_adj_loop).type(torch.FloatTensor).to(DEVICE)

    input_mapping = nn.Sequential(nn.Linear(encoder_input_size, d_model))
    gcn = Spatial_Aggregation(sharing_paras, self_loop_adj , d_model , dropout)
    tcnn = Temporal_Aggregation(sharing_paras, nb_head, d_model, kernel_size=kernel_size, dropout=dropout)
   
    uncer_gre_cell = None
    if mid_gate:
        uncer_gre_cell = Uncer_GRU_cell(d_model)
    
    encoder = Encoder_with_Uncer(sharing_paras,mid_gate, gcn, tcnn, uncer_gre_cell, d_model, blocks = num_layers,dropout=dropout)
    final_gate = Gated_Output_layer(sharing_paras, output_gate, d_model, encoder_output_size, history_step, num_for_predict )

    model = EncoderDecoder(
                input_mapping, encoder,final_gate,     
                stib,sampling,spatial_embedding,temporal_embedding,                         
                num_of_vertices,num_of_timeslices,d_model,DEVICE
                )
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model 
