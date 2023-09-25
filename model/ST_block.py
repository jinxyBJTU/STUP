import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np

class Spatial_Aggregation_wto_Uncer(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels,dropout=.0):
        super(Spatial_Aggregation_wto_Uncer, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.Theta = nn.Linear(in_channels, in_channels, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x,spatial_embedding=None):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, F_in)
        :return: (batch_size, N, F_out)
        '''
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        x = x.permute(0, 2, 1, 3)     #  (B T V F)
        x = x.reshape(-1,num_of_vertices,in_channels)

        if spatial_embedding!=None:
            relation_matrix = F.relu(torch.matmul(spatial_embedding, spatial_embedding.transpose(-2, -1))) #  (N,N)
            relation_matrix = self.dropout(F.softmax(relation_matrix, dim=-1)).unsqueeze(0)
            
            x = F.relu(self.Theta(torch.matmul(relation_matrix , x))) 
            x = x.reshape(batch_size, num_of_timesteps, num_of_vertices, self.out_channels).transpose(1, 2)
        else:
            temp = F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x)))
            x = temp.reshape(batch_size, num_of_timesteps, num_of_vertices, self.out_channels).transpose(1, 2)

        return self.dropout(x)

class Spatial_Aggregation(nn.Module):
    def __init__(self, sharing_paras, topology_adj, channels ,dropout=.0):
        super(Spatial_Aggregation, self).__init__()
        self.topo_Adj_matrix = topology_adj     #+ torch.eye(len(topology_adj)).to(topology_adj.device)
        self.in_channels = channels
        self.out_channels = channels

        self.linear1 = nn.Linear(channels, channels, bias=False)
        if not sharing_paras:
            self.linear2 = nn.Linear(channels, channels, bias=False)
        
        self.sharing_paras = sharing_paras
        self.dropout = nn.Dropout(dropout)

    def forward(self, ori_flow, uncer, spatial_embedding):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, T, f)
        :param uncer: (batch_size, N, T, f)
        :param spatial_embedding: (N, f)
        :return: (batch_size, N, T, f)
        '''
       
        batch_size, num_of_vertices, num_of_timesteps, in_channels = ori_flow.shape
        
        #  (B V T F)-> (B T V F)-> (B*T, V F)
        ori_flow = ori_flow.permute(0, 2, 1, 3).reshape(-1, num_of_vertices, in_channels)    
        uncer = uncer.permute(0, 2, 1, 3).reshape(-1, num_of_vertices, in_channels) 

        # attention
        if spatial_embedding!=None:

            spatial_embedding = spatial_embedding.unsqueeze(0).unsqueeze(2).expand(batch_size,num_of_vertices,num_of_timesteps,(in_channels))
            spatial_embedding = spatial_embedding.permute(0, 2, 1, 3).reshape(-1, num_of_vertices, in_channels)
            
            temp_hidden_state = torch.cat([ori_flow,spatial_embedding],dim=-1)

            relation_matrix_h = F.relu(torch.matmul(temp_hidden_state, temp_hidden_state.transpose(-2, -1))) #  (N,N)
            relation_matrix_h = self.dropout(F.softmax(relation_matrix_h, dim=-1))
           
        # sharing paras
        if self.sharing_paras:
            if spatial_embedding!=None:
                ori_flow = F.relu(self.linear1(torch.matmul(relation_matrix_h , ori_flow))) 
                uncer = F.relu(self.linear1(torch.matmul(relation_matrix_h , uncer)))
            else:
                ori_flow = F.relu(self.linear1(torch.matmul(self.topo_Adj_matrix, ori_flow)))
                uncer = F.relu(self.linear1(torch.matmul(self.topo_Adj_matrix, uncer))) 
        else:
            if spatial_embedding!=None:
                ori_flow = F.relu(self.linear1(torch.matmul(self.topo_Adj_matrix, ori_flow)))
                uncer = F.relu(self.linear2(torch.matmul(self.topo_Adj_matrix, uncer)))
            else:
                ori_flow = F.relu(self.linear1(torch.matmul(self.topo_Adj_matrix, ori_flow)))
                uncer = F.relu(self.linear2(torch.matmul(self.topo_Adj_matrix, uncer))) 

        ori_flow = ori_flow.reshape(batch_size, num_of_timesteps, num_of_vertices, self.out_channels).transpose(1, 2)
        uncer = uncer.reshape(batch_size, num_of_timesteps, num_of_vertices, self.out_channels).transpose(1, 2)
        
        return ori_flow, uncer


class Temporal_Aggregation_wto_Uncer(nn.Module):
    def __init__(self, nb_head, d_model, kernel_size=3, dropout=.0):
        super(Temporal_Aggregation_wto_Uncer, self).__init__()
        
        self.h = nb_head # 8
        self.d_k = (d_model) // nb_head # 8
        self.padding = (kernel_size - 1)//2
        
        self.conv1D_aware_temporal_context1 = nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding))
        self.linear_output1 = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, value, temporal_embedding=None):
        """
        query: (B,N,T,F)
        temporal_embedding: (B,T,F/2)
        return ：(B,N,T,F)
        """
        nbatches, num_of_vertices, T, d_model = value.shape
        # (batch, N, T, d_model)->(batch, d_model, N, T)->(batch, N, h, T, d_k)
        value = self.conv1D_aware_temporal_context1(value.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, num_of_vertices, -1).permute(0, 3, 1, 4, 2) 

        if temporal_embedding!=None:
            temporal_embedding = temporal_embedding.unsqueeze(1).expand(nbatches,num_of_vertices,T,d_model)
            temporal_embedding = temporal_embedding.view(nbatches, num_of_vertices, -1, self.h, self.d_k).transpose(2, 3) # (B, N, h, T, D)
            
            temp_hidden_state = torch.cat([value , temporal_embedding],dim=-1)
            relation_matrix_h = F.relu(torch.matmul(temp_hidden_state, temp_hidden_state.transpose(-2, -1)))    #  (B, T, T)      
            relation_matrix_h = self.dropout(F.softmax(relation_matrix_h, dim=-1)) 
        else:
            relation_matrix_h = torch.eye(T)+torch.triu(torch.ones(T,T),diagonal=1)-torch.triu(torch.ones(T,T),diagonal=2)+torch.tril(torch.ones(T,T),diagonal=-1)-torch.tril(torch.ones(T,T),diagonal=-2)
            relation_matrix_h = relation_matrix_h.to(value.device).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(nbatches,num_of_vertices,self.h,T,T)
        
        value = torch.matmul(relation_matrix_h, value) # BNHTT
        
        value = value.transpose(2, 3).contiguous()  # (B, N, h, T, D) -> (B, N, T, h, D) 
        value = value.view(nbatches, num_of_vertices, -1, self.h * self.d_k)  # (batch, N, T, d_model)

        value = F.relu(self.linear_output1(value))

        return value


class Temporal_Aggregation(nn.Module):
    def __init__(self, sharing_paras, nb_head, d_model, kernel_size=3, dropout=.0):
        super(Temporal_Aggregation, self).__init__()
        
        self.h = nb_head # 8
        self.d_k = (d_model) // nb_head # 8
        self.padding = (kernel_size - 1)//2
        
        self.conv1D_aware_temporal_context1 = nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding))
        self.linear_output1 = nn.Linear(d_model, d_model)
        if not sharing_paras:
            self.conv1D_aware_temporal_context2 = nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding))
            self.linear_output2 = nn.Linear(d_model, d_model)
        self.sharing_paras = sharing_paras
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, value, uncer, temporal_embedding=None):
        """
        query: (B,N,T,F)
        temporal_embedding: (B,T,F/2)
        return ：(B,N,T,F)
        """
        nbatches, num_of_vertices, T, d_model = value.shape
        # (batch, N, T, d_model)->(batch, d_model, N, T)->(batch, N, h, T, d_k)
        value = self.conv1D_aware_temporal_context1(value.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, num_of_vertices, -1).permute(0, 3, 1, 4, 2) 
        if self.sharing_paras :
            uncer = self.conv1D_aware_temporal_context1(uncer.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, num_of_vertices, -1).permute(0, 3, 1, 4, 2) 
        else:
            uncer = self.conv1D_aware_temporal_context2(uncer.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, num_of_vertices, -1).permute(0, 3, 1, 4, 2) 
        

        if temporal_embedding!=None:
            temporal_embedding = temporal_embedding.unsqueeze(1).expand(nbatches,num_of_vertices,T,d_model)
            temporal_embedding = temporal_embedding.view(nbatches, num_of_vertices, -1, self.h, self.d_k).transpose(2, 3) # (B, N, h, T, D)
            
            temp_hidden_state = torch.cat([value , temporal_embedding],dim=-1)
            relation_matrix_h = F.relu(torch.matmul(temp_hidden_state, temp_hidden_state.transpose(-2, -1)))    #  (B, T, T)      
            relation_matrix_h = self.dropout(F.softmax(relation_matrix_h, dim=-1)) 
        else:
            relation_matrix_h = torch.eye(T)+torch.triu(torch.ones(T,T),diagonal=1)-torch.triu(torch.ones(T,T),diagonal=2)+torch.tril(torch.ones(T,T),diagonal=-1)-torch.tril(torch.ones(T,T),diagonal=-2)
            relation_matrix_h = relation_matrix_h.to(value.device).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(nbatches,num_of_vertices,self.h,T,T)
        
        # print(value.shape) 
        # print(relation_matrix_h.shape)
        # exit()
        value = torch.matmul(relation_matrix_h, value) # BNHTT
        uncer = torch.matmul(relation_matrix_h, uncer)
       
        value = value.transpose(2, 3).contiguous()  # (B, N, h, T, D) -> (B, N, T, h, D) 
        value = value.view(nbatches, num_of_vertices, -1, self.h * self.d_k)  # (batch, N, T, d_model)

        uncer = uncer.transpose(2, 3).contiguous()  # (B, N, h, T, D) -> (B, N, T, h, D) 
        uncer = uncer.view(nbatches, num_of_vertices, -1, self.h * self.d_k)  # (batch, N, T, d_model)

        value = F.relu(self.linear_output1(value))
        if self.sharing_paras : 
            uncer = F.relu(self.linear_output1(uncer))
        else:
            uncer = F.relu(self.linear_output2(uncer))

        return value,uncer




class Temporal_Aggregation_attenTCNN(nn.Module):
    def __init__(self, nb_head, d_model, kernel_size=3, dropout=.0):
        super(Temporal_Aggregation_attenTCNN, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linear_value = nn.Linear(d_model, d_model) 
        self.linear_output = nn.Linear(d_model, d_model)

        self.padding = (kernel_size - 1)//2
        self.conv1Ds_aware_temporal_context = nn.ModuleList()
        # 2 causal conv: 1  for query, 1 for key
        for _ in (range(2)):
            self.conv1Ds_aware_temporal_context.append(
                nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding)) ) 
            
        self.dropout = nn.Dropout(p=dropout)
                

    def attention(self,query, key, value):
        '''
        :param query:  (batch, N, h, T, d_k)
        :param key: (batch, N, h, T, d_k)
        :param value: (batch, N, h, T, d_k)
        '''
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # scores: (batch, N, h, T1, T2)

        p_attn = self.dropout(F.softmax(scores, dim=-1)) 
       
        return torch.matmul(p_attn, value), p_attn  # (batch, N, h, T1, d_k), (batch, N, h, T1, T2)
            

    def forward(self, query, key, value):
        """
        x: (B,N,T,F)
        return ：(B,N,T,F)
        """
        nbatches,num_of_vertices,_,_ = query.shape
        # (batch, N, T, d_model)->(batch, d_model, N, T) ->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)

        query, key = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, num_of_vertices, -1).permute(0, 3, 1, 4, 2) 
                for l, x in zip(self.conv1Ds_aware_temporal_context, (query, key))]

        value = self.linear_value(value).view(nbatches, num_of_vertices, -1, self.h, self.d_k).transpose(2, 3)
        x, self.attn = self.attention(query, key, value)
        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, num_of_vertices, -1, self.h * self.d_k)  # (batch, N, T1, d_model)
        
        return self.linear_output(x)
