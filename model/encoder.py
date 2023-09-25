import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np


class Uncer_atten_cell(nn.Module):
    def __init__(self,d_model):
        super(Uncer_atten_cell,self).__init__()
        
        self.query_linear = nn.Linear(d_model,d_model)
        self.key_linear = nn.Linear(d_model,d_model)
        self.value_pre = nn.Linear(d_model,d_model)
        self.value_uncer = nn.Linear(d_model,d_model)
        self.d_model = d_model//2

    def forward(self,pre_hidden_state,uncer_state): 
        
        B,N,T,channels = pre_hidden_state.shape
        query , key =  self.query_linear(pre_hidden_state),self.key_linear(uncer_state) # B N T F

        pre = self.value_pre(pre_hidden_state)
        uncer = self.value_uncer(uncer_state)

        query_temporal,query_spatial = query[:,:,:,:self.d_model],query[:,:,:,self.d_model:] # B N T channels//2
        key_temporal,key_spatial = key[:,:,:,:self.d_model],key[:,:,:,self.d_model:] # B N T channels//2
        pre_temporal,pre_spatial = pre[:,:,:,:self.d_model],pre[:,:,:,self.d_model:] # B N T channels//2
        uncer_temporal,uncer_spatial = uncer[:,:,:,:self.d_model],uncer[:,:,:,self.d_model:] # B N T channels//2
        

        # temporal
        query_temporal = query_temporal.reshape(-1,T,channels//2)     #  (B V T F)->(B*V T F)
        key_temporal = key_temporal.reshape(-1,T,channels//2)     #  (B V T F)->(B*V T F)
        pre_temporal = pre_temporal.reshape(-1,T,channels//2)     #  (B V T F)->(B*V T F)
        uncer_temporal = uncer_temporal.reshape(-1,T,channels//2)     #  (B V T F)->(B*V T F)
       
        relation_temporal = F.softmax(torch.matmul(query_temporal, key_temporal.transpose(-2, -1)), dim=-1) 
        pre_temporal = torch.matmul(relation_temporal,pre_temporal)
        uncer_temporal = torch.matmul(relation_temporal.transpose(-2, -1),uncer_temporal)

        pre_temporal = pre_temporal.reshape(B,N,T,channels//2) 
        uncer_temporal = uncer_temporal.reshape(B,N,T,channels//2) 

        # spatial
        query_spatial = query_spatial.permute(0, 2, 1, 3).reshape(-1,N,channels//2)     #  (B V T F)->(B T V F)->(B*T V F)
        key_spatial = key_spatial.permute(0, 2, 1, 3).reshape(-1,N,channels//2)     #  (B V T F)->(B T V F)->(B*T V F)
        pre_spatial = pre_spatial.permute(0, 2, 1, 3).reshape(-1,N,channels//2)     #  (B V T F)->(B T V F)->(B*T V F)
        uncer_spatial = uncer_spatial.permute(0, 2, 1, 3).reshape(-1,N,channels//2)     #  (B V T F)->(B T V F)->(B*T V F)

        relation_spatial = F.softmax(torch.matmul(query_spatial, key_spatial.transpose(-2, -1)), dim=-1) 
        pre_spatial = torch.matmul(relation_spatial,pre_spatial)
        uncer_spatial = torch.matmul(relation_spatial.transpose(-2, -1),uncer_spatial)
        pre_spatial = pre_spatial.reshape(B,T,N,channels//2).permute(0, 2, 1, 3) 
        uncer_spatial = uncer_spatial.reshape(B,T,N,channels//2).permute(0, 2, 1, 3) 

        pre = torch.cat((pre_temporal,pre_spatial),dim=-1)
        uncer = torch.cat((uncer_temporal,uncer_spatial),dim=-1)
 
        return pre,uncer

class Uncer_GRU_cell(nn.Module):
    def __init__(self,d_model):
        super(Uncer_GRU_cell,self).__init__()
        self.gate = nn.Linear(2*d_model,2*d_model)
        self.update_uncer = nn.Linear(2*d_model,d_model)
        self.update_pre = nn.Linear(2*d_model,d_model)
        self.hidden_dim = d_model

    def get_init_stat(self,pre_hidden_state):
        # 最后改一下分布，换成正态分布试一下
        return torch.zeros_like(pre_hidden_state).to(pre_hidden_state.device)

    def forward(self,pre_hidden_state,uncer_state): 
        """
        input:  x[i]:(B,N,T,D)
        return: (B,N,T,D)
        """
        
        input_and_state = torch.cat((uncer_state, pre_hidden_state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)

        uncertainty_hat = torch.tanh(self.update_uncer(torch.cat((pre_hidden_state, r*uncer_state), dim=-1)))
        pre_hat = torch.tanh(self.update_pre(torch.cat((uncer_state, r*pre_hidden_state), dim=-1)))

        uncer_state =  (1-z)*uncer_state + z*uncertainty_hat
        pre_hidden_state =  (1-z)*pre_hidden_state + z*pre_hat

        return pre_hidden_state,uncer_state

class Encoder_wto_Uncer(nn.Module):
    def __init__(self, gcn, tcnn,d_model, blocks = 4,dropout=.0):
        super(Encoder_wto_Uncer, self).__init__()

        self.filter_convs = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.tcnn_norm = nn.ModuleList()
        self.gconv_norm = nn.ModuleList()

        self.tcnn_dropout = nn.ModuleList() 
        self.gconv_dropout = nn.ModuleList()

        self.blocks = blocks
        for _ in range(blocks):
            self.tcnn_norm.append(nn.LayerNorm(d_model))
            self.tcnn_dropout.append(nn.Dropout(dropout))
            self.filter_convs.append(copy.deepcopy(tcnn))           # ,padding=(0, self.padding)
            
            self.gconv_norm.append(nn.LayerNorm(d_model))
            self.gconv_dropout.append(nn.Dropout(dropout))
            self.gconv.append(copy.deepcopy(gcn))

        self.final_norm_pre = nn.LayerNorm(d_model)
        
    def forward(self, x ,spatial_embedding, temporal_embedding):
        """
        x: (B,N,T,F)
        return ：(B,N,T,F)
        """
        hidden_state = None

        # if save:
        #     hidden_state = []
        #     hidden_state.append(x.unsqueeze(1).detach().cpu().numpy())

        for i in range(self.blocks):
            
            norm_x = self.tcnn_norm[i](x)

            x = self.tcnn_dropout[i](self.filter_convs[i](norm_x, temporal_embedding)) + x
            
            norm_x = self.gconv_norm[i](x)
            x  = self.gconv_dropout[i](self.gconv[i](norm_x, spatial_embedding)) + x
            
        return self.final_norm_pre(x)#,hidden_state

class Encoder_with_Uncer(nn.Module):
    def __init__(self,sharing_paras, mid_gate, gcn, tcnn, uncer_gru_cell, d_model, blocks = 4,dropout=.0):
        super(Encoder_with_Uncer, self).__init__()

        self.sharing_paras = sharing_paras
        self.mid_gate = mid_gate

        self.tempo_convs = nn.ModuleList()
        self.gra_conv = nn.ModuleList()
        self.tcnn_norm1 = nn.ModuleList()
        self.gconv_norm1 = nn.ModuleList()
        if not sharing_paras:
            self.tcnn_norm2 = nn.ModuleList()
            self.gconv_norm2 = nn.ModuleList()
        if mid_gate:
            self.uncer_gru_cell = nn.ModuleList()

        # self.tcnn_dropout = nn.ModuleList() 
        # self.gconv_dropout = nn.ModuleList()
        
        self.blocks = blocks
        for _ in range(blocks):
            self.tcnn_norm1.append(nn.LayerNorm(d_model))
            self.tempo_convs.append(copy.deepcopy(tcnn))                # ,padding=(0, self.padding)
            
            self.gconv_norm1.append(nn.LayerNorm(d_model))
            self.gra_conv.append(copy.deepcopy(gcn))
            if not sharing_paras:
                self.tcnn_norm2.append(nn.LayerNorm(d_model))
                self.gconv_norm2.append(nn.LayerNorm(d_model))

            if mid_gate:
                self.uncer_gru_cell.append(copy.deepcopy(uncer_gru_cell))   # ,padding=(0, self.padding)
        
        self.final_norm_pre = nn.LayerNorm(d_model)
        if not sharing_paras:
            self.final_norm_uncer = nn.LayerNorm(d_model)
        
    def forward(self, x, init_uncer_state,spatial_embedding, temporal_embedding):
        """
        x: (B,N,T,F)
        return ：(B,N,T,F)
        """
        # skip = torch.tensor(0.).to(x.device)

        # uncer_state = self.uncer_gru_cell[0].get_init_stat(x)
        uncer_state = init_uncer_state

        for i in range(self.blocks):

            # norm + residual
            norm_x = self.tcnn_norm1[i](x)
            if self.sharing_paras:
                norm_uncer = self.tcnn_norm1[i](uncer_state)
            else:
                norm_uncer = self.tcnn_norm2[i](uncer_state)

            temp_x, temp_uncer_state = self.tempo_convs[i](norm_x, norm_uncer, temporal_embedding)
            x = x + temp_x
            uncer_state = uncer_state + temp_uncer_state

            # norm + residual
            norm_x = self.gconv_norm1[i](x)
            if self.sharing_paras:
                norm_uncer = self.gconv_norm1[i](uncer_state)
            else:
                norm_uncer = self.gconv_norm2[i](uncer_state)

            temp_x, temp_uncer_state  = self.gra_conv[i](norm_x, norm_uncer, spatial_embedding)
            x = x + temp_x
            uncer_state = uncer_state + temp_uncer_state
            if self.mid_gate:
                x , uncer_state =  self.uncer_gru_cell[i](x, uncer_state)


        traffic_state = self.final_norm_pre(x)
        if self.sharing_paras:
            uncer_state = self.final_norm_pre(uncer_state)
        else:
            uncer_state = self.final_norm_uncer(uncer_state)
        return  traffic_state , uncer_state