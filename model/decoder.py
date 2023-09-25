import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
from model.encoder import Uncer_GRU_cell

class TCNN_output_layer(nn.Module):
    def __init__(self, skip_channels,out_dim,history_step,future_step):
        super(TCNN_output_layer, self).__init__()
        end_channels = skip_channels//2
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)
        
        kernel  = history_step-future_step + 1
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,kernel),
                                    bias=True)
    def forward(self, x):
        '''
        x:  (B, N, T, D)
        '''  
        x = x.permute(0,3,1,2)
        x = F.relu(x)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x = x.permute(0,2,3,1) 
        return x

class Multi_class_TCNN_output_layer(nn.Module):
    def __init__(self,node_type, skip_channels,out_dim,history_step,future_step):
        super(Multi_class_TCNN_output_layer, self).__init__()
        end_channels = skip_channels//2
        kernel  = history_step-future_step + 1
        self.end_conv_1 = nn.ModuleList()
        self.end_conv_2 = nn.ModuleList()

        self.node_type = node_type
        self.out_dim = out_dim
        self.future_step = future_step
        self.end_channels = end_channels

        for i in range(node_type):
            self.end_conv_1.append(nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1,1),
                                    bias=True))
        for i in range(node_type):
            self.end_conv_2.append(nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,kernel),
                                    bias=True))
        
    # def forward(self, x,cluster_labels):
    #     '''
    #         x:  (B, N, T, D)
    #     ''' 
    #     cluster_labels = cluster_labels.cpu().numpy()
    #     num_of_clusters = np.max(cluster_labels)
    #     x = x.permute(0,3,1,2)  # (B, D, N, T)
    #     x = F.relu(x)
    #     temp= []
    #     idx_temp = []
    #     for clu_idx in range(num_of_clusters+1):
    #         clu_members = np.argwhere(cluster_labels==clu_idx).squeeze()
    #         idx_temp+=clu_members.tolist()
    #         temp.append(self.end_conv_1[clu_idx](x[:,:,clu_members,:]))
    #     x = torch.cat(temp,dim=2)
    #     _,idx = torch.sort(torch.tensor(idx_temp).to(x.device))
    #     x = x.index_select(2,idx)
        
    #     x = F.relu(x)
    #     temp= []
    #     for clu_idx in range(num_of_clusters+1):
    #         clu_members = np.argwhere(cluster_labels==clu_idx).squeeze()
    #         temp.append(self.end_conv_2[clu_idx](x[:,:,clu_members,:]))
    #     x = torch.cat(temp,dim=2)
    #     _,idx = torch.sort(torch.tensor(idx_temp).to(x.device))
    #     x = x.index_select(2,idx)
        
    #     x = x.permute(0,2,3,1) 
    #     return x
    def forward(self, x,cluster_labels):
        #   x:  (B, N, T, D)
        cluster_labels = cluster_labels.cpu().numpy()
        b,n,t,d = x.shape

        x = x.permute(0,3,1,2)  # (B, D, N, T)
        x = F.relu(x)

        temp_conv = torch.zeros((b,self.end_channels,n,t)).to(x.device)

        for clu_idx in range(self.node_type):
            temp_x = torch.zeros((b,d,n,t)).to(x.device)
            clu_members = np.argwhere(cluster_labels==clu_idx)
            temp_x[clu_members[:,0],:,clu_members[:,1],:] = x[clu_members[:,0],:,clu_members[:,1],:]
            temp_conv[clu_members[:,0],:,clu_members[:,1],:] = self.end_conv_1[clu_idx](temp_x)[clu_members[:,0],:,clu_members[:,1],:]
        x = temp_conv    

        temp_conv = torch.zeros((b,self.out_dim,n,self.future_step)).to(x.device)
        
        x = F.relu(x)
        for clu_idx in range(self.node_type):
            temp_x = torch.zeros((b,self.end_channels,n,t)).to(x.device)
            clu_members = np.argwhere(cluster_labels==clu_idx)
            temp_x[clu_members[:,0],:,clu_members[:,1],:] = x[clu_members[:,0],:,clu_members[:,1],:]
            temp_conv[clu_members[:,0],:,clu_members[:,1],:] = self.end_conv_2[clu_idx](temp_x)[clu_members[:,0],:,clu_members[:,1],:]
        x = temp_conv    

        x = x.permute(0,2,3,1) 
        return x
        
class Ada_output_layer(nn.Module):
    def __init__(self, node_type, dim_in, dim_out):
        super(Ada_output_layer, self).__init__()
        self.weights_pool1 = nn.Parameter(torch.FloatTensor(node_type,  dim_in, 32))
        self.weights_pool2 = nn.Parameter(torch.FloatTensor(node_type,  32, dim_out))
        # self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        # self.num_for_predict = num_for_predict
        self.out_channels = dim_out

    def forward(self, x , node_label):
        '''
        x:  (B, N, T, D)
        node_label:  (N, C)
        '''  
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        x = x.permute(0, 2, 1, 3)     #  (B T V F)
        x = x.reshape(-1,num_of_vertices,in_channels) #  (B*T V F)

        weights1 = torch.einsum('nc,cio->nio', node_label, self.weights_pool1)
        weights2 = torch.einsum('nc,cio->nio', node_label, self.weights_pool2)

        x = F.relu(x)     
        x = torch.einsum('bni,nio->bno', x, weights1) #+ bias     #b, N, dim_out
        x = F.relu(x)
        x = torch.einsum('bni,nio->bno', x, weights2) #+ bias     #b, N, dim_out

        x = x.reshape(batch_size, num_of_timesteps, num_of_vertices, self.out_channels).transpose(1, 2)
        return x

class DUQ_outputLayer(nn.Module):
    def __init__(self,output_dim ):
        super(DUQ_outputLayer, self).__init__()
        # 针对不同的输出维度需要改输入输出通道
        # 针对不同的预测步长需要改卷积核
        self.conv = nn.Conv2d(in_channels=1,out_channels = output_dim,
                                kernel_size=(1,6),
                                bias=True)
        self.output_dim = output_dim
    def cal_similarity(self,flow,cluster_adj):
        """
            Flow (B T N N)
        """   
        similarity = torch.matmul(flow, flow.transpose(2, 3))
        num_for_clu =  torch.sum(cluster_adj,axis=-1).unsqueeze(-1)  # (N,1)
        similarity = torch.sum(( similarity*cluster_adj ) /num_for_clu  , axis=-1)  # (B T N N) -> (B T N)

        return similarity

    def get_cluster_adj(self,type_label):
        num_of_vertices = len(type_label)
        
        type_label1 = type_label.unsqueeze(-1)
        type_label2 = type_label.unsqueeze(0)
       
        tmp1 = type_label1.expand(num_of_vertices,num_of_vertices)
        tmp2 = type_label2.expand(num_of_vertices,num_of_vertices)

        cluster_adj = torch.where(tmp1==tmp2,torch.ones_like(tmp1),torch.zeros_like(tmp1))
        
        return cluster_adj

    def forward(self,uncer_state,type_label):
        """
        Flow (B N T D)
        """
        B,N,T,D  = uncer_state.shape
        cluster_adj = self.get_cluster_adj(type_label)
        
        uncer_state =  uncer_state.permute(0, 2, 1, 3)  # (B T N D)
        uncertainty_every_node = self.cal_similarity(uncer_state , cluster_adj)  # (B T N)
        uncertainty_every_node = uncertainty_every_node.permute(0, 2, 1).unsqueeze(1)  # (B T N)->(B N T) ->(B 1 N T)
        sigma = F.relu(self.conv(uncertainty_every_node)) #->(B 1 N T)
        return sigma.permute(0, 2, 3, 1) # (B D N T) -> (B N T D) 

class Gated_Output_layer(nn.Module):
    def __init__(self, sharing_paras, output_gate, d_model, out_fea_dim, history_step, future_step ):
        super(Gated_Output_layer, self).__init__()

        self.sharing_paras = sharing_paras
        self.output_gate = output_gate

        self.pre_tcnn = nn.ModuleList()
        if not sharing_paras:
            self.uncer_tcnn = nn.ModuleList()
        
        if output_gate:
            self.gate = nn.ModuleList()

        kernel  = history_step-future_step + 1

        in_channels = [d_model,d_model//2]
        out_channels = [d_model//2,out_fea_dim]
        kernel_size = [1,kernel]
        gate_size = [(d_model,d_model//2),(2*out_fea_dim,out_fea_dim)]
        for i in range(2):
            self.pre_tcnn.append(nn.Conv2d(in_channels=in_channels[i],
                                  out_channels=out_channels[i],
                                  kernel_size=(1,kernel_size[i]),
                                  bias=True))
            if not sharing_paras:
                self.uncer_tcnn.append(nn.Conv2d(in_channels=in_channels[i],
                                    out_channels=out_channels[i],
                                    kernel_size=(1,kernel_size[i]),
                                    bias=True))
            if output_gate:
                self.gate.append(nn.Linear(gate_size[i][0],gate_size[i][1]) )
                # self.gate.append(Uncer_GRU_cell(out_channels[i]) )


    def forward(self, pre_hidden_state,uncer_state):
        '''
        pre_hidden_state:  (B, N, T, D)
        uncer_state:  (B, N, T, D)
        ''' 
        for i in range(2):
            # (B, N, T, D) ->  (B, D, N, T) 
            pre_hidden_state = pre_hidden_state.permute(0,3,1,2)
            uncer_state = uncer_state.permute(0,3,1,2)

            pre_hidden_state = F.relu(pre_hidden_state)
            uncer_state = F.relu(uncer_state)

            pre_hidden_state = self.pre_tcnn[i](pre_hidden_state)
            if self.sharing_paras:
                uncer_state = self.pre_tcnn[i](uncer_state)
            else:
                uncer_state = self.uncer_tcnn[i](uncer_state)

            # (B, D, N, T) ->  (B, N, T, D)
            pre_hidden_state = pre_hidden_state.permute(0,2,3,1)
            uncer_state = uncer_state.permute(0,2,3,1)
            if self.output_gate:
                input_and_state = torch.cat((pre_hidden_state, uncer_state), dim=-1)
                f_gate  = torch.tanh(self.gate[i](input_and_state))
                pre_hidden_state = pre_hidden_state + uncer_state*f_gate

                # pre_hidden_state , uncer_state =  self.gate[i](pre_hidden_state, uncer_state)

        return pre_hidden_state, uncer_state
