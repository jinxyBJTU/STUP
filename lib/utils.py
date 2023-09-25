import os
from time import time
import numpy as np
import torch
import torch.nn.functional as F 
import torch.utils.data
from scipy.sparse.linalg import eigs
from sklearn.cluster import SpectralClustering

from .metrics import masked_mae_np, masked_mape_np, masked_mse_np, picp_np

def delta_loss(delta, loss_ratio):
    def loss(y_pred, y_true,sigma):
        loss0 = torch.abs(y_pred - y_true).mean()
        loss1 = torch.abs( delta * torch.abs(y_pred - y_true) - sigma ).mean()
        loss2 = nle_loss(y_pred, y_true, sigma)
        
        total_loss = loss_ratio*(loss0 + loss1) + loss2
        return loss_ratio*loss0, loss_ratio*loss1, total_loss
    return loss
    
def maemis_loss(y_pred, y_true,sigma):
    """
    y_pred: T B V F
    y_true: T B V 
    """
    mask = (y_true != 0).float()
    mask /= mask.mean()
    pho = 0.05
    factor = 1.
    loss0 = torch.abs(y_pred - y_true) 
    loss1 = torch.max(2*factor*sigma, torch.zeros_like(y_pred).to(y_true.device))

    loss2 = torch.max(y_true - (y_pred + factor*sigma) ,    torch.zeros_like(y_pred).to(y_true.device))*2/pho
    loss3 = torch.max((y_pred-factor*sigma) - y_true ,     torch.zeros_like(y_pred).to(y_true.device))*2/pho
    loss = loss0 + loss1 + loss2 + loss3
  
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean(), [loss0 , loss1 , loss2 ,loss3]

def nle_loss(y_pred, y_true, sigma):
    # B, N, T, D = y_true.shape
    loss =  torch.log(sigma**2+1)/2 + torch.div((y_true-y_pred)**2,2*sigma**2)
    return torch.mean(loss)

def get_psedo_label_from_spatial_neibors(inputs,y_pred,W_loop):
    B, N, T, D = y_pred.shape
    var_spatial = torch.zeros([B,N,T,D]).to(y_pred.device)
    # print(W_loop)
    for node in range(N):
        neigh = np.argwhere(W_loop[node]>0).squeeze()    
        neigh_flow = y_pred[:,neigh,:,:]
        var_spatial[:,node,:,:] = torch.std(neigh_flow,axis=1)
        
    var_ep = torch.std(inputs/inputs.shape[2],axis=2).unsqueeze(2).expand(B, N, T, D)
    return var_ep

def get_psedo_label_from_clu_members(inputs,y_pred,cluster_labels):
    B, N, T, D = y_pred.shape
    cluster_labels = cluster_labels.cpu().numpy()
    var_spatial = torch.zeros([B,N,T,D]).to(y_pred.device)
    # print(W_loop)
    num_of_clusters = np.max(cluster_labels)
    
    for clu_idx in range(num_of_clusters+1):
        clu_members = np.argwhere(cluster_labels==clu_idx).squeeze()
        members_flow = y_pred[:,clu_members,:,:]
        var_spatial[:,clu_members,:,:] = torch.std(members_flow,axis=1).unsqueeze(1)
        
    var_ep = torch.std(inputs/inputs.shape[2],axis=2).unsqueeze(2).expand(B, N, T, D)
    return var_spatial 

def target_distribution(q):
    """
    q:(B,N,C)
    """
    weight = q**2 / torch.sum(q,1,keepdims=True)
    return (weight / torch.sum(weight, -1,keepdims=True))

def re_normalization(x, mean, std):
    x = x * std + mean
    return x

def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x

def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x

def perform_SpectralClustering(n_clusters,adj):
    
    y_pred = SpectralClustering(n_clusters=n_clusters,affinity='precomputed').fit_predict(adj)
    
    return y_pred
    
def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))} 

            with open(distance_df_filename, 'r') as f:
                f.readline() 
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:  

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA

def get_adjacency_matrix_2direction(dataset_name, adj_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''

    adj_file = os.path.join(adj_filename)
    if 'npy' in adj_file:
        adj_mx = np.load(adj_file)
        
        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)
        
        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))} 

            with open(adj_filename, 'r') as f:
                f.readline()  
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    A[id_dict[j], id_dict[i]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
                    distaneA[id_dict[j], id_dict[i]] = distance
            return A, distaneA

        else:  

            with open(adj_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    A[j, i] = 1
                    distaneA[i, j] = distance
                    distaneA[j, i] = distance
            
            return A, distaneA

def get_Laplacian(A):
    '''
    compute the graph Laplacian, which can be represented as L = D − A

    Parameters
    ----------
    A: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    Laplacian matrix: np.ndarray, shape (N, N)

    '''

    assert (A-A.transpose()).sum() == 0 

    D = np.diag(np.sum(A, axis=1))  

    L = D - A  

    return L

def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1)) 

    L = D - W 

    lambda_max = eigs(L, k=1, which='LR')[0].real 

    return (2 * L) / lambda_max - np.identity(W.shape[0])

def sym_norm_Adj(W, adj_with_self_loop):
    '''
    compute Symmetric normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    Symmetric normalized Laplacian: (D^hat)^1/2 A^hat (D^hat)^1/2; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    if adj_with_self_loop:
        W_loop = W
        W_wto_loop = W - np.identity(N)
    else:
        W_loop = W + np.identity(N)
        W_wto_loop = W

    D_loop = np.diag(np.power(np.sum(W_loop, axis=1),-0.5))
    D_wto_loop = np.diag(np.power(np.sum(W_wto_loop, axis=1),-0.5))

    sym_norm_adj_loop = np.dot(np.dot(D_loop,W_loop),D_loop)
    sym_norm_adj_wto_loop = np.dot(np.dot(D_wto_loop,W_wto_loop),D_wto_loop)

    return sym_norm_adj_loop , sym_norm_adj_wto_loop ,  W_loop , W_wto_loop

def norm_Adj(W):
    '''
    compute  normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    normalized Adj matrix: (D^hat)^{-1} A^hat; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    W = W + np.identity(N)  
    D = np.diag(1.0/np.sum(W, axis=1))
    norm_Adj_matrix = np.dot(D, W)

    return norm_Adj_matrix

def compute_val_loss(net, log_writer, val_loader, criterion,  val_target_tensor, max, min, norm):
    '''
    compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param epoch: int, current epoch
    :return: val_loss
    '''

    net.train(False) 
    _max = max.transpose((0,1,3,2))
    _min = min.transpose((0,1,3,2))
    with torch.no_grad():
        val_target_tensor = val_target_tensor.transpose(-1, -2) .cpu().numpy()

        tmp = []  
        prediction = []
        start_time = time()

        for batch_index, batch_data in enumerate(val_loader):

            encoder_inputs, labels, time_stamps = batch_data
            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
            labels = labels.transpose(-1, -2)   # (B，N，T，F)

            predict_output,sigma = net(encoder_inputs,time_stamps)

            loss0,loss1,loss2 = criterion(predict_output, labels, sigma)

            tmp.append( loss2.item() )
            prediction.append(predict_output.detach().cpu().numpy())
        
        prediction = np.concatenate(prediction, 0)  # (batch, N, T', 1)

        if norm == 'MinMax':
            prediction = re_max_min_normalization(prediction, _max, _min)# B N T F
            val_target_tensor = re_max_min_normalization(val_target_tensor, _max, _min)

        mae = masked_mae_np(val_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        rmse = masked_mse_np(val_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0) ** 0.5
        mape = masked_mape_np(val_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        
        message = f"  | {mae:<7.2f}{rmse:<7.2f}{mape:<7.2f}{time() - start_time:<6.2f}s"
        print(message, end='', flush=False)
        log_writer.write('Val:'+message)

        validation_loss = sum(tmp) / len(tmp)

    return validation_loss

def predict_and_save_results(net, log_writer, data_loader, data_target_tensor, max, min, norm,save='False'):
    '''
    for transformerGCN
    :param net: nn.Module
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param epoch: int
    :param _max: (1, 1, 3, 1)
    :param _min: (1, 1, 3, 1)
    :return:
    '''
    net.train(False)  # ensure dropout layers are in test mode
    _max = max.transpose((0,1,3,2))
    _min = min.transpose((0,1,3,2))
    start_time = time()

    with torch.no_grad():

        data_target_tensor = data_target_tensor.transpose(-1, -2) .cpu().numpy() 

        loader_length = len(data_loader)  # nb of batch

        prediction = []
        sigma = []
        input = []  

        start_time = time()
        total_time = 0

        for batch_index, batch_data in enumerate(data_loader):
            encoder_inputs, labels,time_stamps = batch_data
            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
            labels = labels.transpose(-1, -2)  # (B, N, T, 1)
            
            batch_start_time = time()
            predict_output, sigma_output = net(encoder_inputs, time_stamps)
            total_time += time()-batch_start_time
            
            input.append(encoder_inputs.detach().cpu().numpy())     # (batch, T', 1)
            prediction.append(predict_output.detach().cpu().numpy())
            sigma.append(sigma_output.detach().cpu().numpy())

        test_cost_time = time() - start_time
     
        input = np.concatenate(input, 0)  # (batch, N, T', 1)
        prediction = np.concatenate(prediction, 0)  # (batch, N, T', 1)
        sigma = np.concatenate(sigma, 0)  # (batch, N, T', 1)
           
        if norm == 'MinMax':
            input = re_max_min_normalization(input, _max, _min)
            prediction = re_max_min_normalization(prediction, _max, _min)
            data_target_tensor = re_max_min_normalization(data_target_tensor, _max, _min)

        mae = masked_mae_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        rmse = masked_mse_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0) ** 0.5
        mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        
        picp, intervals,mis = picp_np(data_target_tensor, prediction, sigma)
        total_time = total_time/len(data_loader)

        message = f" | {mae:<7.2f}{rmse:<7.2f}{mape:<7.2f}{picp:<7.2f}{intervals:<7.2f}{mis:<7.2f}{total_time:<6.2f}s"

        print(message, end='', flush=False)
        log_writer.write('Test:'+message)

        return input,prediction, data_target_tensor,sigma
    
def load_data(dataset_name, encoder_input_size, norm,
                                DEVICE, batch_size, shuffle=True, percent=1.0):
    '''
    将x,y都处理成归一化到[-1,1]之前的数据;
    每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
    该函数会把hour, day, week的时间串起来；
    注： 从文件读入的数据，x,y都是归一化后的值
    :param graph_signal_matrix_filename: str
    :param num_of_hours: intw
    :param num_of_weeks: int
    :param DEVICE:
    :param batch_size: int
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)
    '''

   
    filename = os.path.join('data/',dataset_name, dataset_name +'.npz') 
    file_data = np.load(filename)
  
    train_x = file_data['train_x'][:, :, 0:encoder_input_size, :]  # (10181, 307, 3, 12)
    val_x = file_data['val_x'][:, :, 0:encoder_input_size, :]
    test_x = file_data['test_x'][:, :, 0:encoder_input_size, :]
    
    train_target = file_data['train_target'][:, :, 0:encoder_input_size, :]  # (10181, 307, F,12)
    val_target = file_data['val_target'][:, :, 0:encoder_input_size, :]
    test_target = file_data['test_target'][:, :, 0:encoder_input_size, :]

    train_timestamp = file_data['train_timestamp']  # (10181, 1)
    val_timestamp = file_data['val_timestamp']
    test_timestamp = file_data['test_timestamp']

    train_x_length = train_x.shape[0]
    scale = int(train_x_length*percent)
    train_x = train_x[:scale]
    train_target = train_target[:scale]
    train_timestamp = train_timestamp[:scale]

    _max = file_data['mean'][:, :, 0:encoder_input_size, :]  # (1, 1, 3, 1)
    _min = file_data['std'][:, :, 0:encoder_input_size, :]  # (1, 1, 3, 1)
    
    train_x = max_min_normalization(train_x, _max, _min)
    val_x = max_min_normalization(val_x, _max, _min)
    test_x = max_min_normalization(test_x, _max, _min)
    train_target_norm = max_min_normalization(train_target, _max, _min)
    val_target_norm = max_min_normalization(val_target, _max, _min)
    test_target_norm = max_min_normalization(test_target, _max, _min)

    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
   
    train_target_tensor = torch.from_numpy(train_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    train_timestamp = torch.from_numpy(train_timestamp).type(torch.LongTensor).to(DEVICE)  # (B, N, T)
   
    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor,train_timestamp)
      
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    #  ------- val_loader -------
    val_decoder_input_start = val_x[:, :,:, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    val_decoder_input = np.concatenate((val_decoder_input_start, val_target_norm[:, :, :,:-1]), axis=3)  # (B, N, T)
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
   
    val_target_tensor = torch.from_numpy(val_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    val_timestamp = torch.from_numpy(val_timestamp).type(torch.LongTensor).to(DEVICE)  # (B, N, T)

  
    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor, val_timestamp)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    #  ------- test_loader -------
    test_decoder_input_start = test_x[:, :, :, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
   
    test_decoder_input = np.concatenate((test_decoder_input_start, test_target_norm[:, :, :,:-1]), axis=3)  # (B, N, T)

    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
 
    test_target_tensor = torch.from_numpy(test_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    test_timestamp = torch.from_numpy(test_timestamp).type(torch.LongTensor).to(DEVICE)  # (B, N, T)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor,test_timestamp)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # print
    print('train:', train_x_tensor.size(),  train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_target_tensor.size())
    
    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _max, _min


