#!/usr/bin/env python
# coding: utf-8
import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_dir)
import argparse
import configparser
import shutil
import warnings
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lib.utils import (compute_val_loss, delta_loss, get_adjacency_matrix_2direction, load_data, predict_and_save_results)
from model.model_deep import make_model
warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/PEMSD7-M.conf', type=str, help="configuration file path")
parser.add_argument("--seed", type=int, default= 1)
parser.add_argument('--cuda', type=str, default= '0')
args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
torch.backends.cudnn.enabled = False
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.cuda.set_device(int(args.cuda))
else:
    args.cuda = 'cpu'
DEVICE = torch.device('cuda:{}'.format(args.cuda))

config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config), flush=True)
config.read(args.config)
#------------------------------------- Data config-------------------------------------
data_config = config['Data']
adj_filename = data_config['adj_filename']
dataset_name = data_config['dataset_name']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None
num_of_vertices = int(data_config['num_of_vertices'])
history_step = int(data_config['history_step'])
num_for_predict = int(data_config['num_for_predict'])
norm = data_config['norm']
adj_with_self_loop = bool(int(data_config['adj_with_self_loop']))
#------------------------------------- Training config-------------------------------------
training_config = config['Training']
batch_size = int(training_config['batch_size'])
start_epoch = int(training_config['start_epoch']) 
total_epochs = int(training_config['total_epochs'])
learning_rate = float(training_config['learning_rate'])


encoder_input_size = int(training_config['encoder_input_size'])
encoder_output_size = int(training_config['encoder_output_size'])
dropout = float(training_config['dropout'])
kernel_size = int(training_config['kernel_size'])
num_layers = int(training_config['num_layers'])
d_model = int(training_config['d_model'])
nb_head = int(training_config['nb_head'])
time_slices = int(training_config['time_slices'])
delta = float(training_config['delta'])
loss_ratio = float(training_config['loss_ratio'])

sharing_paras = bool(int(training_config['sharing_paras'])) 
stib = bool(int(training_config['stib'])) 
sampling = bool(int(training_config['sampling'])) 
spatial_embedding = bool(int(training_config['spatial_embedding'])) 
temporal_embedding = bool(int(training_config['temporal_embedding'])) 
mid_gate = bool(int(training_config['mid_gate'])) 
output_gate = bool(int(training_config['output_gate'])) 

#------------------------------------- Over -------------------------------------
folder_dir = '{}_{}delta_{}loss_{}lr_{}batchsize_{}layers_{}dmodel_{}dropout_{}{}{}{}{}{}{}'.format(args.seed,delta,loss_ratio,
    learning_rate,batch_size,num_layers,d_model,dropout,int(sharing_paras),int(stib),int(sampling),int(spatial_embedding),int(temporal_embedding),int(mid_gate),int(output_gate))
print('folder_dir:', folder_dir)
params_path = os.path.join('experiments', dataset_name, folder_dir)

adj_mx, distance_mx = get_adjacency_matrix_2direction(dataset_name, adj_filename, num_of_vertices, id_filename)

train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _max, _min = load_data(
    dataset_name, encoder_input_size,  norm, DEVICE, batch_size,shuffle=True)

net  = make_model(DEVICE, adj_mx, adj_with_self_loop,num_of_vertices,history_step, num_for_predict,
        sharing_paras,stib,sampling,spatial_embedding,temporal_embedding,mid_gate,output_gate,
         encoder_input_size, encoder_output_size, d_model, nb_head, time_slices,
        num_layers, dropout=dropout, kernel_size=kernel_size)

def train_main(log_writer):
    if (start_epoch == 0) and (not os.path.exists(params_path)): 
        os.makedirs(params_path)
        print('create params directory %s' % (params_path), flush=True)
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path), flush=True)
    elif (start_epoch > 0) and (os.path.exists(params_path)): 
        print('train from params directory %s' % (params_path), flush=True)
    else:
        raise SystemExit('Wrong type of model!')

   
    criterion = delta_loss(delta, loss_ratio) 
    
    optimizer = optim.Adam(net.parameters(), lr=learning_rate) 
    
    log_filename = os.path.join(params_path, 'log.csv')
    log_writer.open(log_filename, mode="a")

    total_param = 0
   
    for param_tensor in net.state_dict():
        total_param += np.prod(net.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param, flush=True)

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    if start_epoch > 0:

        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)
        net.load_state_dict(torch.load(params_filename))
        print('start epoch:', start_epoch, flush=True)
        print('load weight from: ', params_filename, flush=True)

    start_time = time()
    message =  f"{'-' * 5} | {'-' * 7}{'Training'}{'-' * 7} | {'-' * 7}{'Validation'}{'-' * 7} | {'-' * 7}{'Testing'}{'-' * 7} |"
    print(  message )
    log_writer.write(message)
    message = f"{'Epoch':^5} | {'pre_loss':^9}{'Time-train':^7} | {'RMSE':^6}{'MAPE':^6}{'Time-val':^8} | {'RMSE':^6}{'MAPE':^6}{'PICP':^6}{'intervals':^6}{'mis':^6}{'Time-test':^8} |"
    print(  message  )
    log_writer.write(message)

    not_improved_count = 0

    for epoch in range(start_epoch, total_epochs):
        net.train()  
        train_start_time = time()
        total_pre_loss = 0
        
        total_mae_loss = 0
        total_bound_loss = 0
        total_time = 0
        
        for batch_index, batch_data in enumerate(train_loader):
            
            encoder_inputs, labels, time_stamps = batch_data
            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
            B, N, T, D = encoder_inputs.shape  # (B, N, T, F)
            labels = labels.transpose(-1, -2) # (B, N, T, f)
            
            batch_start_time = time()

            outputs,sigma = net(encoder_inputs,time_stamps)
           
            loss0, loss1, loss2 = criterion(outputs, labels, sigma)
            total_pre_loss += loss2.item()
            total_mae_loss += loss0.item()
            total_bound_loss +=  loss1.item()
            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()
            
            total_time += time()-batch_start_time

            global_step += 1
            
        total_pre_loss = total_pre_loss/len(train_loader)
        total_mae_loss = total_mae_loss/len(train_loader)
        total_bound_loss = total_bound_loss/len(train_loader)
        total_time = total_time/len(train_loader)
        
        message = f"{epoch:<5} | {total_pre_loss:<7.2f}{total_mae_loss:<7.2f}{total_bound_loss:<7.2f}{total_time:<7.2f}s "
        
        print('\r' + message, end='', flush=False)
        log_writer.write('Train:'+ message)

        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
      
        val_loss = compute_val_loss(net, log_writer, val_loader, criterion, val_target_tensor,_max,_min,norm)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
          
            print(f" ", end='', flush=True)
            predict_main(best_epoch, log_writer, test_loader, test_target_tensor, _max, _min, norm )
            not_improved_count = 0
        else:
            not_improved_count += 1
        
        print()
        log_writer.write('-' * 20)
        if not_improved_count == 20:
            print("early stop")
            break
        
    print('apply the best val model on the test data set ...', flush=True)

    predict_main(best_epoch, log_writer, test_loader, test_target_tensor, _max, _min,  norm,  save = True)

def predict_main(epoch, log_writer, data_loader, data_target_tensor, _max, _min, norm,  save =False ):
    '''
    在测试集上，测试指定epoch的效果
    :param epoch: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param _max: (1, 1, 3, 1)
    :param _min: (1, 1, 3, 1)
    :param type: string
    :return:
    '''

    params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

  
    net.load_state_dict(torch.load(params_filename))

    input, prediction, data_target_tensor, sigma = predict_and_save_results(net, log_writer, data_loader, data_target_tensor,  _max, _min, norm, save)
    
    if save :
 
        save_path = os.path.join('experiments', dataset_name, folder_dir)
     
        np.savez_compressed(
            save_path ,
            input = input,
            prediction=prediction,
            target=data_target_tensor,
            sigma=sigma,
            )
        print('\n'+save_path)
  
def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=True, is_file=True,flush = True):

        if is_file:
            self.file.write(message+'\n')
            if flush:
                self.file.flush()



if __name__ == "__main__":
    
    log_writer = Logger()
    
    setup_seed(args.seed)
    train_main(log_writer)















