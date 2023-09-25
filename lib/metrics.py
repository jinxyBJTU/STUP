# -*- coding:utf-8 -*-

import numpy as np
import torch

def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

def masked_mse_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):  
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= mask.mean()
        mse = (y_true - y_pred) ** 2
        return np.mean(np.nan_to_num(mask * mse))

def masked_mae_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):  
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= mask.mean()
        mae = np.abs(y_true - y_pred)
        return np.mean(np.nan_to_num(mask * mae))
        
def picp_np(y_true, y_pred,sigma):
    """
    y_true,y_pred : B, N, T, D
    sigma : B, N, T, D
    """
    B, N, T, D = y_true.shape 

    factor = 1.65 #  1.65-90%
    pho = 0.05
    upper_bound = y_pred + factor*sigma
    lower_bound = y_pred - factor*sigma
                
    result1 = np.where(y_true<upper_bound ,np.ones_like(y_true),np.zeros_like(y_pred))
    result2 = np.where(y_true>lower_bound ,np.ones_like(y_true),np.zeros_like(y_pred))
    
    recalibrate_rate = np.sum(result1*result2)/np.prod(y_true.shape)
    
    mask = np.not_equal(y_true, 0).astype('float32')
    mask /= mask.mean()

    intervals = np.maximum(2*factor*sigma, 0)
    loss2 = np.maximum(y_true - (y_pred + factor*sigma) ,  0)*2/pho
    loss3 = np.maximum((y_pred-factor*sigma) - y_true ,    0)*2/pho
    mis =  intervals + loss2 + loss3
    mis = mis * mask
    mis[mis != mis] = 0
    mis = mis.mean()
   

    return recalibrate_rate * 100, intervals.mean(), mis