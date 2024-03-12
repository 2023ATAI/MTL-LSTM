import argparse
import pickle
from pathlib import PosixPath, Path
import time
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from train import train
from eval import test
from eval_mtl import test_mtl,test_mtls
from data import Dataset
from config import get_args
from utils import GetKGE,GetNSE,GetPCC,_plotloss,_plotbox,_boxkge,_boxnse,_boxpcc
from loss import NaNMSELoss
from utils import _plotloss
import torch
# ------------------------------------------------------------------------------ 
# Original author : Qingliang Li, Cheng Zhang, 12/23/2022
# Edited by Jinlong Zhu, Gan Li	
# Inspired by Lu Li
# ------------------------------------------------------------------------------

def main(cfg):
    device = torch.device(cfg['device']) if torch.cuda.is_available() else torch.device('cpu')
    print('Now we training {d_p} product in {sr} spatial resolution'.format(d_p=cfg['product'],sr=str(cfg['spatial_resolution'])))
    print('1 step:-----------------------------------------------------------------------------------------------------------------')
    print('[ATAI {d_p} work ] Make & load inputs'.format(d_p=cfg['workname']))
    path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/'
    if not os.path.isdir (path):
        os.makedirs(path)
    #To determine whether the raw data in LandBench has been processed, it is necessary to convert it according to the requirements of the designed model.
    if os.path.exists(path+'/x_train_norm.npy'):
        print(' [ATAI {d_p} work ] loading input data'.format(d_p=cfg['workname']))
        x_train_shape = np.load(path+'x_train_norm_shape.npy',mmap_mode='r')
        x_train = np.memmap(path+'x_train_norm.npy',dtype=cfg['data_type'],mode='r+',shape=(x_train_shape[0],x_train_shape[1], x_train_shape[2], x_train_shape[3]))
        x_test_shape = np.load(path+'x_test_norm_shape.npy',mmap_mode='r')
        x_test = np.memmap(path+'x_test_norm.npy',dtype=cfg['data_type'],mode='r+',shape=(x_test_shape[0],x_test_shape[1], x_test_shape[2], x_test_shape[3]))
        y_train = np.load(path+'y_train_norm.npy',mmap_mode='r')
        y_test = np.load(path+'y_test_norm.npy',mmap_mode='r')

        with open(path + 'static_norm.npy', 'rb') as f:
            static = pickle.load(f)

        file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])

        mask = np.load(path+file_name_mask)
        

    else:     
        print('[ATAI {d_p} work ] making input data'.format(d_p=cfg['workname']))
        cls = Dataset(cfg) #FIXME: saving to input path
        x_train, y_train, x_test, y_test, static, lat, lon,mask = cls.fit(cfg)
    # load scaler for inverse
    if cfg['normalize_type'] in ['region']:                                      
        scaler_x = np.memmap(path+'scaler_x.npy',dtype=cfg['data_type'],mode='r+',shape=(2, x_train.shape[1], x_train.shape[2], x_train.shape[3]))
        scaler_y = np.memmap(path+'scaler_y.npy',dtype=cfg['data_type'],mode='r+',shape=(2, y_train.shape[1], y_train.shape[2], y_train.shape[3]))  
    elif cfg['normalize_type'] in ['global']:    
        scaler_x = np.memmap(path+'scaler_x.npy',dtype=cfg['data_type'],mode='r+',shape=(2, x_train.shape[3]))
        scaler_y = np.memmap(path+'scaler_y.npy',dtype=cfg['data_type'],mode='r+',shape=(2, y_train.shape[3]))  
    # ------------------------------------------------------------------------------------------------------------------------------
    print('2 step:-----------------------------------------------------------------------------------------------------------------')
    print('[ATAI {d_p} work ] Train & load {m_n} Model'.format(d_p=cfg['workname'],m_n=cfg['modelname']))
    print('[ATAI {d_p} work ] Wandb info'.format(d_p=cfg['workname']))
# ------------------------------------------------------------------------------------------------------------------------------
    #Model training
    out_path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' + cfg['workname'] + '/' + cfg['modelname'] +'/focast_time '+ str(cfg['forcast_time']) +'/'
    if not os.path.isdir (out_path):
        os.makedirs(out_path)   
    if os.path.exists(out_path+cfg['modelname'] +'_para.pkl'):
        if cfg["modelname"] in ['SoftMTLv1']:
            model1 = torch.load(out_path + cfg['modelname'] + '_para.pkl')
            model2 = torch.load(out_path + 'SoftMTLv2' + '_para.pkl')
            print('[ATAI {d_p} work ] loading trained model'.format(d_p=cfg['workname']))

        else:
            model = torch.load(out_path+cfg['modelname']+'_para.pkl')
            print('[ATAI {d_p} work ] loading trained model'.format(d_p=cfg['workname']))
    else:
        # train 
        print('[ATAI {d_p} work ] training {m_n} model'.format(d_p=cfg['workname'],m_n=cfg['modelname']))

        train(x_train, y_train, static, mask, scaler_x, scaler_y, cfg, 0,path,out_path,device)
        if cfg["modelname"] in ['SoftMTLv1']:
            model1 = torch.load(out_path + cfg['modelname'] + '_para.pkl')
            model2 = torch.load(out_path + 'SoftMTLv2' + '_para.pkl')
        else:
            model = torch.load(out_path+cfg['modelname']+'_para.pkl')
        print('[ATAI {d_p} work ] finish training {m_n} model'.format(d_p=cfg['workname'],m_n=cfg['modelname']))   
    # ------------------------------------------------------------------------------------------------------------------------------
    print('3 step:-----------------------------------------------------------------------------------------------------------------')  
    print('[ATAI {d_p} work ] Make predictions by {m_n} Model'.format(d_p=cfg['workname'],m_n=cfg['modelname']))  
# ------------------------------------------------------------------------------------------------------------------------------
    print('x_test shape :',x_test.shape)
    print('y_test shape :',y_test.shape)
    print('static shape :',static[0].shape)
    print('scaler_x shape is',scaler_x.shape)
    print('scaler_y shape is',scaler_y.shape)
    #Model testing
# ------------------------------------------------------------------------------------------------------------------------------   
# save predicted values and true values
#     print('[ATAI {d_p} work ] Saving predictions by {m_n} Model and we hope to use "postprocess" and "plot_test" codes for detailed analyzing'.format(d_p=cfg['workname'],m_n=cfg['modelname']))
#     np.save(out_path +'_predictions.npy', y_pred)
#     np.save(out_path + 'observations.npy', y_test)
    if cfg["modelname"] in ['MHLSTMModel','MSLSTMModel','MMOE']:
        print("运行成功了",model)
        # print("y_test ",y_test)
        y_preds,y_tests = test_mtl(x_test, y_test, static, scaler_y, cfg, model, device)
        # y_tests[0] = np.array(y_tests)[0]
        # nt, nlat, nlon = y_tests[0].shape
        # lossmse = torch.nn.MSELoss()
        # kge = []
        # nse = []
        # pcc = []
        # loss_test = []
        # for iii in range(cfg['num_repeat']):
        #     loss_time = []
        #     kge_time = []
        #     pcc_time = []
        #     nse_time = []
        #     for q in range(nlat):
        #         for w in range(nlon):
        #             if not np.isnan(y_preds[iii][:,q,w]).any():
        #                 loss_time.append(NaNMSELoss.fit(cfg, y_preds[iii][:,q,w].squeeze(), y_tests[iii][:,q,w].squeeze(), lossmse))
        #                 kge_time.append(GetKGE(y_preds[iii][:,q,w].squeeze(), y_tests[iii][:,q,w].squeeze()))
        #                 pcc_time.append(GetPCC(y_preds[iii][:,q,w].squeeze(), y_tests[iii][:,q,w].squeeze()))
        #                 nse_time.append(GetNSE(y_preds[iii][:,q,w].squeeze(), y_tests[iii][:,q,w].squeeze()))
        #     kge.append(kge_time)
        #     nse.append(pcc_time)
        #     pcc.append(nse_time)
        #     loss_test.append(loss_time)
        #
        # _plotbox(cfg, loss_test)
        # _boxkge(cfg, kge)
        # _boxpcc(cfg, pcc)
        # _boxnse(cfg, nse)
        print(
            '[ATAI {d_p} work ] Saving predictions by {m_n} Model and we hope to use "postprocess" and "evaluate" codes for detailed analyzing'.format(
                d_p=cfg['workname'], m_n=cfg['modelname']))
        with open(out_path + '_predictions_s.npy', 'wb') as f:
            pickle.dump(y_preds, f)
        with open(out_path + 'observations_s.npy', 'wb') as f:
            pickle.dump(y_tests, f)
        # np.save(out_path + '_predictions_s.npy', y_preds)
        # np.save(out_path + 'observations_s.npy', y_tests)

    elif cfg["modelname"] in ['SoftMTLv1']:
        print("运行SOFT成功了")
        # print("y_test ",y_test)
        y_pred1, y_test1, y_pred2, y_test2, = test_mtls(x_test, y_test, static, scaler_y, cfg, model1, model2, device)
        print(
            '[ATAI {d_p} work ] Saving predictions by {m_n} Model and we hope to use "postprocess" and "evaluate" codes for detailed analyzing'.format(
                d_p=cfg['workname'], m_n=cfg['modelname']))
        np.save(out_path + '_predictions_s.npy', y_pred1)
        np.save(out_path + 'observations_s.npy', y_test1)
        np.save(out_path + '_predictions_r.npy', y_pred2)
        np.save(out_path + 'observations_r.npy', y_test2)
    else:
        print("没有运行成功啊！！！！！！！！！！")
        y_pred, y_test = test(x_test, y_test, static, scaler_y, cfg, model, device)
        # ------------------------------------------------------------------------------------------------------------------------------
    # 将测试集跑出来的结果保存
        print('[ATAI {d_p} work ] Saving predictions by {m_n} Model and we hope to use "postprocess" and "evaluate" codes for detailed analyzing'.format(d_p=cfg['workname'],m_n=cfg['modelname']))
        with open(out_path + '_predictions.npy', 'wb') as f:
            pickle.dump(y_pred, f)
        with open(out_path + 'observations.npy', 'wb') as f:
            pickle.dump(y_test, f)

    out_path_mhl = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg['spatial_resolution']) + '/' + cfg[
        'workname'] + '/' + cfg['modelname'] + '/focast_time ' + str(cfg['forcast_time']) + '/'

    # nt, nlat, nlon = y_tests[0].shape
    # r_mhls = np.full((nlat, nlon), np.nan)
    # r = []
    # #
    # #
    # for ntask in range(cfg['num_repeat']):
    #     for i in range(nlat):
    #         for j in range(nlon):
    #             if not (np.isnan(y_tests[ntask][:, i, j]).any()):
    #                 # print(' y_pred_mhls[:, i, j] is', y_pred_mhls[:, i, j])
    #                 # print(' y_pred_mhlv[:, i, j] is', y_pred_mhlv[:, i, j])
    #                 # print(' y_test_mhl[:, i, j] is', y_test_mhl[:, i, j])
    #                 # urmse_mhls[i, j] = _mse(y_test_mhls[ntask,:, i, j], y_pred_mhls[ntask,:, i, j])
    #                 # r2_mhls[i, j] = r2_score(y_test_mhls[:, i, j], y_pred_mhls[:, i, j])
    #                 r_mhls[i, j] = np.corrcoef(y_tests[ntask][:, i, j], y_preds[ntask][:, i, j])[0, 1]
    #     r.append(r_mhls)
    # np.save(out_path_mhl + 'r_' + cfg['modelname'] + '.npy', r)
    # for i in range(cfg['num_repeat']):
    #     print('the average r of', cfg['label'][i], 'model is :', np.nanmedian(r[i]))


if __name__ == '__main__':
    cfg = get_args()
    main(cfg)
