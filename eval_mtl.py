import numpy as np
import torch
from utils import r2_score
import time
from data_gen import erath_data_transform
import sys
from data import Dataset

def batcher_lstm(x_test, y_test, aux_test, seq_len,forcast_time):
    n_t, n_feat = x_test.shape
    n = (n_t-seq_len-forcast_time)
    x_new = np.zeros((n, seq_len, n_feat))*np.nan
    # print('x_new',x_new.shape)
    y_new = np.zeros((n,2))*np.nan
    # print('y_new', y_new.shape)
    aux_new = np.zeros((n,aux_test.shape[0]))*np.nan
   #x_new每次加载7天 y_new加载8天 多处一天是每个7天进行一次预测？
    for i in range(n):
        x_new[i] = x_test[i:i+seq_len]
        y_new[i] = y_test[i+seq_len+forcast_time]
        aux_new[i] = aux_test
    # print("x_new  is", x_new)
    return x_new, y_new, aux_new

def batcher_mtl(x_test, y_test, aux_test, seq_len,forcast_time):
    a = torch.from_numpy(x_test)
    # if torch.isnan(a).any().item():
    #     print("no")
    # else:
    #     print("yes")
    #     print("yes")
    #     print("yes")
    n_t, n_feat = x_test.shape
    _, n_out = y_test.shape
    n = (n_t-seq_len-forcast_time)
    x_new = np.zeros((n, seq_len, n_feat))*np.nan
    # print('x_new',x_new.shape)
    y_new = np.zeros((n,n_out))*np.nan
    # print('y_new', y_new.shape)
    aux_new = np.zeros((n,aux_test.shape[0]))*np.nan
   #x_new每次加载7天 y_new加载8天 多处一天是每个7天进行一次预测？
    for i in range(n):
        x_new[i] = x_test[i:i+seq_len]
        #shape 是 365，2
        y_new[i] = y_test[i+seq_len+forcast_time]

        aux_new[i] = aux_test
    # print("x_new  is", x_new)
    # shape 是 365，1
    return x_new, y_new,  aux_new

def test_mtls(x, y, static, scaler, cfg, model1,model2,device):
    cls = Dataset(cfg)
    model1.eval()
    model2.eval()
    # get mean, std (1, ngrid, 6)
    #mean, std = np.array(scaler[0]), np.array(scaler[1])
    #mean = torch.from_numpy(mean).to(device)
    #std = torch.from_numpy(std).to(device)
    # if y.shape[0]-cfg["seq_len"]-cfg["forcast_time"]!=365:
    #     seq_len = 7
    #     x = x[5-1:]
    #     y_pred_ens = np.zeros((y.shape[0]-seq_len-cfg['forcast_time']+1-5, y.shape[1], y.shape[2]))*np.nan
    #     y_pred_ens1 = np.zeros((y.shape[0]-seq_len-cfg['forcast_time']+1-5, y.shape[1], y.shape[2]))*np.nan
    #     y_pred_ens2 = np.zeros((y.shape[0]-seq_len-cfg['forcast_time']+1-5, y.shape[1], y.shape[2]))*np.nan
    #     y_true = y[seq_len+cfg['forcast_time']+5-1:,:,:,0]
    # else:

    y_pred_ens = np.zeros((y.shape[0]-cfg["seq_len"]-cfg['forcast_time'], y.shape[1], y.shape[2]))*np.nan
    y_pred_ens1 = np.zeros((y.shape[0]-cfg["seq_len"]-cfg['forcast_time'], y.shape[1], y.shape[2]))*np.nan
    y_pred_ens2 = np.zeros((y.shape[0]-cfg["seq_len"]-cfg['forcast_time'], y.shape[1], y.shape[2]))*np.nan
    # y2 = np.divide(y2,1000)
    # y_true = y[cfg["seq_len"]+cfg['forcast_time']:,:,:,0]
    y_true1 = y[cfg["seq_len"]+cfg['forcast_time']:,:,:,0]
    y_true2 = y[cfg["seq_len"]+cfg['forcast_time']:,:,:,1]
    print('x shape is',x.shape)
    print('y_true shape is',y.shape)
    print('the true label shape is: {ts} and the predicton shape is: {ps}'.format(ts=y_true1.shape, ps=y_pred_ens1.shape))
    print('the true label shape is: {ts} and the predicton shape is: {ps}'.format(ts=y_true2.shape, ps=y_pred_ens2.shape))
    mask1 = y_true1 == y_true1
    mask2 = y_true2 == y_true2
    t_begin = time.time()
    # ------------------------------------------------------------------------------------------------------------------------------
    # for each grid by MTL model
    if cfg["modelname"] in ['SoftMTLv1']:
        count = 1
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                x_new, y_new, static_new = batcher_mtl(x[:, i, j, :], y[:, i, j, :], static[i, j, :], cfg["seq_len"],
                                                        cfg['forcast_time'])
                # if np.sum(x_new)!=0:
                # print('x_new are',x_new)
                xx = []
                # print("y_new.size(1)",len(y_new[1]))
                for i in range(len(y_new[1])):
                    xt = torch.from_numpy(x_new).to(device)
                    s = torch.from_numpy(static_new).to(device)
                    s = s.unsqueeze(1)
                    s = s.repeat(1, xt.shape[1], 1)
                    xt = torch.cat([xt, s], 2)
                    xx.append(xt)
                pred1 = model1(xx[0], static_new)
                pred2 = model2(xx[1], static_new)
                pred = [pred1,pred2]
                # pred = pred*std[i, j, :]+mean[i, j, :] #(nsample,1,1)

                # print('pred1  is', pred.shape)
                # print('pred2  is', pred2)
                #經過 反正則化出來的結果就是nan
                pp = []
                for ii in range(len(pred)):
                    p = pred[ii].cpu().detach().numpy()
                    p = np.squeeze(p)
                    if cfg["normalize"] and cfg['normalize_type'] in ['region']:
                        p = cls.reverse_normalize(p, 'output', scaler[:, i, j, 0], 'minmax', -1)
                    elif cfg["normalize"] and cfg['normalize_type'] in ['global']:
                        p = cls.reverse_normalize(p, 'output', scaler, 'minmax', -1)
                    pp.append(p)
                # print('pre reverse is',pred.shape)
                # print('pred2 reverse is',pred2)
                y_pred_ens1[:, i, j] = pp[0]
                y_pred_ens2[:, i, j] = pp[1]
                if count % 1000 == 0:
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                    print('\r', end="")
                    print('Remain {fs} thound predictions'.format(fs=(x.shape[1] * x.shape[2] - count) / 1000))
                    sys.stdout.flush()
                time.sleep(0.0001)
                count = count + 1
    #  7月5  預測數據y_pred_ens1和y_pred_ens2是空的
    t_end = time.time()
    print('y_pred_ens shape is', y_pred_ens.shape)

    print('scaler shape is', scaler.shape)
    y_true_mask1 = y_true1[mask1]
    y_true_mask2 = y_true2[mask2]
    # y_pred_ens_mask = y_pred_ens[mask]
    y_pred_ens_mask1 = y_pred_ens1[mask1]
    y_pred_ens_mask2 = y_pred_ens2[mask2]
    print('y_true_mask shape is : {ts}'.format(ts=y_true_mask1.shape))
    print('y_true_mask shape is : {ts}'.format(ts=y_true_mask2.shape))
    # print('the true label shape is: {ts} and the predicton shape is: {ps}'.format(ts=y_true.shape, ps=y_pred_ens.shape))
    print('the true label shape is: {ts} and the predicton shape is: {ps}'.format(ts=y_true1.shape,
                                                                                  ps=y_pred_ens1.shape))
    print('the true label shape is: {ts} and the predicton shape is: {ps}'.format(ts=y_true2.shape,
                                                                                  ps=y_pred_ens2.shape))
    # log
    r_rmse = _rmse(y_true_mask1, y_pred_ens_mask1)
    r2_ens1 = r2_score(y_true_mask1, y_pred_ens_mask1)
    # r2_ens2 = r2_score(y_true_mask2, y_pred_ens_mask2)
    r_ens1 = np.corrcoef(y_true_mask1, y_pred_ens_mask1)
    # r_ens2 = np.corrcoef(y_true_mask2, y_pred_ens_mask2)
    nt, nlat, nlon = y_true1.shape
    # mask
    # mask=y_test_mhl==y_test_mhl

    # cal perf

    r2_mhls = np.full((nlat, nlon), np.nan)
    r_mhls = np.full((nlat, nlon), np.nan)
    rmse_mhls = np.full((nlat, nlon), np.nan)
    r2_mhls1 = np.full((nlat, nlon), np.nan)
    r_mhls1 = np.full((nlat, nlon), np.nan)
    rmse_mhls1 = np.full((nlat, nlon), np.nan)
    a = np.array(y_true1)
    # 全都接近0  負數
    a1 = np.array(y_true2)
    # nlat 45   nlon90
    # for i in range(nlat):
    #     for j in range(nlon):
    #         if not (np.isnan(y_true1[:, i, j]).any()):
    #             r_mhls[i, j] = np.corrcoef(a[:, i, j], y_pred_ens1[:, i, j])[0, 1]
    #             rmse_mhls[i, j] = _rmse(a[:, i, j], y_pred_ens1[:, i, j])
    #             r2_mhls[i, j] = r2_score(a[:, i, j], y_pred_ens1[:, i, j])
    #         if not (np.isnan(y_true2[:, i, j]).any()):
    #             r_mhls1[i, j] = np.corrcoef(a1[:, i, j], y_pred_ens2[:, i, j])[0, 1]
    #             rmse_mhls1[i, j] = _rmse(a1[:, i, j], y_pred_ens2[:, i, j])
    #             r2_mhls1[i, j] = r2_score(a1[:, i, j], y_pred_ens2[:, i, j])
    # print('\033[1;31m%s\033[0m' %
    #       "Median r2 {:.3f} time cost {:.2f}".format(np.nanmedian(r2_mhls), t_end - t_begin))
    # print('\033[1;31m%s\033[0m' %
    #       "Median r_rmse {:.3f} time cost {:.2f}".format(np.nanmedian(rmse_mhls), t_end - t_begin))
    # print('\033[1;31m%s\033[0m' %
    #       "Median r {:.3f} time cost {:.2f}".format(np.nanmedian(r_mhls), t_end - t_begin))
    # print('\033[1;31m%s\033[0m' %
    #       "Median r2 {:.3f} time cost {:.2f}".format(np.nanmedian(r2_mhls1), t_end - t_begin))
    # print('\033[1;31m%s\033[0m' %
    #       "Median r_rmse {:.3f} time cost {:.2f}".format(np.nanmedian(rmse_mhls1), t_end - t_begin))
    # print('\033[1;31m%s\033[0m' %
    #       "Median r {:.3f} time cost {:.2f}".format(np.nanmedian(r_mhls1), t_end - t_begin))
    # print('\033[1;31m%s\033[0m' %
    #       "Median r {:.3f} time cost {:.2f}".format(np.nanmedian(r_ens2), t_end - t_begin))

    return y_pred_ens1, y_true1, y_pred_ens2, y_true2


def test_mtl(x, y, static, scaler, cfg, model,device):
    cls = Dataset(cfg)          
    model.eval()
    # get mean, std (1, ngrid, 6)
    #mean, std = np.array(scaler[0]), np.array(scaler[1])
    #mean = torch.from_numpy(mean).to(device)
    #std = torch.from_numpy(std).to(device)  
    # if y.shape[0]-cfg["seq_len"]-cfg["forcast_time"]!=365:
    #     seq_len = 7
    #     x = x[5-1:]
    #     y_pred_ens = np.zeros((y.shape[0]-seq_len-cfg['forcast_time']+1-5, y.shape[1], y.shape[2]))*np.nan
    #     y_pred_ens1 = np.zeros((y.shape[0]-seq_len-cfg['forcast_time']+1-5, y.shape[1], y.shape[2]))*np.nan
    #     y_pred_ens2 = np.zeros((y.shape[0]-seq_len-cfg['forcast_time']+1-5, y.shape[1], y.shape[2]))*np.nan
    #     y_true = y[seq_len+cfg['forcast_time']+5-1:,:,:,0]
    # else:
    y_pred_enses = []
    y_truees = []
    mask = []
    for i in range(cfg['num_repeat']):
        y_pred_ens = np.zeros((y.shape[0]-cfg["seq_len"]-cfg['forcast_time'], y.shape[1], y.shape[2]))*np.nan
        y_true = y[cfg["seq_len"]+cfg['forcast_time']:,:,:,i]
        y_pred_enses.append(y_pred_ens)
        y_truees.append(y_true)
        mask_time =y_true == y_true
        mask.append(mask_time)
    # y_pred_ens1 = np.zeros((y.shape[0]-cfg["seq_len"]-cfg['forcast_time'], y.shape[1], y.shape[2]))*np.nan
    # y_pred_ens2 = np.zeros((y.shape[0]-cfg["seq_len"]-cfg['forcast_time'], y.shape[1], y.shape[2]))*np.nan
    # # y2 = np.divide(y2,1000)
    # # y_true = y[cfg["seq_len"]+cfg['forcast_time']:,:,:,0]
    # y_true1 = y[cfg["seq_len"]+cfg['forcast_time']:,:,:,0]
    # y_true2 = y[cfg["seq_len"]+cfg['forcast_time']:,:,:,1]
    print('x shape is',x.shape)
    print('y_true shape is',y.shape)
    # print('the true label shape is: {ts} and the predicton shape is: {ps}'.format(ts=y_true1.shape, ps=y_pred_ens1.shape))
    # print('the true label shape is: {ts} and the predicton shape is: {ps}'.format(ts=y_true2.shape, ps=y_pred_ens2.shape))
    # mask1 = y_true1 == y_true1
    # mask2 = y_true2 == y_true2
    t_begin = time.time()
    # ------------------------------------------------------------------------------------------------------------------------------
    # for each grid by MTL model
    if cfg["modelname"] in ['MSLSTMModel','MMOE']:
        count = 1
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                xx = []
                for num in range(cfg['num_repeat']):
                    x_new, y_new, static_new = batcher_mtl(x[:, i, j, :], y[:, i, j, :], static[num][i, j, :], cfg["seq_len"],
                                                            cfg['forcast_time'])

                    xt = torch.from_numpy(x_new).to(device)
                    s = torch.from_numpy(static_new).to(device)
                    s = s.unsqueeze(1)
                    s = s.repeat(1, xt.shape[1], 1)
                    xt = torch.cat([xt, s], 2)
                    xx.append(xt)
                pred = model(xx, static_new)
#
                # pred = pred*std[i, j, :]+mean[i, j, :] #(nsample,1,1)

                # print('pred1  is', pred.shape)
                # print('pred2  is', pred2)
                #經過 反正則化出來的結果就是nan
                pp = []
                for ii in range(len(pred)):
                    p = pred[ii].cpu().detach().numpy()
                    p = np.squeeze(p)
                    if cfg["normalize"] and cfg['normalize_type'] in ['region']:
                        p = cls.reverse_normalize(p, 'output', scaler[:, i, j, ii], 'minmax', -1)
                    elif cfg["normalize"] and cfg['normalize_type'] in ['global']:
                        p = cls.reverse_normalize(p, 'output', scaler, 'minmax', -1)
                    pp.append(p)
                # print('pre reverse is',pred.shape)
                # print('pred2 reverse is',pred2)
                for iq in range(cfg['num_repeat']):
                    y_pred_enses[iq][:,i,j] = pp[iq]
                # y_pred_ens1[:, i, j] = pp[0]
                # y_pred_ens2[:, i, j] = pp[1]
                if count % 1000 == 0:
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                    print('\r', end="")
                    print('Remain {fs} thound predictions'.format(fs=(x.shape[1] * x.shape[2] - count) / 1000))
                    sys.stdout.flush()
                time.sleep(0.0001)
                count = count + 1
# ------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------

    #  7月5  預測數據y_pred_ens1和y_pred_ens2是空的
    t_end = time.time()
    print('y_pred_ens shape is',y_pred_ens.shape)


    print('scaler shape is',scaler.shape)


    return y_pred_enses,y_truees


