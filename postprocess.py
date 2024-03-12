import os
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils import unbiased_rmse, _bias, r2_score, _mse
from config import get_args
from utils import GetKGE, GetNSE, GetPCC, _plotloss, _plotbox, _boxkge, _boxnse, _boxpcc, GetMAE, GetRMSE, _boxbias
from loss import NaNMSELoss

def lon_transform(x):
    x_new = np.zeros(x.shape)
    x_new[:, :, :int(x.shape[2] / 2)] = x[:, :, int(x.shape[2] / 2):]
    x_new[:, :, int(x.shape[2] / 2):] = x[:, :, :int(x.shape[2] / 2)]
    return x_new

def two_dim_lon_transform(x):
  x_new = np.zeros(x.shape)
  x_new[:,:int(x.shape[1]/2)] = x[:,int(x.shape[1]/2):]
  x_new[:,int(x.shape[1]/2):] = x[:,:int(x.shape[1]/2)]
  return x_new

def postprocess(cfg):
    PATH = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg['spatial_resolution']) + '/'
    file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
    mask = np.load(PATH + file_name_mask)
    # ------------------------------------------------------------------------------------------------------------------------------
    if cfg['modelname'] in ['MHLSTMModel', 'MSLSTMModel', 'SoftMTLv1','MMOE','LSTM']:
        # if cfg['modelname'] == 'KDE_LSTM':
        print('KDE_MHLSTMModel ---> ')
        out_path_mhl = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg['spatial_resolution']) + '/' + cfg[
            'workname'] + '/' + cfg['modelname'] + '/focast_time ' + str(cfg['forcast_time']) + '/'
        # y_pred_mhls = np.load(out_path_mhl + '_predictions_s.npy')
        # y_test_mhls = np.load(out_path_mhl + 'observations_s.npy')
        if cfg['modelname'] in ['LSTM']:
            y_pred_mhls = np.load(out_path_mhl + '_predictions.npy')
            y_test_mhls = np.load(out_path_mhl + 'observations.npy')
        elif cfg['label'][0] in ['volumetric_soil_water_layer_1'] and cfg['label'][1] in ['total_evaporation']:
            with open(out_path_mhl + '_predictions_s.npy', 'rb') as f:
                y_pred_mhls = pickle.load(f)
            with open(out_path_mhl + 'observations_s.npy', 'rb') as f:
                y_test_mhls = pickle.load(f)
        else:
            y_pred_mhls = np.load(out_path_mhl + '_predictions_s.npy',allow_pickle=True)
            y_test_mhls = np.load(out_path_mhl + 'observations_s.npy',allow_pickle=True)


        if cfg['modelname'] in ['LSTM']:
            nt, nlat, nlon = y_test_mhls.shape
        else:
            nt, nlat, nlon = y_test_mhls[0].shape
            # y_test_mhls[1] = np.array(y_test_mhls)[1]
            # y_test_mhls[0] = np.array(y_test_mhls)[0]

        # 将数组或者矩阵存储为csv文件可以使用如下代码实现：
        #
        # numpy.savetxt('new.csv', my_matrix, delimiter=',')


        # y_pred_mhle = np.load(out_path_mhl + '_predictions_e.npy')
        # y_test_mhle = np.load(out_path_mhl + 'observations_e.npy')

        # print('y_pred_mhls='+y_pred_mhls.shape,'y_pred_mhlv'+y_pred_mhlv.shape,'y_est'+ y_test_mhl.shape)
        # get shape

        # mask
        # mask=y_test_mhl==y_test_mhl
        # cal perf
        # r2_mhls = np.full((nlat, nlon), np.nan)
        # urmse_mhls = np.full((nlat, nlon), np.nan)

        # rmse_mhls = np.full((nlat, nlon), np.nan)
        # bias_mhls = np.full((nlat, nlon), np.nan)

        r = []
        kge = []
        nse = []
        pcc = []
        loss_test = []
        r2 = []
        rmse = []
        bias = []
        lossmse = torch.nn.MSELoss()
        if cfg['modelname'] in ['LSTM']:
            r_mhls = np.full((nlat, nlon), np.nan)
            loss_time = np.full((nlat, nlon), np.nan)
            kge_time = np.full((nlat, nlon), np.nan)
            pcc_time = np.full((nlat, nlon), np.nan)
            nse_time = np.full((nlat, nlon), np.nan)
            bias_time = np.full((nlat, nlon), np.nan)
            r2_time = np.full((nlat, nlon), np.nan)
            rmse_time = np.full((nlat, nlon), np.nan)

            for i in range(nlat):
                for j in range(nlon):
                    if not (np.isnan(y_test_mhls[:, i, j]).any()):
                        r_mhls[i, j] = np.corrcoef(y_test_mhls[:, i, j], y_pred_mhls[:, i, j])[0, 1]
                        # kge_time[i, j] = GetKGE(y_pred_mhls[:, i, j], y_test_mhls[:, i, j])
                        # pcc_time[i, j] = GetPCC(y_pred_mhls[:, i, j], y_test_mhls[:, i, j])
                        # loss_time[i, j] = NaNMSELoss.fit(cfg,y_pred_mhls[:, i, j], y_test_mhls[:, i, j], lossmse)
                        # nse_time[i, j] = GetNSE(y_pred_mhls[:, i, j], y_test_mhls[:, i, j])
                        # bias_time[i, j] = _bias(y_pred_mhls[:, i, j], y_test_mhls[:, i, j])
                        r2_time[i, j] = r2_score(y_pred_mhls[:, i, j], y_test_mhls[:, i, j])
                        # rmse_time[i, j] = GetRMSE(y_pred_mhls[:, i, j], y_test_mhls[:, i, j])

            kge.append(kge_time)
            nse.append(nse_time)
            pcc.append(pcc_time)
            loss_test.append(loss_time)
            r2.append(r2_time)
            rmse.append(rmse_time)
            bias.append(bias_time)
            r.append(r_mhls)


        else:
            for num_repeat in range(cfg['num_repeat']):
                r_mhls = np.full((nlat, nlon), np.nan)
                loss_time = np.full((nlat, nlon), np.nan)
                kge_time = np.full((nlat, nlon), np.nan)
                pcc_time = np.full((nlat, nlon), np.nan)
                nse_time = np.full((nlat, nlon), np.nan)
                r2_time = np.full((nlat, nlon), np.nan)
                rmse_time = np.full((nlat, nlon), np.nan)
                bias_time = np.full((nlat, nlon), np.nan)
                ture = y_test_mhls[num_repeat]
                pred = y_pred_mhls[num_repeat]
                for i in range(nlat):
                    for j in range(nlon):
                        if not (np.isnan(ture[:, i, j]).any()):
                            r_mhls[i, j] = np.corrcoef(ture[:, i, j], pred[:, i, j])[0, 1]
                            # kge_time[i, j] = GetKGE(pred[:, i, j], ture[:, i, j])
                            # pcc_time[i, j] = GetPCC(pred[:, i, j], ture[:, i, j])
                            # loss_time[i, j] = NaNMSELoss.fit(cfg,pred[:, i, j], ture[:, i, j], lossmse)
                            # nse_time[i, j] = GetNSE(pred[:, i, j], ture[:, i, j])
                            r2_time[i, j] = r2_score(pred[:, i, j], ture[:, i, j])
                            # rmse_time[i, j] = GetRMSE(pred[:, i, j], ture[:, i, j])
                            # bias_time[i, j] = _bias(pred[:, i, j], ture[:, i, j])

                kge.append(kge_time)
                nse.append(nse_time)
                pcc.append(pcc_time)
                loss_test.append(loss_time)
                r2.append(r2_time)
                rmse.append(rmse_time)
                bias.append(bias_time)
                r.append(r_mhls)

        # _plotbox(cfg, loss_test)
        # _boxkge(cfg, kge)
        # _boxpcc(cfg, pcc)
        # _boxnse(cfg, nse)
        # _boxbias(cfg, bias)

        # 画图
        # r_box = []
        # fig = plt.figure()
        # for num_repeat in range(cfg['num_repeat']):
        #     r_box.append(r[num_repeat][~np.isnan(r[num_repeat])])
        # data_r2 = r_box
        # ax = plt.subplot(111)
        # plt.ylabel('R')
        # ax.spines['left'].set_linewidth(2)
        # ax.spines['bottom'].set_linewidth(2)
        # ax.spines['right'].set_linewidth(2)
        # ax.spines['top'].set_linewidth(2)
        # ax.boxplot(data_r2,
        #            notch=True,
        #            patch_artist=True,
        #            showfliers=False,
        #            boxprops=dict(facecolor='lightblue', color='black'))
        # ax.set_xticks(range(1, len(cfg['label']) + 1))
        # ax.set_xticklabels(cfg['label'])
        # plt.show()





        if cfg['modelname'] in ['LSTM']:
            y_test_lstm = lon_transform(y_test_mhls)
            mask[-int(mask.shape[0] / 5.4):, :] = 0
            min_map = np.min(y_test_lstm, axis=0)
            max_map = np.max(y_test_lstm, axis=0)
            mask[min_map == max_map] = 0
            r[0] = r[0][mask == 1]
            # kge[0] = kge[0][mask == 1]
            # nse[0] = nse[0][mask == 1]
            # pcc[0] = pcc[0][mask == 1]
            r2[0] = r2[0][mask == 1]
            # rmse[0] = rmse[0][mask == 1]
            # bias[0] = bias[0][mask == 1]
            print('the average r of', cfg['label'], 'model is :', np.nanmedian(r[0]))
            # print('the average kge of', cfg['label'], 'model is :', np.nanmedian(kge[0]))
            # print('the average nse of', cfg['label'], 'model is :', np.nanmedian(nse[0]))
            print('the average r2 of', cfg['label'], 'model is :', np.nanmedian(r2[0]))
            # print('the average pcc of', cfg['label'], 'model is :', np.nanmedian(pcc[0]))
            # print('the average mse of', cfg['label'], 'model is :', np.nanmedian(loss_test[0]))
            # print('the average rmse of', cfg['label'], 'model is :', np.nanmedian(rmse[0]))
            # print('the average bias of', cfg['label'], 'model is :', np.nanmedian(bias[0]))
            # np.save(out_path_mhl + 'r_' + cfg['modelname'] + '.npy', r)
            np.save(out_path_mhl + 'r2_' + cfg['modelname'] + '.npy', r2)
            # np.save(out_path_mhl + 'kge_' + cfg['modelname'] + '.npy', kge)
            # np.save(out_path_mhl + 'bias_' + cfg['modelname'] + '.npy', bias)
            # np.save(out_path_mhl + 'rmse_' + cfg['modelname'] + '.npy', rmse)
            # np.save(out_path_mhl + 'nse_' + cfg['modelname'] + '.npy', nse)
        else:
            for i in range(cfg['num_repeat']):
                # 第一行是去掉南极部分，后三行应该是去掉值不发生变化的地区
                y_test_lstm = lon_transform(y_test_mhls[i])
                mask[-int(mask.shape[0] / 5.4):, :] = 0
                min_map = np.min(y_test_lstm, axis=0)
                max_map = np.max(y_test_lstm, axis=0)
                mask[min_map == max_map] = 0
                r[i] = r[i][mask[:,:,i] == 1]
                # kge[i] = kge[i][mask[:, :, i] == 1]
                # nse[i] = nse[i][mask[:,:,i] == 1]
                # pcc[i] = pcc[i][mask[:, :, i] == 1]
                r2[i] = r2[i][mask[:,:,i] == 1]
                # rmse[i] = rmse[i][mask[:,:,i] == 1]
                # bias[i] = bias[i][mask[:,:,i] == 1]
                print('the average r of', cfg['label'][i], 'model is :', np.nanmedian(r[i]))
                # print('the average kge of', cfg['label'][i], 'model is :', np.nanmedian(kge[i]))
                # print('the average nse of', cfg['label'][i], 'model is :', np.nanmedian(nse[i]))
                print('the average r2 of', cfg['label'][i], 'model is :', np.nanmedian(r2[i]))
                # print('the average pcc of', cfg['label'][i], 'model is :', np.nanmedian(pcc[i]))
                # print('the average mse of', cfg['label'][i], 'model is :', np.nanmedian(loss_test[i]))
                # print('the average rmse of', cfg['label'][i], 'model is :', np.nanmedian(rmse[i]))
                # print('the average bias of', cfg['label'][i], 'model is :', np.nanmedian(bias[i]))
            if cfg['label'][1] in ['total_evaporation']:
                # with open(out_path_mhl + 'r_' + cfg['modelname'] + '.npy', 'wb') as f:
                #     pickle.dump(r, f)
                with open(out_path_mhl + 'r2_' + cfg['modelname'] + '.npy', 'wb') as f:
                    pickle.dump(r2, f)
                # with open(out_path_mhl + 'kge_' + cfg['modelname'] + '.npy', 'wb') as f:
                #     pickle.dump(kge, f)
                # with open(out_path_mhl + 'bias_' + cfg['modelname'] + '.npy', 'wb') as f:
                #     pickle.dump(bias, f)
                # with open(out_path_mhl + 'rmse_' + cfg['modelname'] + '.npy', 'wb') as f:
                #     pickle.dump(rmse, f)
                # with open(out_path_mhl + 'nse_' + cfg['modelname'] + '.npy', 'wb') as f:
                #     pickle.dump(nse, f)
            else:
                # np.save(out_path_mhl + 'r_' + cfg['modelname'] + '.npy', r)
                np.save(out_path_mhl + 'r2_' + cfg['modelname'] + '.npy', r2)
                # np.save(out_path_mhl + 'kge_' + cfg['modelname'] + '.npy', kge)
                # np.save(out_path_mhl + 'bias_' + cfg['modelname'] + '.npy', bias)
                # np.save(out_path_mhl + 'rmse_' + cfg['modelname'] + '.npy', rmse)
                # np.save(out_path_mhl + 'nse_' + cfg['modelname'] + '.npy', nse)
        # print('the average rr of', cfg['modelname'], 'model is :', np.nanmedian(r_mhlr))
        # # print('the average re of', cfg['modelname'], 'model is :', np.nanmedian(r_mhle))
        # print('the average rmse_s of', cfg['modelname'], 'model is :', np.nanmedian(rmse_mhls))
        # print('the average rmse_r of', cfg['modelname'], 'model is :', np.nanmedian(rmse_mhlr))
        # # print('the average rmse_e of', cfg['modelname'], 'model is :', np.nanmedian(rmse_mhle))
        # print('the average bias_s of', cfg['modelname'], 'model is :', np.nanmedian(bias_mhls))
        # print('the average bias_r of', cfg['modelname'], 'model is :', np.nanmedian(bias_mhlr))
        # # print('the average bias_e of', cfg['modelname'], 'model is :', np.nanmedian(bias_mhle))
        # print('the average urmse_s of', cfg['modelname'], 'model is :', np.nanmedian(urmse_mhls))
        # print('the average urmse_r of', cfg['modelname'], 'model is :', np.nanmedian(urmse_mhlr))
        # print('the average urmse_e of', cfg['modelname'], 'model is :', np.nanmedian(urmse_mhle))

        print('postprocess ove, please go on')


    # ------------------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    cfg = get_args()
    postprocess(cfg)







