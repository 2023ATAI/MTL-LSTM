import time
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch.cuda import random
from tqdm import trange
from data_gen import load_test_data_for_rnn,load_test_data_for_rnn1,load_train_data_for_rnn,load_train_data_for_rnn1,load_test_data_for_cnn, load_train_data_for_cnn,erath_data_transform,sea_mask_rnn,sea_mask_cnn,sea_mask_rnn1
from loss import NaNMSELoss, NaNMSELoss1
from model import MSLSTMModel, SoftMTLv1, SoftMTLv2, MMOE, LSTMModel
from utils import _plotloss, _plotbox, GetKGE, _boxkge, _boxpcc, GetPCC, GetNSE, _boxnse


def train(x,
          y,
          static,
          mask, 
          scaler_x,
          scaler_y,
          cfg,
          num_repeat,
          PATH,
          out_path,
          device,
          num_task=None,
          valid_split=True):
    # seed 設置 確保每次的模型初始化、数据采样以及其他涉及随机性的操作都将是一致的
    SEED = 1024
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    random.seed()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    patience = cfg['patience']
    wait = 0
    best = 9999
    print('the device is {d}'.format(d=device))
    print('y type is {d_p}'.format(d_p=y.dtype))
    print('static type is {d_p}'.format(d_p=static[0].dtype))
    if cfg['modelname'] in ['CNN', 'ConvLSTM']:
#	Splice x according to the sphere shape
        lat_index,lon_index = erath_data_transform(cfg, x)
        print('\033[1;31m%s\033[0m' % "Applied Model is {m_n}, we need to transform the data according to the sphere shape".format(m_n=cfg['modelname']))
    if valid_split:
        nt,nlat,nlon,nf = x.shape  #x shape :nt,nf,nlat,nlon
	#Partition validation set and training set
        N = int(nt*cfg['split_ratio'])
        x_valid, y_valid, static_valid = x[N:], y[N:], static
        x, y = x[:N], y[:N]       

    lossmse = torch.nn.MSELoss()
#	filter Antatctica
    print('x_train shape is', x.shape)
    print('y_train shape is', y.shape)
    print('static_train shape is', static[0].shape)
    print('mask shape is', mask.shape)

    # mask see regions
    #Determine the land boundary
    if cfg['modelname'] in ['MHLSTMModel','MSLSTMModel',"SoftMTLv1","MMOE","LSTM"]:
        if valid_split:
            x_valid, y_valid ,static_valid = sea_mask_rnn1(cfg, x_valid, y_valid, static_valid, mask)
        x, y, static = sea_mask_rnn1(cfg, x, y, static, mask)

    # train and validate
    # NOTE: We preprare two callbacks for training:
    #       early stopping and save best model.
    for num_ in range(cfg['num_repeat']):
        # prepare models
	#Selection model
        if cfg['modelname'] in [ 'MSLSTMModel']:
            mtllstmmodel_cfg = {}
            mtllstmmodel_cfg['input_size'] = cfg["input_size"]
            mtllstmmodel_cfg['hidden_size'] = cfg["hidden_size"]*1
            mtllstmmodel_cfg['out_size'] = 1
            model = MSLSTMModel(cfg,mtllstmmodel_cfg).to(device)
        if cfg['modelname'] in [ 'SoftMTLv1']:
            softmtl_cfg = {}
            softmtl_cfg['input_size'] = cfg["input_size"]
            softmtl_cfg['hidden_size'] = cfg["hidden_size"]*1
            softmtl_cfg['out_size'] = 1
            model1 = SoftMTLv1(cfg,softmtl_cfg).to(device)
            model2 = SoftMTLv2(cfg,softmtl_cfg).to(device)
        if cfg['modelname'] in ['MMOE']:
            MMOEl_cfg = {}
            MMOEl_cfg['input_size'] = cfg["input_size"]
            MMOEl_cfg['hidden_size'] = cfg["hidden_size"]*1
            MMOEl_cfg['out_size'] = 1
            model = MMOE(cfg,MMOEl_cfg).to(device)
        if cfg['modelname'] in ['LSTM']:
            lstmmodel_cfg = {}
            lstmmodel_cfg['input_size'] = cfg["input_size"]
            lstmmodel_cfg['hidden_size'] = cfg["hidden_size"]*1
            lstmmodel_cfg['out_size'] = 1
            model = LSTMModel(cfg,lstmmodel_cfg).to(device)


        # if cfg['modelname'] in [ 'MHLSTMModel']:
        #     mtllstmmodel_cfg = {}
        #     mtllstmmodel_cfg['input_size'] = cfg["input_size"]
        #     mtllstmmodel_cfg['hidden_size'] = cfg["hidden_size"]*1
        #     mtllstmmodel_cfg['out_size'] = 1
        #     model = MHLSTMModel(cfg, mtllstmmodel_cfg).to(device)

      #  model.train()
	 # Prepare for training
    # NOTE: Only use `Adam`, we didn't apply adaptively
    #       learing rate schedule. We found `Adam` perform
    #       much better than `Adagrad`, `Adadelta`.
        if cfg["modelname"] in \
                    ['SoftMTLv1']:
            optimS = torch.optim.Adam([
                {'params': model1.parameters()},
                {'params': model2.parameters()}
            ], lr=cfg['learning_rate'])
        # optimH = torch.optim.Adam([
        #     {'params': model.lstm1.parameters()},
        #     {'params': model.drop.parameters()}
        # ], lr=cfg['learning_rate'])
    #     optimizer = torch.optim.Adam([{'params': model1.lstm1.parameters()}], lr=0.001)
    #     optim1 = torch.optim.Adam(model1.parameters(),lr=cfg['learning_rate'])
    #     optim2 = torch.optim.Adam(model2.parameters(),lr=cfg['learning_rate'])
        if cfg["modelname"] in \
                ['MMOE']:
            optimH = torch.optim.Adam([
                {'params': model.parameters()}
            ], lr=cfg['learning_rate'])
            optim1M = torch.optim.Adam(model.Towers[0].parameters(),lr=cfg['learning_rate'])
            optim2M = torch.optim.Adam(model.Towers[1].parameters(),lr=cfg['learning_rate'])
        if cfg["modelname"] in \
                    ['MSLSTMModel']:
            optimH = torch.optim.Adam([
                {'params': model.lstm1.parameters()}
            ], lr=cfg['learning_rate'])
            optimizers = []
            for i in range(cfg['num_repeat']):  # 用合适的参数初始化您的模型
                optimizer = torch.optim.Adam(model.head_layers[i].parameters(), lr=cfg['learning_rate'])  # 选择适当的优化器和学习率
                optimizers.append(optimizer)
            # optim1H = torch.optim.Adam(model.dense1.parameters(),lr=cfg['learning_rate'])
            # optim2H = torch.optim.Adam(model.dense2.parameters(),lr=cfg['learning_rate'])
        if cfg["modelname"] in \
                    ["LSTM"]:
            optim = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
        epoch_losses1 = []
        epoch_losses2 = []

        with trange(1, cfg['epochs']+1) as pbar:
            for epoch in pbar:
                pbar.set_description(
                    cfg['modelname']+' '+str(num_repeat))
                t_begin = time.time()
                # train
                MSELoss = 0
                for iter in range(0, cfg["niter"]):
 # ------------------------------------------------------------------------------------------------------------------------------
 #  train way for LSTM model
                    # 实际上这个是硬参数共享
                    if cfg["modelname"] in \
                            ['MSLSTMModel']:
                        x_batch,y_batch, aux_batch, _, _ = \
                            load_train_data_for_rnn1(cfg, x, y, static, scaler_y)

                        yy = []
                        xx = []
                        aux = []
                        for i in range(len(y_batch)):
                            pred_time = torch.from_numpy(y_batch[i]).to(device)
                            x_time = torch.from_numpy(x_batch[i]).to(device)
                            aux_time = torch.from_numpy(aux_batch[i]).to(device)
                            aux_time = aux_time.unsqueeze(1)
                            yy.append(pred_time)

                            aux_time = aux_time.repeat(1, x_time.shape[1], 1)
                            # print('aux_batch[:,5,0]',aux_batch[:,5,0])
                            # print('x_batch[:,5,0]',x_batch[:,5,0])
                            # x_batch =torch.Tensor(w)
                            x_time =torch.cat([x_time, aux_time], 2)
                            xx.append(x_time)
                            aux.append(aux_time)
                        pred = model(xx, aux)
                        #  64的shape
                        pp = []
                        for i in pred:
                            i = i.squeeze()
                            pp.append(i)

                    # MMoe  多门控网络
                    if cfg["modelname"] in \
                            ['MMOE']:
                        x_batch,y_batch, aux_batch, _, _ = \
                            load_train_data_for_rnn1(cfg, x, y, static, scaler_y)

                        yy = []
                        xx = []
                        for i in range(len(y_batch)) :
                            q = torch.from_numpy(y_batch[i]).to(device)
                            w = torch.from_numpy(x_batch[i]).to(device)
                            a = torch.from_numpy(aux_batch[i]).to(device)
                            a = a.unsqueeze(1)
                            yy.append(q)

                            a = a.repeat(1, w.shape[1], 1)
                            # print('aux_batch[:,5,0]',aux_batch[:,5,0])
                            # print('x_batch[:,5,0]',x_batch[:,5,0])
                            # x_batch =torch.Tensor(w)
                            w =torch.cat([w, a], 2)
                            xx.append(w)
                        pred = model(xx, aux_batch)
                        #  64的shape
                        pp = []
                        for i in pred:
                            i = i.squeeze()
                            pp.append(i)

                    if cfg["modelname"] in \
                            ['SoftMTLv1']:
                        # generate batch data for Recurrent Neural Network
                        x_batch,y_batch, aux_batch, _, _ = \
                            load_train_data_for_rnn1(cfg, x, y, static, scaler_y)

                        yy = []
                        xx = []
                        pp = []
                        for i in range(len(y_batch)) :
                            q = torch.from_numpy(y_batch[i]).to(device)
                            w = torch.from_numpy(x_batch[i]).to(device)
                            a = torch.from_numpy(aux_batch[i]).to(device)
                            a = a.unsqueeze(1)
                            yy.append(q)

                            a = a.repeat(1, w.shape[1], 1)
                            # print('aux_batch[:,5,0]',aux_batch[:,5,0])
                            # print('x_batch[:,5,0]',x_batch[:,5,0])
                            # x_batch =torch.Tensor(w)
                            w =torch.cat([w, a], 2)
                            if i == 0:
                                pred = model1(w, a)
                            if i == 1:
                                pred = model2(w, a)
                            pred = pred.squeeze()
                            pp.append(pred)

                    if cfg["modelname"] in \
                            ['LSTM']:
                        # generate batch data for Recurrent Neural Network
                        if isinstance(y, list):
                            y = y[0]
                        x_batch, y_batch, aux_batch, _, _ = \
                            load_train_data_for_rnn(cfg, x, y, static, scaler_y)
                        x_batch = torch.from_numpy(x_batch).to(device)
                        aux_batch = torch.from_numpy(aux_batch).to(device)
                        y_batch = torch.from_numpy(y_batch).to(device)
                        aux_batch = aux_batch.unsqueeze(1)
                        aux_batch = aux_batch.repeat(1,x_batch.shape[1],1)
                        #print('aux_batch[:,5,0]',aux_batch[:,5,0])
                        #print('x_batch[:,5,0]',x_batch[:,5,0])
                        x_batch = torch.cat([x_batch, aux_batch], 2)
                        # 与aux无关 每次pred输出的都是新的shape
                        pred = model(x_batch, aux_batch)
                        # print('pred1',pred.shape)
                        pred = torch.squeeze(pred,1)
 #  train way for CNN model
 # ------------------------------------------------------------------------------------------------------------------------------
 # 用兩個模型的lstm層的參數進行L2約束 聯合更新lstm
                    if cfg["modelname"] in ['SoftMTLv1']:
                        w1 = model1.lstm.parameters()
                        w2 = model2.lstm.parameters()
                        # print(w1,w2)
                        loss1 = NaNMSELoss.fit(cfg, pp[0].float(), yy[0].float(), lossmse)
                        loss2 = NaNMSELoss.fit(cfg, pp[1].float(), yy[1].float(), lossmse)

                        # 7月14  修改昨天的錯誤之後 效果好了很多 或許硬共享也可以嘗試一下了  然後跑了個下面的0.01看看效果
                        # 任務2的R 出不來 看看是什麽情況
                        loss = 0.3*loss1+0.7*loss2
                        # for param1, param2 in zip(model1.lstm.parameters(), model2.lstm.parameters()):
                        #     loss += 0.001*(torch.norm(param1 - param2, p=2) ** 2)
                        # losses = [loss1 , loss2]
                        # weight = F.softmax(torch.randn(2), dim=-1)
                        # weight = weight.to(device)
                        # Random Loss Weighting  簡稱RLW
                        # loss 包含所有的損失 與隨機分配的權重張量相乘
                        # loss = sum(losses[i] * weight[i] for i in range(2))
                        for param1, param2 in zip(model1.lstm.parameters(), model2.lstm.parameters()):
                            loss += 0.001*(torch.norm(param1 - param2, p=2) ** 2)
                        # optim1.zero_grad()
                        # optim2.zero_grad()
                        optimS.zero_grad()
                        # loss1.backward()
                        # loss2.backward()
                        loss.backward()
                        # optim1.step()
                        # optim2.step()
                        optimS.step()

                    elif cfg["modelname"] in ['MMOE']:

                        loss1 = NaNMSELoss.fit(cfg, pp[0].float(), yy[0].float(), lossmse)
                        loss2 = NaNMSELoss.fit(cfg, pp[1].float(), yy[1].float(), lossmse)
                        losses = [loss1 , loss2]
                        weight = F.softmax(torch.randn(2), dim=-1)
                        weight = weight.to(device)
                        # Random Loss Weighting  簡稱RLW
                        # loss 包含所有的損失 與隨機分配的權重張量相乘
                        loss = sum(losses[i] * weight[i] for i in range(2))
                        # loss = torch.sum(loss*weight)

                        optimH.zero_grad()
                        optim1M.zero_grad()
                        optim2M.zero_grad()

                        loss.backward(retain_graph=True)
                        loss1.backward(retain_graph=True)
                        loss2.backward(retain_graph=True)

                        optimH.step()
                        optim1M.step()
                        optim2M.step()
                        # 損失函數曲綫
                    elif cfg["modelname"] in ['MSLSTMModel']:
                        losses = []

                        for i in range(cfg['num_repeat']):
                            loss_time = NaNMSELoss.fit(cfg, pp[i].float(), yy[i].float(), lossmse)
                            losses.append(loss_time)

                        optimH.zero_grad()
                        # 权重分配！
                        loss = 0.2 * losses[0] + 0.8 * losses[1]
                        # weight = F.softmax(torch.randn(cfg['num_repeat']), dim=-1)
                        # weight = weight.to(device)
                        # Random Loss Weighting  簡稱RLW
                        # loss 包含所有的損失 與隨機分配的權重張量相乘
                        # loss = sum(losses[i] * weight[i] for i in range(cfg['num_repeat']))
                        loss.backward(retain_graph=True)
                        for i in range(cfg['num_repeat']):
                            op = optimizers[i]
                            op.zero_grad()
                            losses[i].backward(retain_graph=True)
                            op.step()
                        optimH.step()

                    elif cfg["modelname"] in ['LSTM']:
                        loss = NaNMSELoss1.fit(cfg, pred.float(), y_batch.float(), lossmse)
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        MSELoss += loss.item()
# ------------------------------------------------------------------------------------------------------------------------------
                if cfg["modelname"] in ['MSLSTMModel','MHLSTMModel','SoftMTLv1','MMOE']:
                    t_end = time.time()
                    # get loss log
                    loss_str = "Epoch {} Loss {:.3f};Loss1 {:.3f};time {:.2f}".format(
                        epoch, loss,losses[0],
                        t_end - t_begin)
                    # loss_str1 = "Epoch {} Train loss1 Loss {:.3f} time {:.2f}".format(epoch, loss1 / cfg["niter"],
                    #                                                                t_end - t_begin)
                    # loss_str2 = "Epoch {} Train loss2 Loss {:.3f} time {:.2f}".format(epoch, loss2 / cfg["niter"],
                    #                                                                t_end - t_begin)
                    print(loss_str)
                else:
                    t_end = time.time()
                    # get loss log
                    loss_str = "Epoch {} Train MSE Loss {:.3f} time {:.2f}".format(epoch, MSELoss / cfg["niter"],
                                                                                   t_end - t_begin)
                    print(loss_str)

                # validate
		#Use validation sets to test trained models
		#If the error is smaller than the minimum error, then save the model.
                if valid_split:
                    del x_batch, y_batch, aux_batch
                    MSE_valid_loss = 0
                    if epoch % 20 == 0:

                        wait += 1
                        # NOTE: We used grids-mean NSE as valid metrics.
                        t_begin = time.time()
# ------------------------------------------------------------------------------------------------------------------------------
 #  validate way for LSTM model
                        if cfg["modelname"] in ['MSLSTMModel']:
                            gt_list = [i for i in range(0,x_valid[0].shape[0]-cfg['seq_len'],cfg["stride"])]
                            n = (x_valid[0].shape[0]-cfg["seq_len"])//cfg["stride"]

                            losses_val = []

                            for i in range(0, n):
                                #mask
                                x_valid_batch,y_valid_batch, aux_valid_batch, _, _ = \
                         load_test_data_for_rnn1(cfg, x_valid, y_valid, static_valid, scaler_y,cfg["stride"], i, n)
                                yy = []
                                xx = []
                                pred_valid = []
                                for ii in range(len(y_valid_batch)):
                                    q = torch.from_numpy(y_valid_batch[ii]).to(device)
                                    w = torch.from_numpy(x_valid_batch[ii]).to(device)
                                    a = torch.from_numpy(aux_valid_batch[ii]).to(device)
                                    a = a.unsqueeze(1)
                                    yy.append(q)

                                    a = a.repeat(1, w.shape[1], 1)
                                    # print('aux_batch[:,5,0]',aux_batch[:,5,0])
                                    # print('x_batch[:,5,0]',x_batch[:,5,0])
                                    # x_batch =torch.Tensor(w)
                                    w = torch.cat([w, a], 2)
                                    xx.append(w)
                                with torch.no_grad():
                                    pred_valid = model(xx,aux_valid_batch)


                                loss_weight = []
                                for ini in range(cfg['num_repeat']):
                                    loss_ = NaNMSELoss.fit(cfg, pred_valid[ini].squeeze().float(), yy[ini].squeeze().float(), lossmse)
                                    loss_weight.append(loss_)
                                weight = F.softmax(torch.randn(cfg['num_repeat']), dim=-1)
                                weight = weight.to(device)

                                # Random Loss Weighting  簡稱RLW
                                # loss 包含所有的損失 與隨機分配的權重張量相乘
                                # mse_valid_loss = sum(loss_weight[i] * weight[i] for i in range(cfg['num_repeat']))
                                mse_valid_loss = 0.2 * loss_weight[0] + 0.8 * loss_weight[1]
                                # for i in range(len(pred)):
                                #     loss += NaNMSELoss.fit(cfg, pred[i].float(), y_batch[i].float(), lossmse)

                                # 交叉熵
                                # mse_valid_loss2 = Cross.fit(cfg, pred_valid2.squeeze(1), y_valid_batch_v,lossmse_cross)
                                MSE_valid_loss += mse_valid_loss.item()
                                print('after_mse_valid_loss', mse_valid_loss)


                        #         7月13   驗證集的方法沒用上  沒使用驗證集   現在就是把驗證集 用上   看看效果如何，今天所試 發現多個模型效果還不如一個模型  可能是約束策略有問題
                        if cfg["modelname"] in ['SoftMTLv1']:
                            gt_list = [i for i in range(0,x_valid.shape[0]-cfg['seq_len'],cfg["stride"])]
                            n = (x_valid.shape[0]-cfg["seq_len"])//cfg["stride"]
                            for i in range(0, n):
                                #mask
                                x_valid_batch,y_valid_batch, aux_valid_batch, _, _ = \
                         load_test_data_for_rnn1(cfg, x_valid, y_valid, static_valid, scaler_y,cfg["stride"], i, n)
                                yy = []
                                xx = []
                                pred_valid = []
                                for i in range(len(y_valid_batch)):
                                    q = torch.from_numpy(y_valid_batch[i]).to(device)
                                    w = torch.from_numpy(x_valid_batch[i]).to(device)
                                    a = torch.from_numpy(aux_valid_batch[i]).to(device)
                                    a = a.unsqueeze(1)
                                    yy.append(q)

                                    a = a.repeat(1, w.shape[1], 1)
                                    # print('aux_batch[:,5,0]',aux_batch[:,5,0])
                                    # print('x_batch[:,5,0]',x_batch[:,5,0])
                                    # x_batch =torch.Tensor(w)
                                    w = torch.cat([w, a], 2)
                                    with torch.no_grad():
                                        if i == 0:
                                            pred = model1(w, a)
                                        if i == 1:
                                            pred = model2(w, a)
                                        pred = pred.squeeze()
                                        pred_valid.append(pred)

                                    # print('before_pred_vaild1', pred_valid1)
                                    # print('before_pred_vaild2', pred_valid2)
                                    # for i in range(cfg['num_repeat']):
                                    #     pred_valid[i] = pred_valid_f[i]
                                    #     loss_v_sum = loss_v_sum + pred_valid_f[i]

                                # mse_valid_loss = NaNMSELoss.fit(cfg, pred_valid.squeeze(1), y_valid_batch,lossmse)

                                loss1 = NaNMSELoss.fit(cfg, pred_valid[0].float(), yy[0].float(), lossmse)
                                loss2 = NaNMSELoss.fit(cfg, pred_valid[1].float(), yy[1].float(), lossmse)
                                kge1 = GetKGE(pred_valid[0].float(), yy[0].float())
                                kge2 = GetKGE(pred_valid[1].float(), yy[1].float())
                                pcc1 = GetPCC(pred_valid[0].float(), yy[0].float())
                                pcc2 = GetPCC(pred_valid[1].float(), yy[1].float())
                                nse1 = GetNSE(pred_valid[0].float(), yy[0].float())
                                nse2 = GetNSE(pred_valid[1].float(), yy[1].float())
                                # losses = [loss1, loss2]
                                # weight = F.softmax(torch.randn(2), dim=-1)
                                # weight = weight.to(device)
                                mse_valid_loss = 0.2*loss1 + 0.8 * loss2
                                # Random Loss Weighting  簡稱RLW
                                # loss 包含所有的損失 與隨機分配的權重張量相乘
                                # mse_valid_loss = sum(losses[i] * weight[i] for i in range(2))

                                # for i in range(len(pred)):
                                #     loss += NaNMSELoss.fit(cfg, pred[i].float(), y_batch[i].float(), lossmse)

                                # 交叉熵
                                # mse_valid_loss2 = Cross.fit(cfg, pred_valid2.squeeze(1), y_valid_batch_v,lossmse_cross)
                                MSE_valid_loss += mse_valid_loss.item()
                                print('after_mse_valid_loss', mse_valid_loss)
                        if cfg["modelname"] in ['MMOE']:
                            gt_list = [i for i in range(0,x_valid.shape[0]-cfg['seq_len'],cfg["stride"])]
                            n = (x_valid.shape[0]-cfg["seq_len"])//cfg["stride"]
                            for i in range(0, n):
                                #mask
                                x_valid_batch,y_valid_batch, aux_valid_batch, _, _ = \
                         load_test_data_for_rnn1(cfg, x_valid, y_valid, static_valid, scaler_y,cfg["stride"], i, n)
                                yy = []
                                xx = []
                                pred_valid = []
                                for i in range(len(y_valid_batch)):
                                    q = torch.from_numpy(y_valid_batch[i]).to(device)
                                    w = torch.from_numpy(x_valid_batch[i]).to(device)
                                    a = torch.from_numpy(aux_valid_batch[i]).to(device)
                                    a = a.unsqueeze(1)
                                    yy.append(q)

                                    a = a.repeat(1, w.shape[1], 1)
                                    # print('aux_batch[:,5,0]',aux_batch[:,5,0])
                                    # print('x_batch[:,5,0]',x_batch[:,5,0])
                                    # x_batch =torch.Tensor(w)
                                    w = torch.cat([w, a], 2)
                                    xx.append(w)
                                with torch.no_grad():
                                    pred_valid = model(xx,aux_valid_batch)

                                    # print('before_pred_vaild1', pred_valid1)
                                    # print('before_pred_vaild2', pred_valid2)
                                    # for i in range(cfg['num_repeat']):
                                    #     pred_valid[i] = pred_valid_f[i]
                                    #     loss_v_sum = loss_v_sum + pred_valid_f[i]

                                # mse_valid_loss = NaNMSELoss.fit(cfg, pred_valid.squeeze(1), y_valid_batch,lossmse)
                                loss1 = NaNMSELoss.fit(cfg, pred_valid[0].float(), yy[0].float(), lossmse)
                                loss2 = NaNMSELoss.fit(cfg, pred_valid[1].float(), yy[1].float(), lossmse)
                                loss = [loss1, loss2]
                                weight = F.softmax(torch.randn(2), dim=-1)
                                weight = weight.to(device)
                                # Random Loss Weighting  簡稱RLW
                                mse_valid_loss = sum(loss[i] * weight[i] for i in range(2))



                                # for i in range(len(pred)):
                                #     loss += NaNMSELoss.fit(cfg, pred[i].float(), y_batch[i].float(), lossmse)

                                # 交叉熵
                                # mse_valid_loss2 = Cross.fit(cfg, pred_valid2.squeeze(1), y_valid_batch_v,lossmse_cross)
                                MSE_valid_loss += mse_valid_loss.item()
                                print('after_mse_valid_loss', mse_valid_loss)
                        if cfg["modelname"] in ['LSTM']:
                            gt_list = [i for i in range(0,x_valid.shape[0]-cfg['seq_len'],cfg["stride"])]
                            n = (x_valid.shape[0]-cfg["seq_len"])//cfg["stride"]
                            for i in range(0, n):
                                #mask
                                if isinstance(y_valid, list):
                                    y_valid = y_valid[0]
                                x_valid_batch, y_valid_batch, aux_valid_batch, _, _ = \
                         load_test_data_for_rnn(cfg, x_valid, y_valid, static_valid, scaler_y,cfg["stride"], i, n)
                                x_valid_batch = torch.Tensor(x_valid_batch).to(device)
                                y_valid_batch = torch.Tensor(y_valid_batch).to(device)
                                aux_valid_batch = torch.Tensor(aux_valid_batch).to(device)
                                aux_valid_batch = aux_valid_batch.unsqueeze(1)
                                aux_valid_batch = aux_valid_batch.repeat(1,x_valid_batch.shape[1],1)
                                x_valid_batch = torch.cat([x_valid_batch, aux_valid_batch], 2)
                                with torch.no_grad():
                                    pred_valid = model(x_valid_batch, aux_valid_batch)
                                mse_valid_loss = NaNMSELoss.fit(cfg, pred_valid.squeeze(1), y_valid_batch,lossmse)
                                MSE_valid_loss += mse_valid_loss.item()


#  validate way for CNN model
# ------------------------------------------------------------------------------------------------------------------------------


                        if cfg["modelname"] in ['MSLSTMModel','SoftMTLv1','MMOE']:
                            t_end = time.time()
                            print('gt_list=',gt_list)
                            print('gt_list.len=',len(gt_list))
                            mse_valid_loss = MSE_valid_loss / (len(gt_list))
                            print(mse_valid_loss)
                            # get loss log
                            loss_str = '\033[1;31m%s\033[0m' % \
                                       "Epoch {} Val MSE Loss {:.3f}  time {:.2f}".format(epoch, mse_valid_loss,
                                                                                          t_end - t_begin)
                            print(loss_str)
                            val_save_acc = mse_valid_loss
                        else:
                            t_end = time.time()
                            mse_valid_loss = MSE_valid_loss/(len(gt_list))
                            # get loss log
                            loss_str = '\033[1;31m%s\033[0m' % \
                                    "Epoch {} Val MSE Loss {:.3f}  time {:.2f}".format(epoch,mse_valid_loss,
                                        t_end-t_begin)
                            print(loss_str)
                            val_save_acc = mse_valid_loss

                        # save best model by val loss
                        # NOTE: save best MSE results get `single_task` better than `multi_tasks`
                        #       save best NSE results get `multi_tasks` better than `single_task`
                        if val_save_acc < best:
                        #if MSE_valid_loss < best:
                            if cfg["modelname"] in ['SoftMTLv1']:
                                torch.save(model1,out_path+cfg['modelname']+'_para.pkl')
                                torch.save(model2,out_path+'SoftMTLv2'+'_para.pkl')
                                wait = 0  # release wait
                                best = val_save_acc #MSE_valid_loss
                                print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')
                            else:
                                torch.save(model,out_path+cfg['modelname']+'_para.pkl')
                                wait = 0  # release wait
                                best = val_save_acc #MSE_valid_loss
                                print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')
                else:
                    if cfg["modelname"] in ['SoftMTLv1']:
                        if MSELoss < best:
                            best = MSELoss
                            wait = 0
                            torch.save(model1, out_path + cfg['modelname'] + '_para.pkl')
                            torch.save(model2, out_path + 'SoftMTLv2' + '_para.pkl')
                            print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')
                    # save best model by train loss
                    if MSELoss < best:
                        best = MSELoss
                        wait = 0
                        torch.save(model,out_path+cfg['modelname']+'_para.pkl')
                        print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')

                # early stopping
                if wait >= patience:
                    # print(epoch_losses_sum[0].shape)
                    # _plotloss(cfg,epoch_losses_sum)
                    # _plotbox(cfg,epoch_losses_sum)
                    # _boxkge(cfg,kge)
                    # _boxpcc(cfg,pcc)
                    # _boxpcc_data(cfg,pcc_data)
                    # _boxnse(cfg,nse)
                    return
            return


