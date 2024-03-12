import numpy as np
import torch


def sea_mask_mtl(cfg, x, y, aux, mask):
    #print('x shape are', x.shape)
    x = x.transpose(0,3,1,2)
    #print('x new shape are', x.shape)
    y = y.transpose(0,3,1,2)
    aux = aux.transpose(2,0,1)
   # scaler = scaler.transpose(0,3,1,2)
    x = x.reshape(x.shape[0],x.shape[1],x.shape[2]*x.shape[3])
    # print('y shape=',y.shape) 8760,2,45,90,   8764,1,45,90
    # y1=y[:,0:1,:,:]
    # y2=y[:,1:2,:,:]
    y1=y[:,0:1,:,:]
    y2=y[:,1:2,:,:]
    # print('y1 shape = ',y1)
    # print('y2 shape = ',y2)
    y1 = y1.reshape(y1.shape[0],y1.shape[2]*y1.shape[3])
    y2 = y2.reshape(y2.shape[0],y2.shape[2]*y2.shape[3])
    aux = aux.reshape(aux.shape[0],aux.shape[1]*aux.shape[2])
    print('before mask',mask)
    mask = mask.reshape(mask.shape[0]*mask.shape[1])
    print('after mask',mask.shape)
    x = x[:,:,mask==1]
    #cause we got two task so wo return two y
    y1 = y1[:,mask==1]
    y2 = y2[:,mask==1]
    aux = aux[:,mask==1]
    return x, y1, y2, aux

def sea_mask_rnn(cfg, x, y, aux, mask):
    #print('x shape are', x.shape)
    x = x.transpose(0,3,1,2)
    #print('x new shape are', x.shape)
    y = y.transpose(0,3,1,2)
    aux = aux.transpose(2,0,1)
   # scaler = scaler.transpose(0,3,1,2)
    x = x.reshape(x.shape[0],x.shape[1],x.shape[2]*x.shape[3])
    y = y.reshape(y.shape[0],y.shape[2]*y.shape[3])
    aux = aux.reshape(aux.shape[0],aux.shape[1]*aux.shape[2])
    mask = mask.reshape(mask.shape[0]*mask.shape[1])
    x = x[:,:,mask==1]
    y = y[:,mask==1]
    aux = aux[:,mask==1]
    return x, y, aux
def sea_mask_rnn1(cfg, x, y, aux, mask):
    #print('x shape are', x.shape)
    x = x.transpose(0,3,1,2)   # t,nf,45,90
    #print('x new shape are', x.shape)
    y = y.transpose(0,3,1,2)
  # 1,45,90
   # scaler = scaler.transpose(0,3,1,2)
    x = x.reshape(x.shape[0],x.shape[1],x.shape[2]*x.shape[3])
    y_list = []
    x_list = []
    aux_list = []

    mask = mask.reshape(mask.shape[0]*mask.shape[1],mask.shape[2])
    for i in range(y.shape[1]):
        aux_ = aux[i].transpose(2, 0, 1)
        aux_ = aux_.reshape(aux_.shape[0], aux_.shape[1] * aux_.shape[2])
        x_ = x[:, :, mask[:,i] == 1]
        aux_ = aux_[:, mask[:,i] == 1]
        yt = np.reshape(y[:, i:i + 1, :, :], (y.shape[0], y.shape[2] * y.shape[3]))
        # yt = y[:, i:i+1,:,:]
        # yt = yt.reshape(yt.shape[0], yt.shape[2] * yt.shape[3])
        yt = yt[:, mask[:,i] == 1]
        print("yt.shape = ",yt.shape)

        y_list.append(yt)
        x_list.append(x_)
        aux_list.append(aux_)

    return x_list, y_list, aux_list

def sea_mask_cnn(cfg, x, y, aux, mask):
    x = x.transpose(0,3,1,2)
    y = y.transpose(0,3,1,2)
    aux = aux.transpose(2,0,1)
    #scaler = scaler.transpose(0,3,1,2)
    nt, nf, nlat, nlon = x.shape
    ngrid = nlat * nlon
    _index = np.array([i for i in range(0,ngrid,1)])
    mask = mask.reshape(mask.shape[0]*mask.shape[1])
    mask_index = _index[mask==1]
    return x, y, aux, mask_index



# NOTE: `load_train_data` and `load_test_data` is based on
#       Fang and Shen(2020), JHM. It doesn't used all samples
#       (all timestep over all grids)to train LSTM model. 
#       Otherwise, they construct train samples by select 
#       `batch_size` grids, and select `seq_len` timesteps.
#       We found that this method suit for training data that
#       has large similarity (e.g., model's output, reanalysis)
#       However, if we trained on in-situ data such as CAMELE
#       Kratzert et al. (2019), HESS is better.    
#  
# Fang and Shen (2020), JHM
def load_train_data_for_rnn(cfg, x, y, aux, scaler):
    nt, nf, ngrid = x.shape
    mean, std = np.array(scaler[0]), np.array(scaler[1])
    #mean = mean.reshape(mean.shape[0],mean.shape[1]*mean.shape[2])
    #std = std.reshape(std.shape[0],std.shape[1]*std.shape[2])
    idx_time = np.random.randint(0, nt-cfg['seq_len']-cfg["forcast_time"], 1)[0]
    idx_grid = np.random.randint(0, ngrid, cfg['batch_size'])
    x = np.transpose(x, (2,0,1))
    y = np.transpose(y, (1,0))
    aux = np.transpose(aux, (1,0))
    x = x[idx_grid, idx_time:idx_time+cfg['seq_len']]
    y = y[idx_grid, idx_time+cfg['seq_len']+cfg["forcast_time"]] ##
    aux = aux[idx_grid]
    y[np.isinf(y)]=np.nan
    mask = y == y
    x = x[mask]
    y = y[mask]
    aux = aux[mask]
    x[np.isinf(x)]=np.nan
    x = np.nan_to_num(x)
    return x, y, aux, mean, std

def load_train_data_for_rnn1(cfg, x, y, aux, scaler):
    nt, nf, ngrid = x[0].shape
    mean, std = np.array(scaler[0]), np.array(scaler[1])
    #mean = mean.reshape(mean.shape[0],mean.shape[1]*mean.shape[2])
    #std = std.reshape(std.shape[0],std.shape[1]*std.shape[2])
    #随机选一天和这一天后的365天
    idx_time = np.random.randint(0, nt-cfg['seq_len']-cfg["forcast_time"], 1)[0]
    # 在格点中随机选择batch_size个点
    idx_grid = np.random.randint(0, ngrid, cfg['batch_size'])


    yy = []
    xx = []
    aa = []
    for i in range(len(y)):
        xt = np.transpose(x[i], (2, 0, 1))
        y_ = np.transpose(y[i], (1, 0))
        auxt = np.transpose(aux[i], (1, 0))
        # 这里就是上面选择的batch_size个点和选中时间的数据
        xt = xt[idx_grid, idx_time:idx_time + cfg['seq_len']]
        y_ = y_[idx_grid, idx_time+cfg['seq_len']+cfg["forcast_time"]]
        auxt = auxt[idx_grid]

        y_[np.isinf(y_)]=np.nan
        mask1 = y_ == y_
        xt = xt[mask1]
        y_ = y_[mask1]
        auxt = auxt[mask1]
        xt[np.isinf(xt)] = np.nan
        xt = np.nan_to_num(xt)


        yy.append(y_)
        xx.append(xt)
        aa.append(auxt)
    # print("yy.shaoe = ", len(yy))
    # yy = torch.tensor(yy)
    # yy = torch.cat(yy, dim=-1)
    # print("yy.shaoe = ",yy.shape)

    return xx, yy, aa, mean, std


def load_test_data_for_rnn(cfg, x, y, aux, scaler, stride,i, n):

    nt, nf, ngrid = x.shape
    x = np.transpose(x, (2,0,1))
    y = np.transpose(y, (1,0))
    aux = np.transpose(aux, (1,0))

    mean, std = np.array(scaler[0]), np.array(scaler[1])
    x_new = x[:,i*stride:i*stride+cfg["seq_len"],:][0:ngrid:2*stride,:,:]
    y_new = y[0:ngrid:2*stride,i*stride+cfg["seq_len"]+cfg["forcast_time"]]

    aux_new = aux[0:ngrid:2*stride,:]

    y_new[np.isinf(y_new)] = np.nan
    mask = y_new == y_new
    x_new = x_new[mask]
    y_new = y_new[mask]
    aux_new = aux_new[mask]
    x_new[np.isinf(x_new)] = np.nan
    x_new = np.nan_to_num(x_new)
    return x_new, y_new, aux_new, np.tile(mean, (1, n, 1)), np.tile(std, (1,n,1))


def load_test_data_for_rnn1(cfg, x, y, aux, scaler, stride,i, n):


    nt, nf, ngrid = x[0].shape


    yy=[]
    xx=[]
    aa=[]
    mean, std = np.array(scaler[0]), np.array(scaler[1])

    for j in range(len(y)):
        x_ = np.transpose(x[j], (2, 0, 1))
        aux_ = np.transpose(aux[j], (1, 0))
        y_ = np.transpose(y[j], (1,0))
        aux_new = aux_[0:ngrid:2 * stride, :]
        x_new = x_[:, i * stride:i * stride + cfg["seq_len"], :][0:ngrid:2 * stride, :, :]
        y_new = y_[0:ngrid:2*stride,i*stride+cfg["seq_len"]+cfg["forcast_time"]]
        y_new[np.isinf(y_new)] = np.nan
        mask = y_new == y_new
        y_new = y_new[mask]
        x_new = x_new[mask]
        x_new[np.isinf(x_new)] = np.nan
        x_new = np.nan_to_num(x_new)
        aux_new = aux_new[mask]
        yy.append(y_new)
        xx.append(x_new)
        aa.append(aux_new)

    return xx, yy, aa, np.tile(mean, (1, n, 1)), np.tile(std, (1,n,1))

'''
def load_test_data_for_rnn(cfg, x, y, aux, scaler, stride,i, n):
    x = x.transpose(0,3,1,2)
    y = y.transpose(0,3,1,2)
    aux = aux.transpose(2,0,1)
    x = x.reshape(x.shape[0],x.shape[1],x.shape[2]*x.shape[3])
    nt, nf, ngrid = x.shape
    y = y.reshape(y.shape[0],y.shape[2]*y.shape[3])
    aux = aux.reshape(aux.shape[0],aux.shape[1]*aux.shape[2])

    x_new = np.zeros((ngrid//(2*stride), cfg["seq_len"], nf))*np.nan
    y_new = np.zeros((ngrid//(2*stride),1))*np.nan
    aux_new = np.zeros((ngrid//(2*stride), aux.shape[0]))*np.nan
    mean, std = np.array(scaler[0]), np.array(scaler[1])
    x_temp = x[i*stride:i*stride+cfg["seq_len"],:,:][:,:,0:ngrid:2*stride]
    x_new = np.transpose(x_temp, (2,0,1))
    y_new = y[i*stride+cfg["seq_len"]+cfg["forcast_time"],0:ngrid:2*stride]
    aux_new = aux[:,0:ngrid:2*stride]
    aux_new = np.transpose(aux_new, (1,0))

    return x_new, y_new, aux_new, np.tile(mean, (1, n, 1)), np.tile(std, (1,n,1))
'''
# ------------------------------------------------------------------------------------------------------------------------------              
def load_train_data_for_cnn(cfg, x, y, aux, scaler,lat_index,lon_index, mask):
    nt, nf, nlat, nlon = x.shape

    ngrid = nlat * nlon
    mean, std = np.array(scaler[0]), np.array(scaler[1])
    mask_index = np.random.randint(0, mask.shape[0], cfg['batch_size'])
    idx_grid = mask[mask_index]
    # ngrid convert nlat and nlon
    idx_lon = ((idx_grid+1) % (nlon+1))-1   
    idx_lon[idx_lon==-1] = nlon-1
    idx_lat =  (idx_grid//(nlon+1))

    idx_time = np.random.randint(0, nt-cfg['seq_len']-cfg["forcast_time"], 1)[0]

    x_new = np.zeros((idx_lon.shape[0], cfg["seq_len"], nf, 2*cfg['spatial_offset']+1, 2*cfg['spatial_offset']+1))*np.nan
    y_new = np.zeros((idx_lon.shape[0]))*np.nan
    aux_new = np.zeros((idx_lon.shape[0], aux.shape[0], 2*cfg['spatial_offset']+1, 2*cfg['spatial_offset']+1))*np.nan

    for i in range (idx_lon.shape[0]):
        lat_index_bias = idx_lat[i] + cfg['spatial_offset']
        lon_index_bias = idx_lon[i] + cfg['spatial_offset']
        x_new[i] = x[idx_time:idx_time+cfg['seq_len'],:,lat_index[lat_index_bias-cfg['spatial_offset']:lat_index_bias+cfg['spatial_offset']+1],:][:,:,:,lon_index[lon_index_bias-cfg['spatial_offset']:lon_index_bias+cfg['spatial_offset']+1]]
        y_new[i] = y[idx_time+cfg['seq_len']+cfg["forcast_time"],:,idx_lat[i], idx_lon[i]] ##
        aux_new[i] = aux[:,lat_index[lat_index_bias-cfg['spatial_offset']:lat_index_bias+cfg['spatial_offset']+1],:][:,:,lon_index[lon_index_bias-cfg['spatial_offset']:lon_index_bias+cfg['spatial_offset']+1]]

    y_new[np.isinf(y_new)]=np.nan
    mask = y_new == y_new
    x_new = x_new[mask]
    y_new = y_new[mask]
    aux_new = aux_new[mask]
    x_new = np.nan_to_num(x_new)
    aux_new = np.nan_to_num(aux_new)
    return x_new, y_new, aux_new, mean, std

def load_test_data_for_cnn(cfg, x, y, aux, scaler, slect_list,lat_index,lon_index, z, stride):
    x = x.transpose(0,3,1,2)
    y = y.transpose(0,3,1,2)
    aux = aux.transpose(2,0,1)
    nt, _, nlat, nlon = y.shape
    ny = (2*nlat//stride)+1
    nx = (2*nlon//stride)+1

    x_new = np.zeros((ny*nx, cfg["seq_len"], x.shape[1], 2*cfg['spatial_offset']+1, 2*cfg['spatial_offset']+1))*np.nan
    y_new = np.zeros((ny*nx))*np.nan
    aux_new = np.zeros((ny*nx, aux.shape[0], 2*cfg['spatial_offset']+1, 2*cfg['spatial_offset']+1))*np.nan
    mean, std = np.array(scaler[0]), np.array(scaler[1])

    count = 0
    for i in range (0, nlon, stride//2):
        for j in range(0, nlat,stride//2):
                lat_index_bias = lat_index[j] + cfg['spatial_offset']
                lon_index_bias = lon_index[i] + cfg['spatial_offset']
                x_new[count] = x[z:z+cfg['seq_len'],:,lat_index[lat_index_bias-cfg['spatial_offset']:lat_index_bias+cfg['spatial_offset']+1],:][:,:,:,lon_index[lon_index_bias-cfg['spatial_offset']:lon_index_bias+cfg['spatial_offset']+1]]
                y_new[count] = y[z+cfg['seq_len']+cfg["forcast_time"],:,j, i] ##
                aux_new[count] = aux[:,lat_index[lat_index_bias-cfg['spatial_offset']:lat_index_bias+cfg['spatial_offset']+1],:][:,:,lon_index[lon_index_bias-cfg['spatial_offset']:lon_index_bias+cfg['spatial_offset']+1]]
                count =count+1
    y_new[np.isinf(y_new)]=np.nan
    mask = y_new == y_new
    x_new = x_new[mask]
    y_new = y_new[mask]
    aux_new = aux_new[mask]   
    x_new = np.nan_to_num(x_new)
    aux_new = np.nan_to_num(aux_new)

    return x_new, y_new, aux_new, np.tile(mean, (1, ny*nx, 1)), np.tile(std, (1,ny*nx,1))
# ------------------------------------------------------------------------------------------------------------------------------              
def load_train_data_for_co(cfg, x, y, aux, scaler):
    nt, _, nlat, nlon = y.shape
    print('y.shape is',y)
    ngrid = nlat * nlon
    mean, std = np.array(scaler[0]), np.array(scaler[1])
    idx_grid = np.random.randint(0, ngrid, cfg['batch_size'])
    # ngrid convert nlat and nlon
    idx_lon = ((idx_grid+1) % (nlon+1))-1   
    idx_lon[idx_lon==-1] = nlon
    idx_lat =  (idx_grid//(nlon+1))

    idx_time = np.random.randint(0, nt-cfg['seq_len'], 1)[0]

    x_new = np.zeros((idx_lon.shape[0], cfg["seq_len"]+1, x.shape[1], 2*cfg['spatial_offset'], 2*cfg['spatial_offset']))*np.nan
    y_new = np.zeros((idx_lon.shape[0]))*np.nan
    aux_new = np.zeros((idx_lon.shape[0], aux.shape[0], 2*cfg['spatial_offset'], 2*cfg['spatial_offset']))*np.nan

    for i in range (idx_lon.shape[0]):
        idx_lat_bias, idx_lon_bias = idx_lat[i]+cfg['spatial_offset'],idx_lon[i]+cfg['spatial_offset']
        x_new[i] = x[idx_time:idx_time+cfg['seq_len']+1,:,
                        idx_lat_bias-cfg['spatial_offset']:idx_lat_bias+cfg['spatial_offset'],
                        idx_lon_bias-cfg['spatial_offset']:idx_lon_bias+cfg['spatial_offset']]
        y_new[i] = y[idx_time+cfg['seq_len']+cfg["forcast_time"],idx_lat[i], idx_lon[i]] ##
        aux_new[i] = aux[:,idx_lat_bias-cfg['spatial_offset']:idx_lat_bias+cfg['spatial_offset'],
                        idx_lon_bias-cfg['spatial_offset']:idx_lon_bias+cfg['spatial_offset']]
    mask = y_new == y_new
    x_new = x_new[mask]
    y_new = y_new[mask]
    aux_new = aux_new[mask]

    return x_new, y_new, aux_new, mean, std

# ------------------------------------------------------------------------------------------------------------------------------              
def erath_data_transform(cfg, x):
    lat_index = np.array([i for i in range(0,x.shape[1])])
    lon_index = np.array([i for i in range(0,x.shape[2])])

    x_up = lat_index[lat_index.shape[0]-cfg['spatial_offset']:lat_index.shape[0]]
    x_down = lat_index[:cfg['spatial_offset']]
    x_left = lon_index[lon_index.shape[0]-cfg['spatial_offset']:lon_index.shape[0]]
    x_right = lon_index[:cfg['spatial_offset']]
    lat_index_new = np.concatenate((x_up,lat_index),axis=0)
    lat_index_new = np.concatenate((lat_index_new,x_down),axis=0)
    lon_index_new = np.concatenate((x_left,lon_index),axis=0)
    lon_index_new = np.concatenate((lon_index_new,x_right),axis=0)
    return lat_index_new,lon_index_new
