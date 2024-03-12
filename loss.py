import torch.nn
import torch

class NaNMSELoss1():
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["modelname"]

    def fit(self, y_pred,y_true,lossmse):
        mask = y_true == y_true
        y_true = y_true[mask]
        y_pred = torch.squeeze(y_pred[mask])
        loss = torch.sqrt(lossmse(y_true, y_pred))
        return loss

class NaNMSELoss():
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["modelname"]

    def fit(self, y_pred,y_true,lossmse):
        mask = y_true == y_true
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if not isinstance(y_pred, torch.Tensor):
            y_true = torch.tensor(y_true)
            y_pred = torch.tensor(y_pred)
        # print('yp',y_pred.shape)
        # print('yt',y_true.shape)
        # print('lossme',lossmse)
        loss = torch.sqrt(lossmse(y_true[:len(y_pred)], y_pred))
        return loss




    
