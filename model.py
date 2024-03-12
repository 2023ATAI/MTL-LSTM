"""
All model structure

<<<<<<<< HEAD
Author: Qingliang Li 
12/23/2022 - V1.0  LSTM, CNN, ConvLSTM edited by Qingliang Li
.........  - V2.0
.........  - V3.0
"""

import torch
import torch.nn as nn
import numpy as np
from convlstm import ConvLSTM

#7月9  由於一直沒有緊張 最好的效果也就是lg_ve_3optim-SD   適度能到0.6多  但是蒸散發下降的也多  所以打算重新寫一個軟共享 用L2約束學習兩個模型的LSTM曾
# 目前暫定是  對數似然加L2約束的lstm曾參數 作爲損失函數
class SoftMTLv1(nn.Module):
    def __init__(self, cfg, softmtl_cfg):
        super(SoftMTLv1, self).__init__()
        self.lstm = nn.LSTM(softmtl_cfg["input_size"], softmtl_cfg["hidden_size"], batch_first=True)
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        self.dense = nn.Linear(softmtl_cfg["hidden_size"], softmtl_cfg["out_size"])

    def forward(self, inputs, aux):
        inputs_new = inputs
        x, _ = self.lstm(inputs_new.float())
        x = self.drop(x)
        # we only predict the last step
        x = self.dense(x[:,-1,:])
        return x

class LSTMModel(nn.Module):
    """single task model"""

    def __init__(self, cfg,lstmmodel_cfg):
        super(LSTMModel,self).__init__()
        self.lstm = nn.LSTM(lstmmodel_cfg["input_size"], lstmmodel_cfg["hidden_size"],batch_first=True)
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        self.dense = nn.Linear(lstmmodel_cfg["hidden_size"],lstmmodel_cfg["out_size"])

    def forward(self, inputs,aux):
        inputs_new = inputs
        x, _ = self.lstm(inputs_new.float())
        x = self.drop(x)
        # we only predict the last step
        x = self.dense(x[:,-1,:])
        return x

class SoftMTLv2(nn.Module):
    def __init__(self, cfg, softmtl_cfg):
        super(SoftMTLv2, self).__init__()
        self.lstm = nn.LSTM(softmtl_cfg["input_size"], softmtl_cfg["hidden_size"], batch_first=True)
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        self.dense = nn.Linear(softmtl_cfg["hidden_size"], softmtl_cfg["out_size"])

    def forward(self, inputs, aux):
        inputs_new = inputs
        x, _ = self.lstm(inputs_new.float())
        x = self.drop(x)
        # we only predict the last step
        x = self.dense(x[:,-1,:])
        return x

class SoftMTLv3(nn.Module):
    def __init__(self, cfg, softmtl_cfg):
        super(SoftMTLv3, self).__init__()
        self.lstm = nn.LSTM(softmtl_cfg["input_size"], softmtl_cfg["hidden_size"], batch_first=True)
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        self.dense = nn.Linear(softmtl_cfg["hidden_size"], softmtl_cfg["out_size"])

    def forward(self, inputs, aux):
        inputs_new = inputs
        x, _ = self.lstm(inputs_new.float())
        x = self.drop(x)
        # we only predict the last step
        x = self.dense(x[:,-1,:])
        return x

# 硬共享
class MSLSTMModel(nn.Module):
    """double task model"""

    def __init__(self, cfg, mtllstmmodel_cfg):
        super(MSLSTMModel, self).__init__()

        self.head_layers = nn.ModuleList()
        self.lstm1 = nn.LSTM(mtllstmmodel_cfg["input_size"], mtllstmmodel_cfg["hidden_size"], batch_first=True)
        # self.drop = nn.Dropout(p=cfg["dropout_rate"])
        for i in range(cfg['num_repeat']):
            self.head_layers.append(nn.Linear(mtllstmmodel_cfg["hidden_size"], mtllstmmodel_cfg["out_size"]))

    # 多個數據輸入
    def forward(self, inputs,aux):
        pred = []
        for i in range(len(self.head_layers)):
            x, _ = self.lstm1(inputs[i].float())
            pred.append(self.head_layers[i](x[:,-1:]))
        return pred

# one exper
class MMOE(nn.Module):
    def __init__(self, cfg, MMOE_cfg):
        super(MMOE, self).__init__()
        self.expert0 = nn.LSTM(MMOE_cfg["input_size"], MMOE_cfg["hidden_size"], batch_first=True)
            # nn.Dropout(p=cfg["dropout_rate"])


        self.gate0 = nn.Sequential(
            nn.Linear(MMOE_cfg["input_size"], 1),
            nn.Softmax(dim=1)
        )
        self.gate1 = nn.Sequential(
            nn.Linear(MMOE_cfg["input_size"], 1),
            nn.Softmax(dim=1)
        )

        self.Gates = nn.ModuleList([
            self.gate0,self.gate1
        ])


        self.tower0 = nn.Linear(MMOE_cfg["hidden_size"], MMOE_cfg["out_size"])


        self.tower1 = nn.Linear(MMOE_cfg["hidden_size"], MMOE_cfg["out_size"])

        self.Towers = nn.ModuleList([
            self.tower0,self.tower1
        ])

    def forward(self, inputs_, aux):

        gate_weights = []
        task_outputs = []
        combined_outputs = []
        for j,inputs in enumerate(inputs_):
            expert_output, _ = self.expert0(inputs.float())


            gate_model = self.Gates[j]
            gate_weight = gate_model(inputs.float())

                # 原本是128  7 3  和  128  7 128  不能相乘  转置后变成128 3 7 就可以相乘了
            combined_output = torch.matmul(gate_weight.transpose(1,2), expert_output)
            combined_outputs.append(combined_output)

            # task_outputs = [Towers(combined_output) for Towers in self.Towers]
        for i,tower_model in enumerate(self.Towers):
            output = tower_model(combined_outputs[i][:,-1:])
            task_outputs.append(output)

        return task_outputs
# 多專家
class aMMOE(nn.Module):
    def __init__(self, cfg, MMOE_cfg):
        super(MMOE, self).__init__()
        self.expert0 = nn.Sequential(
            nn.LSTM(MMOE_cfg["input_size"], MMOE_cfg["hidden_size"], batch_first=True),
            # nn.Dropout(p=cfg["dropout_rate"])
        )
        self.expert1 = nn.Sequential(
            nn.LSTM(MMOE_cfg["input_size"], MMOE_cfg["hidden_size"], batch_first=True),
            # nn.Dropout(p=cfg["dropout_rate"])
        )
        self.expert2 = nn.Sequential(
            nn.LSTM(MMOE_cfg["input_size"], MMOE_cfg["hidden_size"], batch_first=True),
            # nn.Dropout(p=cfg["dropout_rate"])
        )
        self.gate0 = nn.Sequential(
            nn.Linear(MMOE_cfg["input_size"], 1),
            nn.Softmax(dim=1)
        )
        self.gate1 = nn.Sequential(
            nn.Linear(MMOE_cfg["input_size"], 1),
            nn.Softmax(dim=1)
        )

        self.Expers = nn.ModuleList([
            self.expert0,self.expert1,self.expert2
        ])

        self.Gates = nn.ModuleList([
            self.gate0,self.gate1
        ])

        self.tower0 = nn.Sequential(
            nn.Dropout(p=cfg["dropout_rate"]),
            nn.Linear(MMOE_cfg["hidden_size"], MMOE_cfg["out_size"])

        )
        self.tower1 = nn.Sequential(
            nn.Dropout(p=cfg["dropout_rate"]),
            nn.Linear(MMOE_cfg["hidden_size"], MMOE_cfg["out_size"])

        )

        self.Towers = nn.ModuleList([
            self.tower0,self.tower1
        ])

    def forward(self, inputs_, aux):

        gate_weights = []
        task_outputs = []
        combined_outputs = []
        for j,inputs in enumerate(inputs_):
            expert_outputs = []
            for expert_model in self.Expers:
                output,_ = expert_model(inputs.float())
                expert_outputs.append(output)

            gate_model = self.Gates[j]
            gate_weight = gate_model(inputs.float())

            combined_output = 0
            for expert_output in expert_outputs:
                # 原本是128  7 3  和  128  7 128  不能相乘  转置后变成128 3 7 就可以相乘了
                combined_output += torch.matmul(gate_weight.transpose(1,2), expert_output)
            combined_outputs.append(combined_output)

            # task_outputs = [Towers(combined_output) for Towers in self.Towers]
        # for i,tower_model in enumerate(self.Towers):
            output = self.Towers[j](combined_outputs[j][:,-1:])
            task_outputs.append(output)

        return task_outputs

