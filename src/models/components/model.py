'''
Author: 
Date: 2024-03-07 17:05:33
LastEditors: 
LastEditTime: 2024-06-04 10:04:34
Description: 
'''
import torch
from torch import nn

class Activation(nn.Module):
    """Activation
    """
    def __init__(self, act_type, **params):
        super(Activation, self).__init__()
        if act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'leaky_relu':
            self.act = nn.LeakyReLU(**params)
        elif act_type == 'silu':
            self.act = nn.SiLU()
        else:
            raise ValueError(act_type)
     
    def forward(self, x):
        return self.act(x)
    
class MLP(nn.Module):
    """MLP
    """
    def __init__(self, layer_num, in_size, hidden_size, out_size, act, dropout_rate = 0.1):
        super(MLP, self).__init__()

        layers = []
        for layer_id in range(layer_num):
            if layer_id == 0:
                layers.append(nn.Linear(in_size, hidden_size))
                layers.append(nn.Dropout(dropout_rate))
                layers.append(Activation(act))
                
            elif layer_id < layer_num - 1:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.Dropout(dropout_rate))
                layers.append(Activation(act))
            
            else:
                layers.append(nn.Linear(hidden_size, out_size))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x(tensor): (-1, dim).
        """
        return self.mlp(x)
    
    def pre_reduce(self, x, v, z, pos, batch):
        return self(x) # + v.sum() * 0
    
    def post_reduce(self, x):
        return x
    
class MLPModel(nn.Module):
    def __init__(
                self, 
                task_type:str,
                num_tasks:int,
                encoder:nn.Module,
                hidden_channels:int,
                layer_num:int,
                hidden_size:int = -1,
                dropout_rate:float = 0.2,
                act:str = 'leaky_relu',
                reduce_op: str = 'add',
                mean: torch.tensor = torch.tensor([0.0]),  # shape(1,num_tasks)
                std: torch.tensor = torch.tensor([1.0]),  # shape(1,num_tasks)
                ):
        
        super(MLPModel,self).__init__()
        self.task_type = task_type
        self.num_tasks = num_tasks
        self.encoder = encoder
        self.hidden_channels = hidden_channels
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.act = act
        self.reduce_op = reduce_op
        self.mean = mean
        self.std = std
        
        if self.encoder is not None:
            self.hidden_channels = self.encoder.hidden_channels
        # self.hidden_size = self.hidden_size if self.hidden_size>0 else 4*self.hidden_channels
        self.hidden_size = self.hidden_size if self.hidden_size>0 else self.hidden_channels
        
        self.norm = nn.LayerNorm(self.hidden_channels)
        self.mlp = MLP(
                self.layer_num,
                in_size = self.hidden_channels,
                hidden_size = int(self.hidden_size/2),
                out_size = self.num_tasks,
                act = self.act,
                dropout_rate = self.dropout_rate)
        if self.task_type == 'classification':
            self.out_act = nn.Sigmoid()
            
        
    def forward(self,z,pos,batch_id, charges = None):
        bs = batch_id[-1].item()+1
        x = z.view(bs, -1)
        x = self.norm(x)
        pred = self.mlp(x)
        if self.task_type == 'classification':
            pred = self.out_act(pred)
        else:
            pred = pred*self.std.to(pred.device) + self.mean.to(pred.device)
        return pred, None
