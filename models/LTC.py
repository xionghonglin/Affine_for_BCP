import torch
import torch.nn as nn
import torch.nn.functional as F
from models.linear import Linear
from .model_parts import *


class LTC(nn.Module):
    def __init__(self, in_channel, out_channel, trilinear=False):
        super(LTC, self).__init__()
        self.in_channels = in_channel
        self.out_channels = out_channel
        self.trilinear = trilinear
        self.inc = (DoubleConv(self.in_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
       
        self.trans = (TransConv(512, 256))
        self.up1 = (Up(256, 128,trilinear))
        self.up2 = (Up(128, 64,trilinear))
        self.up3 = (Up(64, 32,trilinear))
        self.outc = (OutConv(32, out_channel)) 
        dim_num = 256
        self.linear_4 = Linear(in_dim=1, out_dim=256, hidden_list = [dim_num,dim_num,dim_num])
        self.linear_3 = Linear(in_dim=1, out_dim=128, hidden_list = [dim_num,dim_num,dim_num])
        self.linear_2 = Linear(in_dim=1, out_dim=64, hidden_list = [dim_num,dim_num,dim_num])
        self.linear_1 = Linear(in_dim=1, out_dim=32, hidden_list = [dim_num,dim_num,dim_num])


    def time_mapping(self, m1, m2):
        # b, c, h, w, d = feat.shape
        self.mapping_1_src = self.linear_1(m1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.mapping_2_src = self.linear_2(m1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.mapping_3_src = self.linear_3(m1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.mapping_4_src = self.linear_4(m1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # self.mapping_1_tgt = self.linear_1(m2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # self.mapping_2_tgt = self.linear_2(m2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # self.mapping_3_tgt = self.linear_3(m2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.mapping_4_tgt = self.linear_4(m2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return self.mapping_1_src, self.mapping_2_src, self.mapping_3_src, self.mapping_1_src, \
           self.mapping_4_tgt
        

    def forward(self, x, mon1, mon2):
        self.time_mapping(mon1, mon2)
        x = self.inc(x)
        c1 = x
        x = self.down1(x)
        c2 = x
        x = self.down2(x)
        c3 = x
        x = self.down3(x)
        x = x
        c4 = x
        x = x*self.mapping_4_tgt
        
        x = self.trans(x,c4)
        x = self.up1(x,c3)
        x = self.up2(x,c2)
        x = self.up3(x,c1)
        x = self.outc(x)
        return x

