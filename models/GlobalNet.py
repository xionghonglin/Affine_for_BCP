import torch
import torch.nn as nn
import torch.nn.functional as F
from models.linear import Linear
from .model_parts import *

class GlobalNet(nn.Module):
    def __init__(self):
        super(GlobalNet, self).__init__() 
        self.affine_linear = Linear(in_dim=512, out_dim=3, hidden_list = [256,256])
        self.MaxPool = nn.MaxPool3d(kernel_size=2)
        self.month_mapping = Linear(in_dim=2, out_dim=256, hidden_list = [256,256])
        self.final_conv = Linear(in_dim=4032, out_dim=256,  hidden_list = [256])
       
   
        self.m_1 = nn.Sequential(*[default_conv(1,64,3, bias=True)])
        self.m_2 = nn.Sequential(*[ResBlock(64,3) , ResBlock(64,3), default_conv(64,32,3, bias=True)]) #64
        self.m_3 = nn.Sequential(*[ResBlock(32,3) , ResBlock(32,3), default_conv(32,32,3, bias=True)])#32
        self.m_4 = nn.Sequential(*[ResBlock(32,3) , ResBlock(32,3), default_conv(32,32,3, bias=True)]) #32
        self.m_5 = nn.Sequential(*[ResBlock(32,3) , ResBlock(32,3), default_conv(32,16,3, bias=True)])  #32
        self.m_6 = nn.Sequential(*[ResBlock(16,3) , ResBlock(16,3), default_conv(16,16,3, bias=True)])  #16
        self.m_7 = nn.Sequential(*[ResBlock(16,3) , ResBlock(16,3), default_conv(16,16,3, bias=True)])   #16                                                            #16

    def forward(self, x, mon1, mon2):
        x = self.MaxPool(x)
        x = self.m_1(x)
        x = self.m_2(x)
        x = self.MaxPool(x)
        x = self.m_3(x)
        x = self.MaxPool(x)
        x = self.m_4(x)
        x = self.MaxPool(x)
        x = self.m_5(x)
        x = self.MaxPool(x)
        x = self.m_6(x)
        x = x.reshape(x.shape[0],-1)
        x = self.final_conv(x)

        mo = torch.cat((mon1,mon2),dim=1)
        mo_mapping = self.month_mapping(mo)

        x = torch.cat((x,mo_mapping),dim=1)
        affine = self.affine_linear(x)
        return affine

