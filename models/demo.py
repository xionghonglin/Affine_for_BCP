import torch
import torch.nn as nn
import torch.nn.functional as F
from linear import Linear
from model_parts import *
import torch.nn.init as init


class GlobalNet(nn.Module):
    def __init__(self):
        super(GlobalNet, self).__init__() 
        self.affine_linear = Linear(in_dim=4034, out_dim=12, hidden_list = [256,256,256])
        self.MaxPool = nn.MaxPool3d(kernel_size=2)
        self.affine_transform = DirectAffineTransform()
        # self.month_src_mapping = Linear(in_dim=256, out_dim=12, hidden_list = [256,256,256])
        # self.month_tgt_mapping = Linear(in_dim=256, out_dim=12, hidden_list = [256,256,256])
   
        self.m_1 = nn.Sequential(*[default_conv(1,64,3, bias=True)])
        self.m_2 = nn.Sequential(*[ResBlock(64,3) , ResBlock(64,3), default_conv(64,32,3, bias=True)]) #64
        self.m_3 = nn.Sequential(*[ResBlock(32,3) , ResBlock(32,3), default_conv(32,32,3, bias=True)])#32
        self.m_4 = nn.Sequential(*[ResBlock(32,3) , ResBlock(32,3), default_conv(32,32,3, bias=True)]) #32
        self.m_5 = nn.Sequential(*[ResBlock(32,3) , ResBlock(32,3), default_conv(32,16,3, bias=True)])  #32
        self.m_6 = nn.Sequential(*[ResBlock(16,3) , ResBlock(16,3), default_conv(16,16,3, bias=True)])  #16
        self.m_7 = nn.Sequential(*[ResBlock(16,3) , ResBlock(16,3), default_conv(16,16,3, bias=True)])   #16                                                            #16

    def forward(self, x, mon1, mon2):
        src = x
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
        # x = self.MaxPool(x)
        # x = self.m_7(x)
        b = x.shape[0]
        x = x.reshape(b,-1)
        x = torch.cat((x,mon1),dim=1)
        x = torch.cat((x,mon2),dim=1)
        affine = self.affine_linear(x)
        x = self.affine_transform(src, affine)
        return x


class DirectAffineTransform(nn.Module):
    def __init__(self):
        super(DirectAffineTransform, self).__init__()

        self.id = torch.zeros(( 3, 4)).cuda()
        self.id[ 0, 0] = 1
        self.id[ 1, 1] = 1
        self.id[ 2, 2] = 1

    def forward(self, x, affine_para):
        affine_matrix = affine_para.reshape(-1, 3, 4) + self.id

        grid = F.affine_grid(affine_matrix, x.shape, align_corners=True)
        transformed_x = F.grid_sample(x, grid, mode='bilinear', align_corners=True)

        return transformed_x, affine_matrix

Globa = GlobalNet().cuda()
b = 2
x = torch.ones(b,1,192,224,192).cuda()
mon1 = torch.ones(b,1).cuda()
mon2 = torch.ones(b,1).cuda()

for i in range(10):
    pred = Globa(x,mon1,mon2)
    
    print(pred)
    pred = None