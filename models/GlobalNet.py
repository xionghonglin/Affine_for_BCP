import torch
import torch.nn as nn
import torch.nn.functional as F
from models.linear import Linear
from .model_parts import *

class GlobalNet(nn.Module):
    def __init__(self):
        super(GlobalNet, self).__init__() 
        self.affine_linear = Linear(in_dim=4288, out_dim=3, hidden_list = [256,256,256])
        self.MaxPool = nn.MaxPool3d(kernel_size=2)
        self.affine_transform = DirectAffineTransform()
        self.month_mapping = Linear(in_dim=2, out_dim=256, hidden_list = [256,256,256])
        #self.month_tgt_mapping = Linear(in_dim=2, out_dim=256, hidden_list = [256,256,256])
   
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

        b = x.shape[0]
        x = x.reshape(b,-1)
        mo = torch.cat((mon1,mon2),dim=1)
        mo_mapping = self.month_mapping(mo)
        x = torch.cat((x,mo_mapping),dim=1)
        affine = self.affine_linear(x)
        x = self.affine_transform(src, affine)[0]
        return x

class DirectAffineTransform(nn.Module):
    def __init__(self):
        super(DirectAffineTransform, self).__init__()

        self.id = None
        self.scaling_m = None

    def forward(self, x, affine_para):
        b, c, h, w, d = x.shape

        self.id = torch.zeros((b, 3, 4)).cuda()
        self.id[:, 0, 0] = 1
        self.id[:, 1, 1] = 1
        self.id[:, 2, 2] = 1
        # Matrix that register x to its center of mass
        id_grid = F.affine_grid(self.id, x.shape, align_corners=True)

        to_center_matrix = torch.zeros((b, 4, 4)).cuda()
        reversed_to_center_matrix = torch.zeros((b, 4, 4)).cuda()

        x_sum = torch.sum(x)
        center_mass_x = torch.sum(x.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 0]) / x_sum
        center_mass_y = torch.sum(x.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 1]) / x_sum
        center_mass_z = torch.sum(x.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 2]) / x_sum

        to_center_matrix[:, 0, 3] = center_mass_x
        to_center_matrix[:, 1, 3] = center_mass_y
        to_center_matrix[:, 2, 3] = center_mass_z
        to_center_matrix[:, 0, 0] = 1
        to_center_matrix[:, 1, 1] = 1
        to_center_matrix[:, 2, 2] = 1
        to_center_matrix[:, 3, 3] = 1
        reversed_to_center_matrix[:, 0, 3] = -center_mass_x
        reversed_to_center_matrix[:, 1, 3] = -center_mass_y
        reversed_to_center_matrix[:, 2, 3] = -center_mass_z
        reversed_to_center_matrix[:, 0, 0] = 1
        reversed_to_center_matrix[:, 1, 1] = 1
        reversed_to_center_matrix[:, 2, 2] = 1
        reversed_to_center_matrix[:, 3, 3] = 1


        
        self.scaling_m = torch.zeros((b, 4, 4)).cuda()#torch.eye(4).cuda()
        scaling_xyz = 1 + (affine_para[:, 0:3] * 0.5)
        self.scaling_m[:, 0, 0] = scaling_xyz[:, 0]
        self.scaling_m[:, 1, 1] = scaling_xyz[:, 1]
        self.scaling_m[:, 2, 2] = scaling_xyz[:, 2]
        self.scaling_m[:, 3, 3] = 1

        affine_matrix = torch.matmul(to_center_matrix, torch.matmul(self.scaling_m, reversed_to_center_matrix))
        #print(affine_matrix.shape)
        affine_matrix = affine_matrix[:, 0:3, :]
        #print(affine_matrix.shape)

        grid = F.affine_grid(affine_matrix, x.shape, align_corners=True)
        transformed_x = F.grid_sample(x, grid, mode='bilinear', align_corners=True)

        return transformed_x, affine_matrix
    
class Center_of_mass_initial_pairwise(nn.Module):
    def __init__(self):
        super(Center_of_mass_initial_pairwise, self).__init__()
        self.id = None
        self.to_center_matrix = None
        
        # self.id = torch.zeros((1, 3, 4)).cuda()
        # self.id[0, 0, 0] = 1
        # self.id[0, 1, 1] = 1
        # self.id[0, 2, 2] = 1

        # self.to_center_matrix = torch.zeros((1, 3, 4)).cuda()
        # self.to_center_matrix[0, 0, 0] = 1
        # self.to_center_matrix[0, 1, 1] = 1
        # self.to_center_matrix[0, 2, 2] = 1

    def forward(self, x, y):

        b, c, h, w, d = x.shape
        self.id = torch.zeros((b, 3, 4)).cuda()
        self.id[:, 0, 0] = 1
        self.id[:, 1, 1] = 1
        self.id[:, 2, 2] = 1

        self.to_center_matrix = torch.zeros((b, 3, 4)).cuda()
        self.to_center_matrix[:, 0, 0] = 1
        self.to_center_matrix[:, 1, 1] = 1
        self.to_center_matrix[:, 2, 2] = 1


        # center of mass of x -> center of mass of y
        id_grid = F.affine_grid(self.id, x.shape, align_corners=True)
        # mask = (x > 0).float()
        # mask_sum = torch.sum(mask)
        x_sum = torch.sum(x)
        x_center_mass_x = torch.sum(x.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 0])/x_sum
        x_center_mass_y = torch.sum(x.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 1])/x_sum
        x_center_mass_z = torch.sum(x.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 2])/x_sum

        y_sum = torch.sum(y)
        y_center_mass_x = torch.sum(y.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 0]) / y_sum
        y_center_mass_y = torch.sum(y.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 1]) / y_sum
        y_center_mass_z = torch.sum(y.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 2]) / y_sum

        self.to_center_matrix[:, 0, 3] = x_center_mass_x - y_center_mass_x
        self.to_center_matrix[:, 1, 3] = x_center_mass_y - y_center_mass_y
        self.to_center_matrix[:, 2, 3] = x_center_mass_z - y_center_mass_z

        grid = F.affine_grid(self.to_center_matrix, x.shape, align_corners=True)
        transformed_image = F.grid_sample(x, grid, align_corners=True)

        # print(affine_para)
        # print(output_affine_m[0:3])

        return transformed_image, grid

# class DirectAffineTransform(nn.Module):
#     def __init__(self):
#         super(DirectAffineTransform, self).__init__()

#         self.id = torch.zeros(( 3, 4)).cuda()
#         self.id[0, 0] = 1
#         self.id[1, 1] = 1
#         self.id[2, 2] = 1


#         self.scaling_m = None

#     def forward(self, x, affine_para):

#         self.scaling_m = torch.eye(4).cuda()
#         scaling_xyz = 1 + (affine_para[0, 0:3] * 0.5)
#         self.scaling_m[0, 0] = scaling_xyz[0]
#         self.scaling_m[1, 1] = scaling_xyz[1]
#         self.scaling_m[2, 2] = scaling_xyz[2]
#         affine_matrix = self.scaling_m[0:3].unsqueeze(0)# self.id
    

#         grid = F.affine_grid(affine_matrix, x.shape, align_corners=True)
#         transformed_x = F.grid_sample(x, grid, mode='bilinear', align_corners=True)

#         return transformed_x, affine_matrix