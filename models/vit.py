import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import torch.nn.functional as F

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class Linear(nn.Module):
    def __init__(self, in_dim=0, out_dim=0, hidden_list = []):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            #layers.append(nn.Dropout(p=drop))
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.time_embedding = Linear(in_dim=2, out_dim=1024,  hidden_list = [256,512])
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.affine_transform = AffineCOMTransform()

    def forward(self, video, time):
        src = video
        x = self.to_patch_embedding(video)
        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        
        time_tokens = self.time_embedding(time).unsqueeze(1)
  
        x = torch.cat((time_tokens, x), dim=1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 2)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        affine = self.mlp_head(x)
        affined = self.affine_transform(src, affine)[0]
        return affined
    
class AffineCOMTransform(nn.Module):
    def __init__(self, use_com=True):
        super(AffineCOMTransform, self).__init__()

        self.translation_m = None
        self.rotation_x = None
        self.rotation_y = None
        self.rotation_z = None
        self.rotation_m = None
        self.shearing_m = None
        self.scaling_m = None

        self.id = torch.zeros((1, 3, 4)).cuda()
        self.id[0, 0, 0] = 1
        self.id[0, 1, 1] = 1
        self.id[0, 2, 2] = 1

        self.use_com = use_com

    def forward(self, x, affine_para):
        # print("x:",x.shape)
        # print("affine_para:",affine_para.shape)
        # Matrix that register x to its center of mass
        id_grid = F.affine_grid(self.id, x.shape, align_corners=True)

        to_center_matrix = torch.eye(4).cuda()
        reversed_to_center_matrix = torch.eye(4).cuda()
        if self.use_com:
            x_sum = torch.sum(x)
            center_mass_x = torch.sum(x.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 0]) / x_sum
            center_mass_y = torch.sum(x.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 1]) / x_sum
            center_mass_z = torch.sum(x.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 2]) / x_sum

            to_center_matrix[0, 3] = center_mass_x
            to_center_matrix[1, 3] = center_mass_y
            to_center_matrix[2, 3] = center_mass_z
            reversed_to_center_matrix[0, 3] = -center_mass_x
            reversed_to_center_matrix[1, 3] = -center_mass_y
            reversed_to_center_matrix[2, 3] = -center_mass_z

        self.translation_m = torch.eye(4).cuda()
        self.rotation_x = torch.eye(4).cuda()
        self.rotation_y = torch.eye(4).cuda()
        self.rotation_z = torch.eye(4).cuda()
        self.rotation_m = torch.eye(4).cuda()
        self.shearing_m = torch.eye(4).cuda()
        self.scaling_m = torch.eye(4).cuda()

        trans_xyz = affine_para[0, 0:3]
        rotate_xyz = affine_para[0, 3:6] * math.pi
        shearing_xyz = affine_para[0, 6:9] * math.pi
        scaling_xyz = 1 + (affine_para[0, 9:12] * 0.5)

        self.translation_m[0, 3] = trans_xyz[0]
        self.translation_m[1, 3] = trans_xyz[1]
        self.translation_m[2, 3] = trans_xyz[2]
        self.scaling_m[0, 0] = scaling_xyz[0]
        self.scaling_m[1, 1] = scaling_xyz[1]
        self.scaling_m[2, 2] = scaling_xyz[2]

        self.rotation_x[1, 1] = torch.cos(rotate_xyz[0])
        self.rotation_x[1, 2] = -torch.sin(rotate_xyz[0])
        self.rotation_x[2, 1] = torch.sin(rotate_xyz[0])
        self.rotation_x[2, 2] = torch.cos(rotate_xyz[0])

        self.rotation_y[0, 0] = torch.cos(rotate_xyz[1])
        self.rotation_y[0, 2] = torch.sin(rotate_xyz[1])
        self.rotation_y[2, 0] = -torch.sin(rotate_xyz[1])
        self.rotation_y[2, 2] = torch.cos(rotate_xyz[1])

        self.rotation_z[0, 0] = torch.cos(rotate_xyz[2])
        self.rotation_z[0, 1] = -torch.sin(rotate_xyz[2])
        self.rotation_z[1, 0] = torch.sin(rotate_xyz[2])
        self.rotation_z[1, 1] = torch.cos(rotate_xyz[2])

        self.rotation_m = torch.mm(torch.mm(self.rotation_z, self.rotation_y), self.rotation_x)

        self.shearing_m[0, 1] = shearing_xyz[0]
        self.shearing_m[0, 2] = shearing_xyz[1]
        self.shearing_m[1, 2] = shearing_xyz[2]

        output_affine_m = torch.mm(to_center_matrix, torch.mm(self.shearing_m, torch.mm(self.scaling_m,
                                                                                        torch.mm(self.rotation_m,
                                                                                                 torch.mm(
                                                                                                     reversed_to_center_matrix,
                                                                                                     self.translation_m)))))
        grid = F.affine_grid(output_affine_m[0:3].unsqueeze(0), x.shape, align_corners=True)
        transformed_x = F.grid_sample(x, grid, mode='bilinear', align_corners=True)

        return transformed_x, output_affine_m[0:3].unsqueeze(0)
