import torch
from models.vit import ViT
from tqdm import tqdm



v = ViT(
    image_size = 224,          # image size
    frames = 192,               # number of frames
    image_patch_size = 32,     # image patch size
    frame_patch_size = 32,      # frame patch size
    num_classes = 12,
    dim = 1024,
    depth = 1,
    heads = 8,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)
v = v.cuda()
img = torch.randn(1, 1, 192, 224, 224).cuda() # (batch, channels, frames, height, width)
time = torch.randn(1,2).cuda()
preds = v(img, time)
print(preds)
