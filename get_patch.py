import os
import SimpleITK as sitk
import random
import torch
def get_slice(path_src,path_tgt,path_src_patch,path_tgt_patch):
    filenames1 = os.listdir(path_src)
    #filenames2 = os.listdir(path_tgt)
    w = 64
    for file in filenames1:
        img_path_src = os.path.join(path_src,file)
        img_src = sitk.ReadImage(img_path_src)
        vol_src = sitk.GetArrayFromImage(img_src)

        img_path_tgt = os.path.join(path_tgt,file)
        img_tgt = sitk.ReadImage(img_path_tgt)
        vol_tgt = sitk.GetArrayFromImage(img_tgt)
        
        for i in range(20):
            x0 = random.randint(10, 100)
            y0 = random.randint(30, 180)
            z0 = random.randint(10, 120)
            
            crop_src = vol_src[x0:x0+w,y0:y0+w,z0:z0+w]
            crop_tgt = vol_tgt[x0:x0+w,y0:y0+w,z0:z0+w]
            f = file.split('.')[0]
            name = f+'_{}.nii.gz'.format(i)
            result_src = os.path.join(path_src_patch, name)
            result_tgt = os.path.join(path_tgt_patch, name)
            img1 = sitk.GetImageFromArray(crop_src)
            
            sitk.WriteImage(img1,result_src)
            img2 = sitk.GetImageFromArray(crop_tgt)
            sitk.WriteImage(img2,result_tgt)

path_src = './data/train/src'
path_tgt = './data/train/tgt'
path_src_patch = './data/train/patch_src_64'
path_tgt_patch = './data/train/patch_tgt_64'
get_slice(path_src,path_tgt,path_src_patch,path_tgt_patch)

