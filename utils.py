
import os
import torch
import numpy as np
import SimpleITK as sitk
from skimage.metrics import structural_similarity
from tqdm import tqdm
import shutil


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)

def set_log_path(path):
    global _log_path
    _log_path = path

def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)

def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    return log

def read_img(in_path):
    img_lit = []
    mo_src_list = []
    mo_tgt_list = []
    filenames = os.listdir(in_path)
    for f in tqdm(filenames):
        mo1 = float(f.split('_')[1])
        if mo1 == 0:
            mo1 = 0.5
        mo2 = float(f.split('_')[2].split('.')[0])
        mo_src_list.append(mo1)
        mo_tgt_list.append(mo2)
        img = sitk.ReadImage(os.path.join(in_path, f))
        img_vol = sitk.GetArrayFromImage(img)
        img_lit.append(img_vol)
    return img_lit*20, mo_src_list*20, mo_tgt_list*20

def read_test_img(in_path):
    img_lit = []
    mo_src_list = []
    mo_tgt_list = []
    filenames = os.listdir(in_path)
    for f in tqdm(filenames):
        mo1 = float(f.split('_')[1])
        if mo1 == 0:
            mo1 = 0.5
        mo2 = float(f.split('_')[2].split('.')[0])
        mo_src_list.append(mo1)
        mo_tgt_list.append(mo2)
        img = sitk.ReadImage(os.path.join(in_path, f))
        img_vol = sitk.GetArrayFromImage(img)
        img_lit.append(img_vol)
    return img_lit, mo_src_list, mo_tgt_list

def get_month(in_path):
    mo_src_list = []
    mo_tgt_list = []
    filenames = os.listdir(in_path)
    for f in tqdm(filenames):
        mo1 = float(f.split('_')[1])
        if mo1 == 0:
            mo1 = 0.5
        mo2 = float(f.split('_')[2].split('.')[0])
        mo_src_list.append(mo1)
        mo_tgt_list.append(mo2)
    return mo_src_list, mo_tgt_list


def psnr(image, ground_truth):
    mse = np.mean((image - ground_truth) ** 2)
    if mse == 0.:
        return float('inf')
    data_range = np.max(ground_truth) - np.min(ground_truth)
    return 20 * np.log10(data_range) - 10 * np.log10(mse)


def ssim(image, ground_truth):
    data_range = np.max(ground_truth) - np.min(ground_truth)
    return structural_similarity(image, ground_truth, data_range=data_range)

def write_ddf(vol, out_path, ref_path, new_spacing=None):
    img_ref = sitk.ReadImage(ref_path)
    img = sitk.GetImageFromArray(vol)
    img.SetDirection(img_ref.GetDirection())
    if new_spacing is None:
        img.SetSpacing(img_ref.GetSpacing())
    else:
        img.SetSpacing(tuple(new_spacing))
    img.SetOrigin(img_ref.GetOrigin())
    sitk.WriteImage(img, out_path)
    print('Save to:', out_path)

def write_img(vol, out_path, ref_path, new_spacing=None):
    img_ref = sitk.ReadImage(ref_path)
    img = sitk.GetImageFromArray(np.round(vol))
    img.SetDirection(img_ref.GetDirection())
    if new_spacing is None:
        img.SetSpacing(img_ref.GetSpacing())
    else:
        img.SetSpacing(tuple(new_spacing))
    img.SetOrigin(img_ref.GetOrigin())
    sitk.WriteImage(img, out_path)
    print('Save to:', out_path)


def normal(in_image):
    value_max = np.max(in_image)
    value_min = np.min(in_image)
    return (in_image - value_min) / (value_max - value_min)


def dice(array1, array2, labels=None, include_zero=False):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.
    Parameters:
        array1: Input array 1.
        array2: Input array 2.
        labels: List of labels to compute dice on. If None, all labels will be used.
        include_zero: Include label 0 in label list. Default is False.
    """
    if labels is None:
        labels = np.concatenate([np.unique(a) for a in [array1, array2]])
        labels = np.sort(np.unique(labels))
    if not include_zero:
        labels = np.delete(labels, np.argwhere(labels == 0)) 

    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem

def default_unet_features():
    nb_features = [
        [32, 64, 128, 128],             # encoder
        [128, 128, 64, 32, 1]  # decoder
    ]
    return nb_features
