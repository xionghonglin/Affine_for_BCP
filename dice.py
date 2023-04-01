import SimpleITK as sitk
import numpy as np
import os
import loss


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
        
# #train
# pred_path = "./re/train_700.nii.gz"
# target_path = "./re/labelMNBCP116056.nii.gz"
#test

pred_path = 'D:\project/2023\pair_data\src_affine_org'
tgt_path = 'D:\project/2023\pair_data/tgtcrop'
dice1 = 0
dice2 = 0
dice3 = 0
list = os.listdir(pred_path)
dice1 = []
dice2 = []
dice3 = []
n = len(list)
ncc = 0
for file in os.listdir(pred_path):
    pred_img_path = os.path.join(pred_path,file)
    tgt_img_path = os.path.join(tgt_path,file)
    pred = sitk.ReadImage(pred_img_path)
    tgt = sitk.ReadImage(tgt_img_path)
    pred = sitk.GetArrayFromImage(pred)
    tgt = sitk.GetArrayFromImage(tgt)
    nccloss = loss.NCC(pred,tgt)
    print(file, nccloss)
    ncc += nccloss

print(ncc/702)


#     dicer = dice(pred,tgt)
#     print(file,dicer[0],dicer[1],dicer[2])

#     dice1.append(dicer[0])
#     dice2.append(dicer[1])
#     dice3.append(dicer[2])

# print('csf',np.average(dice1),'gm',np.average(dice2),'wm',np.average(dice3))
# print('csf',np.std(dice1),'gm',np.std(dice2),'wm',np.std(dice3))

# pred = sitk.ReadImage(pred_path)
# target = sitk.ReadImage(target_path)
# pred = sitk.GetArrayFromImage(pred)
# pred = np.round(pred)
# target = sitk.GetArrayFromImage(target)

# dice_list=[]
# dice_info=[]

# dicer = dice(pred,target)
# #dice_info.append(dice_list.item())
# print(dicer)

