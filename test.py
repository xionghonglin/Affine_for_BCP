import loss
import SimpleITK as sitk
import torch.nn as nn
import utils
from torch.utils.tensorboard import SummaryWriter
import argparse
from models.GlobalNet import GlobalNet
import torch
import data
import os
import numpy as np
from tqdm import tqdm
# import time
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--inchannel', type=int, default=1, dest='inchannel')
    parser.add_argument('--outchannel', type=int, default=1, dest='outchannel')
    # dir
    parser.add_argument('--src_test', type=str,
                        default='../data/bcp_affine_rigid/bcp_affine/val/src', dest='src_test')
    parser.add_argument('--tgt_test', type=str,
                        default='../data/bcp_affine_rigid/bcp_affine/val/tgt', dest='tgt_test')
    parser.add_argument('--load_model', type=str,
                        default='./best_25.pkl', dest='load_model')
    parser.add_argument('--result_dir', type=str,
                        default='./result', dest='result_dir')


    args = parser.parse_args()

    inchannel = args.inchannel
    outchannel = args.outchannel
    src_test = args.src_test
    tgt_test = args.tgt_test


    
    exp = 'AFFINE_COM_EPOCH25'
    result_dir = os.path.join(args.result_dir, exp)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    model_dir = args.load_model

    
    test_loader = data.loader_test(path_src=src_test,
                                  path_tgt=src_test,
                                  batch_size=1,
                                  is_train=False)


 
    file = os.listdir(src_test)
    model = GlobalNet()
    sv_file = torch.load(model_dir)
    model.load_state_dict(sv_file['model_state_dict'])
    model = model.cuda()
    with torch.no_grad():
        for i, (batch) in enumerate(tqdm(test_loader)):
            for k, v in batch.items():
                batch[k] = v

            src = batch['src'].unsqueeze(1).float().cuda()
            tgt = batch['tgt'].unsqueeze(1).float().cuda()
            mo1 = batch['m1'].unsqueeze(1).float().cuda()
            mo2 = batch['m2'].unsqueeze(1).float() .cuda()          
            pred_tgt = model(src/3, mo1, mo2)*3
            pred_tgt = pred_tgt.cpu().detach().numpy().squeeze(0).squeeze(0)
            pred_tgt = np.round(pred_tgt)
            pred_path = os.path.join(result_dir,file[i])
            ref = os.path.join('../data/bcp_affine_rigid/bcp_affine/val/tgt',file[i])
            utils.write_img(pred_tgt,pred_path,ref_path=ref)
            

    
