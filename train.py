import loss
import SimpleITK as sitk
import torch.nn as nn
import utils
from torch.utils.tensorboard import SummaryWriter
import argparse
from models.GlobalNet import GlobalNet
from models.GlobalNet import Center_of_mass_initial_pairwise
import torch
import data
import os
import numpy as np
from tqdm import tqdm
# import time
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



def prepare_training(resume):
    model = GlobalNet().cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    epoch_start = 0
    if args.resume is not None:
        sv_file = torch.load(resume)
        model.load_state_dict(sv_file['model_state_dict'])
        model = model.cuda()
        optimizer.load_state_dict(sv_file['optimizer_state_dict'])
        epoch_start = sv_file['epoch'] + 1
    
    return model, optimizer, epoch_start


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--inchannel', type=int, default=1, dest='inchannel')
    parser.add_argument('--outchannel', type=int, default=1, dest='outchannel')

    # dir
    parser.add_argument('--src_train', type=str,
                        default='./test/src', dest='src_train') # ../data/bcp_affine_rigid/bcp_affine/val/src../data_bcp_affine/train/src
    parser.add_argument('--tgt_train', type=str,
                        default='./test/mat', dest='tgt_train') #../data_bcp_affine/train/tgt
    parser.add_argument('--src_val', type=str,
                        default='./test/src', dest='src_val') #../data_bcp_affine/train/tgt
    parser.add_argument('--tgt_val', type=str,
                        default='./test/mat', dest='tgt_val') #../data_bcp_affine/val/tgt
    parser.add_argument('--model_dir', type=str,
                        default='./checkpoint', dest='model_dir')
    parser.add_argument('--result_dir', type=str,
                        default='./result', dest='result_dir')
    parser.add_argument('--log_dir', type=str,
                        default='./log', dest='log_dir')
    parser.add_argument('--resume', type=str,
                        default=None, dest='resume')

    # training parameters
    parser.add_argument('-lr', type=float, default=1e-4, dest='lr')
    parser.add_argument('-lr_decay_epoch', type=int,
                        default=200, dest='lr_decay_epoch')
    parser.add_argument('-epoch', type=int, default=1000, dest='epoch')
    parser.add_argument('-summary_epoch', type=int,
                        default=200, dest='summary_epoch')
    parser.add_argument('-bs', type=int, default=2, dest='batch_size')
    parser.add_argument('-gpu', type=int, default=0, dest='gpu')

    args = parser.parse_args()

    inchannel = args.inchannel
    outchannel = args.outchannel

    src_train = args.src_train
    tgt_train = args.tgt_train
    src_val = args.src_val
    tgt_val = args.tgt_val
    lr = args.lr
    lr_decay_epoch = args.lr_decay_epoch
    epoch = args.epoch
    summary_epoch = args.summary_epoch
    batch_size = args.batch_size
    gpu = args.gpu

    
    exp = 'affine_com_mp'
    log_dir = os.path.join(args.log_dir, exp)
    model_dir = os.path.join(args.model_dir, exp)
    result_dir = os.path.join(args.result_dir, exp)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log = utils.set_save_path(log_dir)
    writer = SummaryWriter(log_dir)
    
    train_loader = data.loader_train(path_src=src_train,
                                     path_tgt=tgt_train,
                                     batch_size=batch_size,
                                     is_train=True)
    val_loader = data.loader_val(path_src=src_val,
                                   path_tgt=tgt_val,
                                   batch_size=batch_size,
                                   is_train=False)

    

 
    # model = LTC(in_channel=1, out_channel=1).cuda()
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    model, optimizer, epoch_start = prepare_training(args.resume)
    imgloss_func = nn.L1Loss()
    ncc_func = loss.NCCScore()
    minloss = 100

    
    for e in range(epoch_start, epoch+1):
        log_info = []
        model.train()
        loss_train = 0
        ddfloss = 0
        regloss = 0
        simloss = 0
        for batch in tqdm(train_loader):
            for k, v in batch.items():
                batch[k] = v.cuda()

            src = batch['src'].unsqueeze(1).float()
            tgt = batch['tgt'].float()
            mo1 = batch['m1'].unsqueeze(1).float()
            mo2 = batch['m2'].unsqueeze(1).float()
            print(tgt.shape)

            pred_tgt = model(src/3, mo1, mo2)
            # print(pred_tgt.shape,tgt.shape)
            img_loss = imgloss_func(tgt, pred_tgt)
            
            loss = img_loss 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_train += loss.item()
 
        print('(TRAIN) Epoch[{}/{}], img_loss:{:.10f}'.format(e + 1,
                                                              epoch,
                                                              (loss_train /
                                                               len(train_loader)),
                                                              ))
        writer.add_scalar('loss', (loss_train / len(train_loader)), e)

        log_info.append('loss_train={:.4f}, epoch={}'.format((loss_train / len(train_loader)),
                                        
                                                                          e + 1))

        torch.save({'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, os.path.join(model_dir, 'last.pkl'))
        src = None
        tgt = None
        
        model.eval()
        with torch.no_grad():
            loss_val = 0
            simloss = 0
            NCC_s = 0
            for batch in tqdm(val_loader):
                for k, v in batch.items():
                    batch[k] = v.cuda()

                src = batch['src'].unsqueeze(1).float()
                tgt = batch['tgt'].float()
                mo1 = batch['m1'].unsqueeze(1).float()
                mo2 = batch['m2'].unsqueeze(1).float()

                pred_tgt = model(src/3, mo1, mo2)
                
                img_loss = imgloss_func(tgt, pred_tgt)
                loss = img_loss
                simloss += img_loss.item()
                loss_val += loss


                if loss_val < minloss:
                    minloss = loss_val
        
                    torch.save({'epoch': e,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()
                                }, os.path.join(model_dir, 'best_{}.pkl'.format(e+1)))

            # writer.add_scalar('val_loss', (loss_val / len(val_loader)), e)
            # writer.add_scalar('ncc_score', (NCC_s / len(val_loader)), e)
            # log_info.append('loss_val={:.4f}, ncc={:.6f}'.format((loss_val / len(val_loader)),
            #                                                      (NCC_s / len(val_loader))))



        log(', '.join(log_info))
