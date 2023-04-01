
import utils
import numpy as np
import SimpleITK as sitk
from torch.utils import data
from scipy import ndimage as nd
import random


def random_crop(src, tgt):
    w = 64
    x0 = random.randint(10, 100)
    y0 = random.randint(30, 180)
    z0 = random.randint(10, 120)

    return src[x0:x0+w,y0:y0+w,z0:z0+w], tgt[x0:x0+w,y0:y0+w,z0:z0+w]

def crop(src, tgt):
    return src[:,16:,:], tgt[:,16:,:]

class ImgTrain(data.Dataset):
    def __init__(self, path_src, path_tgt, is_train):
        self.is_train = is_train
        self.src, self.mo1, self.mo2 = utils.read_test_img(in_path=path_src)
        self.tgt, self.mo1_src, self.mo2_tgt = utils.read_test_img(in_path=path_tgt)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, item):
        src = self.src[item]

        tgt = self.tgt[item]
        #src, tgt = random_crop(src, tgt)
        mo1 = self.mo1[item]
        mo2 = self.mo2[item]
        src, tgt = crop(src, tgt)
        return {'src': src,
                'tgt': tgt,
                'm1': mo1,
                'm2': mo2
                }


def loader_train(path_src, path_tgt, batch_size, is_train):

    return data.DataLoader(
        dataset=ImgTrain(path_src=path_src, path_tgt=path_tgt, is_train=is_train),
        batch_size=batch_size,
        shuffle=is_train
    )

class Imgval(data.Dataset):
    def __init__(self, path_src, path_tgt, is_train):
        self.is_train = is_train
        self.src, self.mo1, self.mo2 = utils.read_test_img(in_path=path_src)
        self.tgt, self.mo1_src, self.mo2_tgt = utils.read_test_img(in_path=path_tgt)
        #self.mo1, self.mo2 = utils.get_month(in_path=path_src)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, item):
        src = self.src[item]
        tgt = self.tgt[item]
        #src, tgt = random_crop(src, tgt)
        mo1 = self.mo1[item]
        mo2 = self.mo2[item]
        src, tgt = crop(src, tgt)
        return {'src': src,
                'tgt': tgt,
                'm1': mo1,
                'm2': mo2
                }

def loader_val(path_src, path_tgt, batch_size, is_train):

    return data.DataLoader(
        dataset=Imgval(path_src=path_src, path_tgt=path_tgt, is_train=is_train),
        batch_size=batch_size,
        shuffle=is_train
    )

class Imgtest(data.Dataset):
    def __init__(self, path_src, path_tgt, is_train):
        self.is_train = is_train
        self.src, self.mo1, self.mo2 = utils.read_test_img(in_path=path_src)
        self.tgt, self.mo1_src, self.mo2_tgt = utils.read_test_img(in_path=path_tgt)
        #self.mo1, self.mo2 = utils.get_month(in_path=path_src)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, item):
        src = self.src[item]
        tgt = self.tgt[item]
        mo1 = self.mo1[item]
        mo2 = self.mo2[item]
        return {'src': src,
                'tgt': tgt,
                'm1': mo1,
                'm2': mo2,
                }
def loader_test(path_src, path_tgt, batch_size, is_train):

    return data.DataLoader(
        dataset=Imgtest(path_src=path_src, path_tgt=path_tgt, is_train=is_train),
        batch_size=batch_size,
        shuffle=is_train
    )


