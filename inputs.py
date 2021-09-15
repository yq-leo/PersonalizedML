# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 19:31:50 2021
@author: QI YU
@email: yq123456leo@outlook.com
"""

import torch
import torch.backends.cudnn as cudnn
import pickle
import os
from network import SmallCNN, Tiny1DCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    cudnn.benchmark = True

def ReadResult(prefix = '', fileName = 'results.rs', extra_folder = ''):
    folderName = 'results/subsets/'
    
    dictfile = open(prefix + folderName + extra_folder + '/' + fileName, 'rb')
    readdict = pickle.load(dictfile)
    dictfile.close()
    
    return readdict

def LoadWeight(subsize, fileName, prefix = ''):
    model = Tiny1DCNN(num_classes = 2).to(device)
    if device.type == 'cuda':
        model = torch.nn.DataParallel(model)
    
    print('==> Loading Pre-trained Weights...')
    assert os.path.isdir(prefix + 'checkpoint'), 'Error: no checkpoint directory found!'
    weight = torch.load(prefix + 'checkpoint/' + fileName)
    model.load_state_dict(weight['net'])
    best_acc = weight['acc']
    end_epoch = weight['epoch']
    
    print('==> Model Information:')
    print('==> Best Accuracy: %.2f%%' % (100 * best_acc))
    print('==> End Epoch: %d' % end_epoch)
    print('==> Loading Completed!')
    
    return model
    
if __name__ == "__main__":
    #model = LoadWeight(fileName = '(0, 1)_ckpt.pth')
    pass