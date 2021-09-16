# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 20:05:35 2021
@author: QI YU
@email: yq123456leo@outlook.com
"""

import torch
import torch.nn as nn
import hiddenlayer as h
from torchviz import make_dot

class CNN1D(nn.Module):
    def __init__(self, num_classes = 6):
        super(CNN1D, self).__init__()
        
        self.features = nn.Sequential(      # 128 * 9
            nn.Conv1d(9, 64, 3),            # 126 * 64
            nn.ReLU(),                      # 126 * 64
            nn.MaxPool1d(2),                # 63 * 64
            nn.Conv1d(64, 128, 3),          # 61 * 128
            nn.ReLU(),                      # 61 * 128
            nn.MaxPool1d(2),                # 30 * 128
            nn.Conv1d(128, 256, 3),         # 28 * 256
            nn.ReLU(),                      # 28 * 256
            nn.MaxPool1d(2))                # 14 * 256
        
        self.classifier = nn.Sequential(    # 14 * 256
            nn.Linear(14 * 256, 512),       # 512
            nn.ReLU(),                      # 512
            nn.Linear(512, 128),            # 128
            nn.ReLU(),                      # 128
            nn.Linear(128, num_classes))    # number of classes
    
    def forward(self, x):
        features_out = self.features(x)
        res = features_out.view(features_out.size(0), -1)
        out = self.classifier(res)
        return out


class Tiny1DCNN(nn.Module):
    def __init__(self, num_classes = 6):
        super(Tiny1DCNN, self).__init__()
        
        self.features = nn.Sequential(      # 128 * 9
            nn.Conv1d(9, 16, 3),            # 126 * 16
            nn.ReLU(),                      # 126 * 16
            nn.MaxPool1d(2))                # 63 * 16
            
        self.classifier = nn.Sequential(    # 63 * 16
            nn.Linear(63 * 16, 32),         # 32
            nn.ReLU(),                      # 32
            nn.Linear(32, num_classes))     # number of classes

    def forward(self, x):
        features_out = self.features(x)
        res = features_out.view(features_out.size(0), -1)
        out = self.classifier(res)
        return out


class SmallCNN(nn.Module):
    def __init__(self, num_classes = 6):
        super(SmallCNN, self).__init__()
        
        self.features = nn.Sequential(      # 128 * 9
            nn.Conv1d(9, 16, 3),            # 126 * 64
            nn.ReLU(),                      # 126 * 64
            nn.MaxPool1d(2),                # 63 * 64
            nn.Conv1d(16, 32, 3),           # 61 * 128
            nn.ReLU(),                      # 61 * 128
            nn.MaxPool1d(2),                # 30 * 128
            nn.Conv1d(32, 64, 3),           # 28 * 256
            nn.ReLU(),                      # 28 * 256
            nn.MaxPool1d(2))                # 14 * 256
        
        self.classifier = nn.Sequential(    # 14 * 256
            nn.Linear(14 * 64, 128),        # 512
            nn.ReLU(),                      # 512
            nn.Linear(128, 32),             # 128
            nn.ReLU(),                      # 128
            nn.Linear(32, num_classes))     # number of classes
    
    def forward(self, x):
        features_out = self.features(x)
        res = features_out.view(features_out.size(0), -1)
        out = self.classifier(res)
        return out


if __name__ == "__main__":
    # nerual network visualization

    '''
    mynet = Tiny1DCNN(2)
    vis_graph = h.build_graph(mynet, torch.zeros([64, 9, 128]))
    vis_graph.theme = h.graph.THEMES["blue"].copy()
    vis_graph.save("networks/demo1.png")
    '''
    
    mynet = Tiny1DCNN(2)
    x = torch.randn(64, 9, 128).requires_grad_(True)
    y = mynet(x)
    MyConvNetVis = make_dot(y, params=dict(list(mynet.named_parameters()) + [('x', x)]))
    MyConvNetVis.format = "png"
    MyConvNetVis.directory = "networks"
    MyConvNetVis.view()
        
        
            
            