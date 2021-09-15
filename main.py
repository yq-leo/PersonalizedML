# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 19:54:46 2021
@author: QI YU
@email: yq123456leo@outlook.com
"""

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import numpy as np
from collections import defaultdict
from itertools import combinations
import time
import os
from sklearn import metrics

from dataset import HARdataset, SubDataset
from network import CNN1D, SmallCNN, Tiny1DCNN
from export import WriteCfsMat, PlotBaseP, PlotBaseR, PlotMetricsGroup, PlotAccCurve, PlotBasePRBar, WriteResult

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    cudnn.benchmark = True

def RunBaseline(attr, prefix = ''):
    classes = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
    wholeset = list(range(6))
    
    lr = attr['lr']
    batch_size = attr['batch_size']
    epochs = attr['epochs']
    
    train_data = HARdataset(group = 'train', prefix = prefix)
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 2)
    test_data = HARdataset(group = 'test', prefix = prefix)
    test_loader = DataLoader(test_data, batch_size = batch_size, num_workers = 2)
    
    model = CNN1D().to(device)
    #model = SmallCNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.985)

    results = defaultdict(list)
    for epoch in range(epochs):
        train_loss = 0
        start_time = time.perf_counter()
        model.train()
        for index, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            if index % 20 == 19:
                print('epoch %d, index %d, loss: %.4f' % (epoch + 1, index + 1, train_loss / 100))
                train_loss = 0
        end_time = time.perf_counter()

        if epoch % 3 == 2:
            for p in optimizer.param_groups:
                p['lr'] *= 0.975
        
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            cfs_mat = np.zeros((6, 6), dtype = int)
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.long().to(device)
                
                outputs = model(inputs.float())
                conf, predicted = torch.max(outputs.detach(), 1)
                current_batch_size = labels.size(0)
                for i in range(current_batch_size):
                    cfs_mat[int(labels[i])][int(predicted[i])] += 1
                total += current_batch_size
                correct += (predicted == labels).sum()
            
            print('-----Results Epoch %s-----' % ('0' * (3 - len(str(epoch + 1))) + str(epoch + 1)))
            print(cfs_mat)
            train_time = end_time - start_time
            print('Training time: %.3fs' % train_time)
            acc = correct / total
            results['accuracy'].append(acc)
            print('Accuracy: %.2f%%' % (100 * acc))
            
            TPFP = cfs_mat.sum(0)
            TPFN = cfs_mat.sum(1)
            for idx in wholeset:
                TP = cfs_mat[idx][idx]
                precision = TP / TPFP[idx] if TPFP[idx] != 0 else np.inf
                recall = TP / TPFN[idx] if TPFN[idx] != 0 else np.inf
                if len(results[idx]) == 0:
                    results[idx] = [[precision], [recall]]
                else:
                    results[idx][0].append(precision)
                    results[idx][1].append(recall)
                print('%s,' % classes[idx] + ' ' * (18 - len(classes[idx])) + 'Precision: %.2f%%, Recall: %.2f%%' % (100 * precision, 100 * recall))
            print('---------------------------')
            print()

        #scheduler.step()

    return results

def BinClassifier(train_loader, test_loader, subset, attr, prefix = ''):
    classes = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
    
    lr = attr['lr']
    epochs = attr['epochs']
    group = attr['group']
    subsize = len(subset)
    
    profolder = ''
    
    #model = CNN1D().to(device)
    model = Tiny1DCNN(num_classes = 2).to(device)
    if device.type == 'cuda':
        model = torch.nn.DataParallel(model)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)
    
    print('-----Labels:[' + ('%s, ' * subsize) % tuple(classes[label] for label in subset) + ']-----')
    results = defaultdict(list)
    best_acc = 0
    for epoch in range(epochs):
        print('----------Epoch %d---------' % (epoch + 1))
        
        train_loss = 0
        for index, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            if index % 10 == 9:
                print('Epoch %d, Index %d, Loss: %.4f' % (epoch + 1, index + 1, train_loss / 100))
                train_loss = 0
        
        with torch.no_grad():
            y_scores_1, y_scores_2 = list(), list()
            y_pred = list()
            
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs.float())
                
                # Probabilities
                y_scr_1 = F.softmax(outputs, dim = 1).cpu().detach().numpy().tolist()
                y_scores_1 += y_scr_1
                
                # Direct label predicition
                conf, predicted = torch.max(outputs.detach(), 1)
                batch_size = labels.size(0)
                y_scr_2= np.zeros((batch_size, 2), dtype = int).tolist()
                for idx, label in enumerate(predicted):
                    y_scr_2[idx][label] = 1
                y_scores_2 += y_scr_2
                
                # Label prediction
                y_pred += predicted.cpu().detach().numpy().tolist()
            
            #print(cfs_mat)
            print('----------Results----------')
            
            # Confusion Matrix
            y_true = test_loader.dataset.y
            y_pred = np.array(y_pred)
            cfs_mat = metrics.confusion_matrix(y_true, y_pred)
            print(cfs_mat)
            
            # Accuracy
            acc = metrics.accuracy_score(y_true, y_pred)
            results['accuracy'].append(acc)
            print('Accuracy: %.2f%%' % (100 * acc))
            print()
            
            # Precision & Recall
            precisions = metrics.precision_score(y_true, y_pred, average = None)
            recalls = metrics.recall_score(y_true, y_pred, average = None)
            print('In-Dictionary label     => Precision: %.2f%%, Recall: %.2f%%' % (100 * precisions[0], 100 * recalls[0]))
            print('Out-of-Dictionary label => Precision: %.2f%%, Recall: %.2f%%' % (100 * precisions[1], 100 * recalls[1]))
            print('---------------------------')
            
            '''
            ## Average Precision
            macro_precision = metrics.precision_score(y_true, y_pred, average = 'macro')
            weighted_precision = metrics.precision_score(y_true, y_pred, average = 'weighted')
            results['macro_precision'].append(macro_precision)
            results['weighted_precision'].append(weighted_precision)
            print('Macro Average Precison: %.2f%%' % (100 * macro_precision))
            print('Weighted Average Precison: %.2f%%' % (100 * weighted_precision))
            
            ## Average Recall
            macro_recall = metrics.recall_score(y_true, y_pred, average = 'macro')
            weighted_recall = metrics.recall_score(y_true, y_pred, average = 'weighted')
            results['macro_recall'].append(macro_recall)
            results['weighted_recall'].append(weighted_recall)
            print('Macro Average Recall: %.2f%%' % (100 * macro_recall))
            print('Weighted Average Recall: %.2f%%' % (100 * weighted_recall))
            print()
            '''
            
            if acc > best_acc:
                print('Saving...')
                state = {'net': model.state_dict(), 'acc': acc, 'epoch': epoch}
                if not os.path.isdir(prefix + 'checkpoint/classifier1/' + group + '/' + profolder):
                    os.mkdir(prefix + 'checkpoint/classifier1/' + group + '/' + profolder)
                torch.save(state, prefix + './checkpoint/classifier1/' + group + '/' + profolder + '%s_ckpt.pth' % str(subset))
                best_acc = acc
            
            if epoch % 10 == 9:
                fileName = str(subset) + '_result_epoch' + '0' * (3 - len(str(epoch + 1))) + str(epoch + 1) + '.rs'
                WriteResult(result = results, prefix = prefix, fileName = fileName, extra_folder = 'classifier1/' + group + '/' + profolder)
                if epoch != 9:
                    lastFile = str(subset) + '_result_epoch' + '0' * (3 - len(str(epoch - 9))) + str(epoch - 9) + '.rs'
                    assert os.path.exists(prefix + 'results/subsets/classifier1/'+ group + '/' + profolder + lastFile), 'Error: No Result File Found!'
                    os.remove(prefix + 'results/subsets/classifier1/' + group + '/' + profolder + lastFile)
                
        scheduler.step()
    
    return results

if __name__ == "__main__":
    #prefix = '/content/gdrive/MyDrive/week2/HAR+1DCNN/'
    prefix = ''
    
    labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
    wholeset = list(range(len(labels)))
    subsize = 2
    
    lr = 0.0005
    batch_size = 64
    epochs = 200
    attr = {'lr': lr, 'batch_size': batch_size, 'epochs': epochs}
    
    attr['group'] = 'fixed_training'
    
    combin2 = list(combinations(wholeset, subsize))
    subsets1 = combin2[:5]
    subsets2 = list()
    for i in range(3, 5):
        subset = tuple(range(i))
        subsets2.append(subset)
    subsets = subsets1 + subsets2
    
    subsets = [(0, 1, 3, 4, 5)]
    
    print(subsets)
    #X = np.linspace(5, 95, 19, dtype = int)
    for subset in subsets:
        train_data = HARdataset(group = 'train', prefix = prefix)
        test_data = HARdataset(group = 'test', prefix = prefix)
    
        train_data.Balance(subset = subset, target = 'classifier1')
        test_data.Balance(subset = subset, target = 'classifier1')
    
        train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 2)
        test_loader = DataLoader(test_data, batch_size = batch_size, num_workers = 2)
    
        results = BinClassifier(train_loader, test_loader, subset, attr, prefix)
    
    