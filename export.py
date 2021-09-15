# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 18:30:16 2021
@author: QI YU
@email: yq123456leo@outlook.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pickle
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from itertools import combinations
from collections import defaultdict

from dataset import HARdataset
from inputs import LoadWeight

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    cudnn.benchmark = True

def ExportBaseRes(results, attr, prefix = ''):
    pass

def PlotBasePRBar(results, attr, classes, prefix = '', fileName = 'PR barchart'):
    plt.rcdefaults()
    fig = plt.figure()
    attrinfo = 'batch_size = %d, ' % attr['batch_size'] + 'lr(init) = %s, ' % ('{:g}'.format(attr['lr'])) + 'epochs = %d' % attr['epochs']
    fig.suptitle('Precision & Recall: Baseline\n' + attrinfo, fontsize = 14, fontweight = 'bold')
    fig.subplots_adjust(top = 0.8)
    
    pBar, rBar = fig.add_subplot(211), fig.add_subplot(212)
    
    y_pos = np.arange(len(classes))
    
    num_sample = 20
    epochs = attr['epochs']
    upperbound = max(epochs - num_sample, 0)
    real_sample = epochs - upperbound
    
    wholeset = list(results.keys())
    wholeset.remove('accuracy')
    
    precisions, recalls = list(), list()
    for label in wholeset:
        pres, recs = results[label][0], results[label][1]
        pre = sum(pres[upperbound:]) / real_sample
        rec = sum(recs[upperbound:]) / real_sample
        precisions.append(pre)
        recalls.append(rec)
    
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    
    pBar.barh(y_pos, precisions, align = 'center')
    pBar.set_yticks(y_pos)
    pBar.set_yticklabels(classes, fontsize = 7)
    pBar.invert_yaxis()
    pBar.set_xlabel('precision')
    for a, b in zip(precisions, y_pos):
        pBar.text(a, b, '%.2f%%' % (100 * a), fontsize = 5)
    
    rBar.barh(y_pos, recalls, align = 'center')
    rBar.set_yticks(y_pos)
    rBar.set_yticklabels(classes, fontsize = 7)
    rBar.invert_yaxis()
    rBar.set_xlabel('recall')
    for a, b in zip(recalls, y_pos):
        rBar.text(a, b, '%.2f%%' % (100 * a), fontsize = 5)
    
    plt.tight_layout()
    plt.savefig(prefix + 'results/' + fileName, dpi = 720)
    plt.show()
    plt.clf()

def PlotAccCurve(results, attr, prefix = '', fileName = 'accuracy.jpg'):
    accuracys = results['accuracy']
    epochs = list(range(len(accuracys)))
    
    X, Y = np.array(epochs), np.array(accuracys)
    
    fig = plt.figure()
    fig.suptitle('Accuracy Curve: Baseline', fontsize=14, fontweight='bold')
    fig.subplots_adjust(top = 0.85)
    
    curve = fig.add_subplot(111)
    curve.set_title('batch_size = %d, ' % attr['batch_size'] + 'lr(init) = %s, ' % ('{:g}'.format(attr['lr'])) + 'epochs = %d' % attr['epochs'])
    curve.plot(X, Y)
    curve.set_xlabel('Number of epoch')
    curve.set_ylabel('Accuracy')
    plt.savefig(prefix + 'results/' + fileName, dpi = 720)
    plt.clf()

def PlotMetricsGroup(results, group, attr, prefix = '', fileName = 'metrics.jpg', save = False):
    if group not in ['Precision', 'Recall', 'Accuracy']:
        print('Invalid Metrics')
        return
    
    fig = plt.figure()
    fig.suptitle(group + ' curve: Multiple Combinations', fontsize = 14, fontweight = 'bold')
    fig.subplots_adjust(top = 0.85)
    
    curve = fig.add_subplot(111)
    curve.set_title('batch_size = %d, ' % attr['batch_size'] + 'lr(init) = %s, ' % ('{:g}'.format(attr['lr'])) + 'epochs = %d' % attr['epochs'])
    subsets = list(results.keys())
    for subset in subsets:
        res = results[subset]
        if group == 'Accuracy':
            metrics = res['accuracy']
        elif group == 'Precision':
            metrics = res['precision']
        else:
            metrics = res['recall']
            
        epochs = list(range(len(metrics)))
        X, Y = np.array(epochs), np.array(metrics)
        curve.plot(X, Y, label = str(subset))
    curve.set_xlabel('Number of epochs')
    curve.set_ylabel(group)
    curve.legend()
    if save:
        plt.savefig(prefix + 'results/' + fileName, dpi = 720)
    plt.show()
    plt.clf()

def PlotBaseP(results, classes, attr, prefix = '', fileName = 'precisions.jpg'):
    wholeset = list(results.keys())
    wholeset.remove('accuracy')
    if wholeset != list(range(len(classes))):
        print('Training Incomplete')
        return
    
    fig = plt.figure()
    fig.suptitle('Precisions Curve: Baseline', fontsize = 14, fontweight = 'bold')
    fig.subplots_adjust(top = 0.85)
    
    curve = fig.add_subplot(111)
    curve.set_title('batch_size = %d, ' % attr['batch_size'] + 'lr(init) = %s, ' % ('{:g}'.format(attr['lr'])) + 'epochs = %d' % attr['epochs'])
    for label in wholeset:
        res = results[label]
        precisions, epochs = res[0], list(range(len(res[0])))
        X, Y = np.array(epochs), np.array(precisions)
        curve.plot(X, Y, label = classes[label])
    curve.set_xlabel('Number of epochs')
    curve.set_ylabel('Precision')
    curve.legend()
    plt.savefig(prefix + 'results/' + fileName, dpi = 720)
    plt.show()
    plt.clf()
    

def PlotBaseR(results, classes, attr, prefix = '', fileName = 'recalls.jpg'):
    wholeset = list(results.keys())
    wholeset.remove('accuracy')
    if wholeset != list(range(len(classes))):
        print('Training Incomplete')
        return
    
    fig = plt.figure()
    fig.suptitle('Recalls Curve: Baseline', fontsize = 14, fontweight = 'bold')
    fig.subplots_adjust(top = 0.85)
    
    curve = fig.add_subplot(111)
    curve.set_title('batch_size = %d, ' % attr['batch_size'] + 'lr(init) = %s, ' % ('{:g}'.format(attr['lr'])) + 'epochs = %d' % attr['epochs'])
    
    for label in wholeset:
        res = results[label]
        recalls, epochs = res[1], list(range(len(res[0])))
        X, Y = np.array(epochs), np.array(recalls)
        plt.plot(X, Y, label = classes[label])
    curve.set_xlabel('Number of epochs')
    curve.set_ylabel('Recalls')
    curve.legend()
    plt.savefig(prefix + 'results/' + fileName, dpi = 720)
    plt.show()
    plt.clf()
    

def WriteCfsMat(results, wholeset, subsize, prefix = '', fileName = 'cfs_mat.xlsx'):
    subsets = list(results.keys())
    if subsets != list(combinations(wholeset, subsize)):
        print('Training Incomplete')
        return
    
    num_sample = 20
    epochs = len(results[subsets[0]][subsets[0][0]][0])
    upperbound = max(epochs - num_sample, 0)
    real_sample = epochs - upperbound
    setsize = len(wholeset)
    
    label_P_mat, label_R_mat = np.ones((setsize, setsize)), np.ones((setsize, setsize))
    other_P_mat, other_R_mat = np.ones((setsize, setsize)), np.ones((setsize, setsize))
    accuracy_mat = np.zeros((setsize, setsize))
    for subset in subsets:
        res_dict = results[subset]
        
        idx1, idx2 = subset
        for label in subset:
            other = idx1 if idx1 != label else idx2
            avg_p = sum(res_dict[label][0][upperbound:]) / real_sample
            label_P_mat[other][label] = round(avg_p, 4)
            avg_r = sum(res_dict[label][1][upperbound:]) / real_sample
            label_R_mat[other][label] = round(avg_r, 4)
        avg_p = sum(res_dict['other'][0][upperbound:]) / real_sample
        other_P_mat[idx1][idx2] = round(avg_p, 4)
        avg_r = sum(res_dict['other'][1][upperbound:]) / real_sample
        other_R_mat[idx1][idx2] = round(avg_r, 4)
        avg_acc = sum(res_dict['accuracy'][0][upperbound:]) / real_sample
        accuracy_mat[idx1][idx2] = round(avg_acc, 4)
    
    head = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
    label_P_df = pd.DataFrame(label_P_mat, index = head, columns = head)
    label_R_df = pd.DataFrame(label_R_mat, index = head, columns = head)
    other_P_df = pd.DataFrame(other_P_mat, index = head, columns = head)
    other_R_df = pd.DataFrame(other_R_mat, index = head, columns = head)
    accuracy_df = pd.DataFrame(accuracy_mat, index = head, columns = head)
    
    cmap = sns.light_palette("green", as_cmap=True)
    label_P_df = label_P_df.style.background_gradient(cmap = cmap)
    label_R_df = label_R_df.style.background_gradient(cmap = cmap)
    
    foldName = 'results/subsets/'
    writer = pd.ExcelWriter(prefix + foldName + fileName, engine='xlsxwriter')
    label_P_df.to_excel(writer, sheet_name = 'Labels Precision', index_label = 'Precision')
    label_R_df.to_excel(writer, sheet_name = 'Labels Recall', index_label = 'Recall')
    other_P_df.to_excel(writer, sheet_name = 'Others Precision', index_label = 'Precision')
    other_R_df.to_excel(writer, sheet_name = 'Others Recall', index_label = 'Recall')
    accuracy_df.to_excel(writer, sheet_name = 'Accuracy', index_label = 'Accuracy')
    
    sheetNames = ['Labels Precision', 'Labels Recall', 'Others Precision', 'Others Recall', 'Accuracy']
    for name in sheetNames:
      worksheet = writer.sheets[name]
      worksheet.set_column('A:G', 20)
    
    writer.save()

def WriteResult(result, prefix, subroute, fileName = 'results.rs'):
    folderName = 'results/subsets/'
    
    rsfile = open(prefix + folderName + subroute + '/' + fileName, 'wb')
    pickle.dump(result, rsfile)
    rsfile.close()

def PlotWeightedTest(subsets, prefix = '', fileName = 'plot1.jpg', save = False):
    X = np.linspace(0, 100, 21)
    
    fig = plt.figure()
    fig.suptitle('Accuracy Curve (HAR)', fontsize = 14, fontweight = 'bold')
    fig.subplots_adjust(top = 0.85)
    
    curve = fig.add_subplot(111)
    curve.set_title("Parameter: Percentage of IN-DICTIONARY label")
    for subset in subsets:
        subsize = len(subset)
        weightName = '%s_ckpt.pth' % str(subset)
        model = LoadWeight(subsize = subsize, fileName = weightName, prefix = prefix)
        
        accuracy = list()
        for proportion in X:
            test_data = HARdataset(group = 'test', prefix = prefix)
            test_data.ShapeWeight(subset = subset, proportion = proportion / 100)
            test_loader = DataLoader(dataset = test_data, batch_size = 64, num_worker = 2)
            results = Test(model = model, subset = subset, test_loader = test_loader)
            accuracy.append(results['accuracy'][0])
        Y = np.array(accuracy)
        curve.plot(X, Y, label = str(subset))
        curve.set_xticks(X)
    
    curve.set_xlabel('Percentage of IN-DICTIONARY label (%)')
    curve.set_ylabel('Accuracy')
    
    curve.legend()
    if save:
        plt.savefig(prefix + 'results/subsets/' + fileName, dpi = 720)
    plt.show()
    plt.clf()

def PlotNotInTest(subsets, prefix = '', fileName = 'plot2.jpg', save = False):
    fig = plt.figure()
    fig.suptitle('Accuracy Curve (HAR)', fontsize = 14, fontweight = 'bold')
    fig.subplots_adjust(top = 0.85)
    
    curve = fig.add_subplot(111)
    curve.set_title("Parameter: Number of OUT-OF-DICTIONARY label")
    for subset in subsets:
        subsize = len(subset)
        weightName = '%s_ckpt.pth' % str(subset)
        model = LoadWeight(subsize = subsize, fileName = weightName, prefix = prefix)
        
        accuracy = list()
        X = list(range(1, 10 - subsize + 1))
        for num_other in X:
            test_data = HARdataset(group = 'test', prefix = prefix)
            test_data.ShapeNotIn(subset = subset, num_other = num_other)
            test_loader = DataLoader(dataset = test_data, batch_size = 64)
            results = Test(model = model, subset = subset, test_loader = test_loader)
            accuracy.append(results['accuracy'][0])
        X = np.array(X)
        Y = np.array(accuracy)
        curve.plot(X, Y, label = str(subset))
        curve.set_xticks(X)
    
    curve.set_xlabel('Number of OUT-OF-DICTIONARY label')
    curve.set_ylabel('Accuracy')
    
    curve.legend()
    if save:
        plt.savefig(prefix + 'results/subsets/' + fileName, dpi = 720)
    plt.show()
    plt.clf()

def Test(model, subset, test_loader):
    results = defaultdict(list)
    subsize = len(subset)
    classes = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
    with torch.no_grad():
        correct = 0
        total = 0
        cfs_mat = np.zeros((subsize + 1, subsize + 1), dtype = int)
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
                
            outputs = model(inputs.float())
            conf, predicted = torch.max(outputs.detach(), 1)
            current_batch_size = labels.size(0)
            for i in range(current_batch_size):
                cfs_mat[int(labels[i])][int(predicted[i])] += 1
            total += current_batch_size
            correct += (predicted == labels).sum()
                
        print('----------Results----------')
        print(cfs_mat)
        acc = int(correct) / int(total)
        results['accuracy'].append(acc)
        print('Accuracy: %.2f%%' % (100 * acc))
            
        TPFP = cfs_mat.sum(0)
        TPFN = cfs_mat.sum(1)
        for idx, number in enumerate(subset):
            TP = cfs_mat[idx][idx]
            precision = TP / TPFP[idx] if TPFP[idx] != 0 else np.inf
            recall = TP / TPFN[idx] if TPFN[idx] != 0 else np.inf
            if len(results[number]) == 0:
                results[number] = [[precision], [recall]]
            else:
                results[number][0].append(precision)
                results[number][1].append(recall)
            print('%s,' % classes[number] + ' ' * (18 - len(classes[number])) + 'Precision: %.2f%%, Recall: %.2f%%' % (100 * precision, 100 * recall))
        TP = cfs_mat[subsize][subsize]
        precision, recall = TP / TPFP[subsize], TP / TPFN[subsize]
        if len(results['other']) == 0:
            results['other'] = [[precision], [recall]]
        else:
            results['other'][0].append(precision)
            results['other'][1].append(recall)
        print('Others,' + ' ' * 12 + 'Precision: %.2f%%, Recall: %.2f%%' % (100 * precision, 100 * recall))
        print('---------------------------')
        print()
    
    return results


if __name__ == "__main__":
    '''
    prefix = ''
    
    wholeset = list(range(6))
    subsets = list(combinations(wholeset, 2))
    subsets = random.sample(subsets, 3)
    
    PlotWeightedTest(subsets = subsets, prefix = prefix, fileName = 'plot1.jpg', save = True)
    PlotNotInTest(subsets = subsets, prefix = prefix, fileName = 'plot2.jpg', save = True)
    '''
    pass