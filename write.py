# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 20:03:11 2021
@author: QI YU
@email: yq123456leo@outlook.com
"""

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
from collections import defaultdict
from itertools import combinations

from inputs import LoadWeight
from dataset import HARdataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    cudnn.benchmark = True

def InitResultForm(attr, subsets, fileName = 'classifier1.xlsx', prefix = ''):
    target = attr['target']
    group = attr['group']
    num_points = attr['num_points']
    
    metrics_list = ['accuracy', 'in_precision', 'in_recall', 'out_precision', 'out_recall', 'auc']
    metrics_names = ['Accuracy', 'In-Precision', 'In-Recall', 'Out_Precision', 'Out_Recall', 'ROC AUC']
    subroute = 'results/subsets/'
    
    index_label = 'Percentage of IN-DICTIONARY Label (%)'
    X = np.linspace(5, 95, num_points)
    
    cmap = sns.light_palette("green", as_cmap = True)
    
    writer = pd.ExcelWriter(prefix + subroute + target + '/' + group + '/' + fileName, engine = 'xlsxwriter')
    index, columns = X, metrics_names
    for subset in subsets:
        data = np.ones((num_points, len(metrics_list)))
        blank_sheet = pd.DataFrame(data, index = index, columns = columns)
        blank_sheet = blank_sheet.style.background_gradient(cmap = cmap)
        blank_sheet.to_excel(writer, sheet_name = str(subset), index_label = index_label)
        worksheet = writer.sheets[str(subset)]
        worksheet.set_column('A:G', 20)
    
    writer.save()


def WriteFormOnce(results, attr, subset, subsets, prefix = '', fileName = 'p1.xlsx'):
    target = attr['target']
    group = attr['group']
    
    metrics_list = ['accuracy', 'in_precision', 'in_recall', 'out_precision', 'out_recall', 'auc']
    metrics_names = ['Accuracy', 'In-Precision', 'In-Recall', 'Out_Precision', 'Out_Recall', 'ROC AUC']
    index_label = 'Percentage of IN-DICTIONARY Label (%)'
    
    # Add results data
    subroute = 'results/subsets/'
    data_xlsx = pd.ExcelFile(prefix + subroute + target + '/' + group + '/' + fileName)
    
    datapoints = list(results.keys())
    #subsets = list(combinations(list(range(6)), len(subset)))
    othersheets = dict()
    for otherset in subsets:
        if otherset == subset:
            data_df = data_xlsx.parse(sheet_name = str(subset), header = None)
            data_mat = np.array(data_df.iloc[1:, 1:], dtype = float)
        else:
            other_df = data_xlsx.parse(sheet_name = str(otherset), header = None)
            other_mat = np.array(other_df.iloc[1:, 1:], dtype = float)
            other_df = pd.DataFrame(other_mat, index = datapoints, columns = metrics_names)
            othersheets[otherset] = other_df
    
    # Add result dara
    for idx1, point in enumerate(datapoints):
        res = results[point]
        #assert list(res.keys()) == metrics_list, 'Error: Missing Metrics!'
        for idx2, metric in enumerate(metrics_list):
            data_mat[idx1][idx2] = round(res[metric], 4)
    data_df = pd.DataFrame(data_mat, index = datapoints, columns = metrics_names)
    
    # Heatmap
    cmap = sns.light_palette("green", as_cmap = True)
    data_df = data_df.style.background_gradient(cmap = cmap)
    
    # Write results data
    writer = pd.ExcelWriter(prefix + subroute + target + '/' + group + '/' + fileName, engine = 'xlsxwriter')
    for otherset in subsets:
        if otherset == subset:
            data_df.to_excel(writer, sheet_name = str(subset), index_label = index_label)
        else:
            other_df = othersheets[otherset]
            other_df = other_df.style.background_gradient(cmap = cmap)
            other_df.to_excel(writer, sheet_name = str(otherset), index_label = index_label)
    
    for name in subsets:
        worksheet = writer.sheets[str(name)]
        worksheet.set_column('A:G', 20)
    
    writer.save()
    

def RunTest(model, subset, test_loader, conf_thres = 0.5):
    results = dict()
    otherset = tuple(i for i in range(6) if i not in subset)
    #classes = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
    
    with torch.no_grad():
        y_scores = np.array(list())
        y_pred = np.array(list())       # Label prediction => shape: (batch_size, )
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
                
            outputs = model(inputs.float())
            
            # Probabilities
            y_score = F.softmax(outputs, dim = 1).cpu().detach().numpy()
            y_score = y_score[:, 1]
            y_scores = np.hstack((y_scores, y_score)) if y_scores.shape[0] != 0 else y_score
            
            # Direct label predictions
            predicted = (y_score >= conf_thres).astype(int)
            y_pred = np.hstack((y_pred, predicted)) if y_scores.shape[0] != 0 else y_score
        
        print('----------Results----------')
        
        # Confusion Matrix
        y_true = test_loader.dataset.y
        cfs_mat = metrics.confusion_matrix(y_true, y_pred)
        print(cfs_mat)
        
        # Accuracy
        acc = metrics.accuracy_score(y_true, y_pred)
        results['accuracy'] = acc
        print('Accuracy: %.2f%%' % (100 * acc))
        print()
        
        # Precision & Recall
        precisions = metrics.precision_score(y_true, y_pred, average = None)
        recalls = metrics.recall_score(y_true, y_pred, average = None)
        results['in_precision'], results['in_recall'] = precisions[0], recalls[0]
        results['out_precision'], results['out_recall'] = precisions[1], recalls[1]
        print('In-Dictionary %s' % str(subset) + ' => Precision: %.2f%%, Recall: %.2f%%' % (100 * precisions[0], 100 * recalls[0]))
        print('Out-of-Dictionary %s' % str(otherset) + ' => Precision: %.2f%%, Recall: %.2f%%' % (100 * precisions[1], 100 * recalls[1]))
        
        # AUC
        auc = metrics.roc_auc_score(y_true, y_scores)
        results['auc'] = auc
        print('ROC AUC: %.4f' % auc)
        print('---------------------------')
        print()
    
    return results

def GenerateForm(attr, subsets, prefix = ''):
    target = attr['target']
    group = attr['group']
    num_points = attr['num_points']
    try:
        conf_thres = attr['conf_thres']
        ckpt_group = group[:14]
    except:
        conf_thres = 0.5
        ckpt_group = group
    
    X = np.linspace(5, 95, num_points, dtype = int)
    
    InitResultForm(attr, subsets, fileName = '%s.xlsx' % target, prefix = prefix)
    for subset in subsets:
        weightname = '%s_ckpt.pth' % str(subset)
        model = LoadWeight(1, target + '/' + ckpt_group + '/' + weightname, prefix)
        res = dict()
        
        for x in X:
            test_data = HARdataset(group = 'test', prefix = prefix)
            test_data.ShapeWeight(subset = subset, proportion = x / 100)
            test_loader = DataLoader(dataset = test_data, batch_size = 64)
            
            results = RunTest(model, subset, test_loader, conf_thres)
            res[x] = results
        
        WriteFormOnce(res, attr, subset, subsets, prefix, fileName = '%s.xlsx' % target)
    
    return True

if __name__ == "__main__":
    prefix = ''
    attr = {'target': 'classifier1', 'group': 'fixed training', 'num_points': 19}
    
    wholeset = list(range(6))
    conb2 = list(combinations(wholeset, 2))
    subsets1 = conb2[:5]
    subsets2 = list()
    for i in range(3, 5):
        subset = tuple(range(i))
        subsets2.append(subset)
    subsets = subsets1 + subsets2
    
    print(subsets)
    flag = GenerateForm(attr, subsets, prefix)
    print(flag)
    
    
    '''
    prefix = '/content/gdrive/MyDrive/week2/HAR+1DCNN/'
    
    subset = (0, 1)
    subsize = len(subset)
    
    subroute = 'fixed training/'
    weightName = '%s_ckpt.pth' % str(subset)
    
    test_data = HARdataset(group = 'test', prefix = prefix)
    test_data.Balance(subset = subset)
    test_loader = DataLoader(dataset = test_data, batch_size = 64)
    
    model = LoadWeight(subsize = subsize, fileName = subroute + weightName, prefix = prefix)
    results = RunTest(model = model, subset = subset, test_loader = test_loader)
    '''
    
    
    