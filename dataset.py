# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 12:22:55 2021
@author: QI YU
@email: yq123456leo@outlook.com
"""

import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class HARdataset(Dataset):
    def __init__(self, group, prefix = ''):
        self.classes = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 
                        'SITTING', 'STANDING', 'LAYING']
        self.X, self.y = load_dataset_group(group, prefix)
        self.y = self.y.astype(np.int64)
        self.labelDict = defaultdict(int)
        for label in self.y:
            self.labelDict[label] += 1
        
        self.num_samples = self.X.shape[0]
        self.group = group
        self.status = 'original'
        self.subset = 'None'
        self.target = 'None'
    
    def ShowData(self):
        print('-----Dataset-----')
        if self.status == 'original':
            for idx in range(6):
                print('Label %d => %s: %d' % (idx, self.classes[idx], self.labelDict[idx]))
        elif self.status == 'balanced':
            pass
        print('-----------------')
    
    def Balance(self, subset, target = 'None'):
        self.subset = subset
        subsize = len(subset)
        inputs, labels = self.X, self.y
        
        dataDict = defaultdict(list)
        for idx, label in enumerate(labels):
            dataDict[int(label)].append(inputs[idx].tolist())
        
        newinputs, newlabels = list(), list()
        
        # 1-stage classifier
        if target == 'None':
            for idx, target in enumerate(subset):
                num = len(dataDict[target])
                newinputs += dataDict[target]
                newlabels += [idx for i in range(num)]
        
            otherset = list(range(6))
            for target in subset:
                otherset.remove(target)
                num_out = int(len(newinputs) / subsize / len(otherset))
            for target in otherset:
                data = dataDict[target]
                random.shuffle(data)
                realdata = data[:num_out]
                newinputs += realdata
                newlabels += [subsize for i in range(len(realdata))]
                
            self.X, self.y = np.array(newinputs), np.array(newlabels, dtype = np.int64)
            
            self.labelDict = defaultdict(int)
            for label in self.y:
                if label == subsize:
                    self.labelDict['other'] += 1
                else:
                    self.labelDict[subset[label]] += 1
        
        # 2-stage classifier: 1st stage
        elif target == 'classifier1':
            otherset = tuple(i for i in range(6) if i not in subset)
            
            num_class = 2 * len(subset) if len(subset) <= len(otherset) else 2 * len(otherset)
            #total = self.num_samples / 6 * num_class
            total = self.num_samples / 6 * 2
            
            num_in_each = int(total / 2 / len(subset))
            num_out_each = int(total / 2 / len(otherset))
            for label in range(10):
                data = dataDict[label]
                random.shuffle(data)
                if label in subset:
                    realdata = data[:num_in_each]
                    newlabels += [0 for i in range(len(realdata))]
                else:
                    realdata = data[:num_out_each]
                    newlabels += [1 for i in range(len(realdata))]
                newinputs += realdata
            
            self.X, self.y = np.array(newinputs), np.array(newlabels, dtype = np.int64)
            
            self.labelDict = defaultdict(int)
            for label in self.y:
                if label == 0:
                    self.labelDict['in-dictionary'] += 1
                else:
                    self.labelDict['out-of-dictionary'] += 1
        
        # 2-stage classifier: 2nd stage
        elif target == 'classifier2':
            for idx, label in enumerate(subset):
                num = len(dataDict[label])
                newinputs += dataDict[label]
                newlabels += [idx for i in range(num)]
            
            self.X, self.y = np.array(newinputs), np.array(newlabels, dtype = np.int64)
            
            self.labelDict = defaultdict(int)
            for label in self.y:
                self.labelDict[subset[int(label)]] += 1
        
        self.num_samples = self.X.shape[0]
        self.target = target
        self.status = 'balanced'
    
    def ShapeWeight(self, subset, proportion):
        self.subset = subset
        inputs, labels = self.X, self.y
        
        dataDict = defaultdict(list)
        for idx, label in enumerate(labels):
            dataDict[int(label)].append(inputs[idx].tolist())
        
        otherset = tuple(i for i in range(6) if i not in subset)
        num_class = 2 * len(subset) if len(subset) <= len(otherset) else 2 * len(otherset)
        #total = self.num_samples / 6 * num_class
        total = self.num_samples / 6 * 2
        
        num_in_each = int(total * proportion / len(subset))
        num_out_each = int(total * (1 - proportion) / len(otherset))
        
        newinputs, newlabels = list(), list()
        for label in range(10):
            data = dataDict[label]
            random.shuffle(data)
            if label in subset:
                realdata = data[:num_in_each]
                newlabels += [0 for i in range(len(realdata))]
            else:
                realdata = data[:num_out_each]
                newlabels += [1 for i in range(len(realdata))]
            newinputs += realdata
            
        self.X, self.y = np.array(newinputs), np.array(newlabels, dtype = np.int64)
            
        self.labelDict = defaultdict(int)
        for label in self.y:
            if label == 0:
                self.labelDict['in-dictionary'] += 1
            else:
                self.labelDict['out-of-dictionary'] += 1
        
        self.num_samples = self.X.shape[0]
        self.target = 'classifier1'
        self.status = 'weighted (%d%% in-dictionary label)' % (100 * proportion)
    
    def ShapeNotIn(self, subset, num_other):
        self.subset = subset
        subsize = len(subset)
        inputs, labels = self.X, self.y
    
        dataDict = defaultdict(list)
        for idx, label in enumerate(labels):
            dataDict[int(label)].append(inputs[idx].tolist())
    
        newinputs, newlabels = list(), list()
        for idx, target in enumerate(subset):
            num = len(dataDict[target])
            newinputs += dataDict[target]
            newlabels += [idx for i in range(num)]
    
        otherset = list(range(6))
        for target in subset:
            otherset.remove(target)
        assert num_other <= len(otherset), 'Error: too many OUT-OF-DICTIONARY labels!'
        otherset = otherset[:num_other]
        num_out = int(len(newinputs) / subsize / num_other)
    
        for other in otherset:
            data = dataDict[other]
            random.shuffle(data)
            realdata = data[:num_out]
            newinputs += realdata
            newlabels += [subsize for i in range(len(realdata))]
    
        self.X, self.y = np.array(newinputs), np.array(newlabels)
        self.num_samples = self.X.shape[0]
        self.status = 'Not-In weighted'
        self.labelDict = defaultdict(int)
        for label in self.y:
            if label == subsize:
                self.labelDict['other'] += 1
            else:
                self.labelDict[subset[label]] += 1
    
    def __repr__(self):
        dataset = "=>Dataset: Human Activity Recognition\n"
        group = "=>Group: %sing set" % self.group + '\n'
        status = "=>Status: %s" % self.status + '\n'
        target = "=>Target: %s" % self.target + '\n'
        sample = "=>Number of samples: %d" % self.num_samples + '\n'
        subset = "=>Subset: %s" % str(self.subset)
        return dataset + group + status + target + sample + subset
        
    def __getitem__(self, index):
        inputs = self.X[index]
        label = self.y[index]
        return inputs, label
    
    def __len__(self):
        return self.num_samples

class SubDataset(Dataset):
    def __init__(self, subset, group, prefix = ''):
        X, y = load_dataset_group(group, prefix)
        
        labels = []
        subsize = len(subset)
        for label in y:
            if int(label) in subset:
                labels.append(subset.index(int(label)))
            else:
                labels.append(subsize)
        labels = np.array(labels, dtype = np.int64)
        
        self.X, self.y = X, labels
        self.num_samples = self.X.shape[0]
    
    def __getitem__(self, index):
        inputs = self.X[index]
        label = self.y[index]
        return inputs, label
    
    def __len__(self):
        return self.num_samples

def load_file(filepath):
    dataframe = pd.read_csv(filepath, header = None, delim_whitespace = True)
    return dataframe.values

# load a list of files into a 3D array of [samples, timesteps, features]
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
	# stack group so that features are the 3rd dimension
    loaded = np.dstack(loaded)
    order = [0, 2, 1]
    loaded = np.transpose(loaded, order)
    return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix = ''):
    filepath = prefix + 'UCI HAR Dataset/' + group + '/Inertial Signals/'
	# load all 9 files as a single array
    filenames = list()
	# total acceleration
    filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' + group + '.txt', 'total_acc_z_' + group + '.txt']
	# body acceleration
    filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group + '.txt', 'body_acc_z_' + group + '.txt']
	# body gyroscope
    filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' + group + '.txt', 'body_gyro_z_' + group + '.txt']
	# load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + 'UCI HAR Dataset/' + group + '/y_' + group + '.txt')
    y = y.reshape(y.shape[0])
    y = y - 1
    return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix = ''):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix)
    print(trainX.shape, trainy.shape)
    # load all test
    testX, testy = load_dataset_group('test', prefix)
    print(testX.shape, testy.shape)
    
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    
    return trainX, trainy, testX, testy

if __name__ == "__main__":
    '''
    prefix = ''
    
    subsets = list()
    for i in range(1, 6):
        subset = tuple(j for j in range(i))
        subsets.append(subset)
    print(subsets)
    
    X = np.linspace(5, 95, 19, dtype = int)
    for subset in subsets:
        print('-----' + str(subset)+ '-----')
        for x in X:
            train_data = HARdataset(group = 'train', prefix = prefix)
            train_data.Balance(subset = subset, target = 'classifier1')
            print("%d%% in-dictionary: %d" % (x, train_data.num_samples))
        print('-------------')
    '''
    
    train_data = HARdataset(group = 'train')
    test_data = HARdataset(group = 'test')
    train_data.Balance(subset = (0, 1, 2, 3, 4), target = 'classifier1')
    test_data.Balance(subset = (0, 1, 2, 3, 4), target = 'classifier1')