
# coding: utf-8

# ### imports

# In[ ]:


print ('Submitted By')
print ('UBITname      = karanman')
print ('Person Number = 50290755')


# In[ ]:


# importing packages
# A package is a collection/directory of python modules
import numpy as np
import pandas as pd
import sys
import os
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score


# ### Generate dataset

# In[ ]:


# There are 3 solutions for the given problem, one is linear regression solution, second is logistic regression solution and the third is neural network solution
# Opening and reading the csv files 
# taking two image pairs from different and similar pairs csv files conactenating the features
# taking two image pairs from different and similar pairs csv files subtracting the features
# csv file is a file with comma separated values
# code for merging features
def merge_features(pairs, feat):
    
        a = pairs.loc[:,'img_id_A'].values
        b = pairs.loc[:,'img_id_B'].values

        

        a_ = feat.loc[a]
        b_ = feat.loc[b]

        a_.reset_index(inplace=True)
        b_.reset_index(inplace=True)

        columns = a_.columns.values

        a_new_columns = []
        b_new_columns = []

        for column in columns:
            a_new_col = 'a_' + column
            b_new_col = 'b_' + column
            a_new_columns.append(a_new_col)
            b_new_columns.append(b_new_col)



        a_.columns = a_new_columns
        b_.columns = b_new_columns

        a_b = pd.concat([a_, b_], axis=1)

        b_id = a_b.loc[:, 'b_img_id']

        a_b.drop('b_img_id', axis=1, inplace=True)

        a_b = pd.concat([b_id, a_b], axis=1)

        a_id = a_b.loc[:, 'a_img_id']

        a_b.drop('a_img_id', axis=1, inplace=True)

        a_b = pd.concat([a_id, a_b], axis=1)

        return a_b


# In[ ]:


# subtracting the features of two images
def sub_features(pairs, feat):

    a = pairs.loc[:,'img_id_A'].values
    b = pairs.loc[:,'img_id_B'].values

    a_ = feat.loc[a]
    b_ = feat.loc[b]

    sub_df = a_.reset_index(drop=True).subtract(b_.reset_index(drop=True))

    sub_df['a_img_id'] = a_.index.values

    sub_df['b_img_id'] = b_.index.values

    sub_df = sub_df.reindex_axis(sorted(sub_df.columns), axis=1)

    return sub_df


# In[ ]:


# generating data with concatenation of features by using human observed data set
# generating data with subtraction of features by using human observed data set
# Opening and reading the csv file 
def genHumanDataSet():
    
    path = 'data/HumanObserved-Dataset/HumanObserved-Features-Data/'
    feat_path = os.path.join(path, 'HumanObserved-Features-Data.csv')  
    
    diff_pairs_path = os.path.join(path, 'diffn_pairs.csv')
    same_pairs_path = os.path.join(path, 'same_pairs.csv')
    
    feat = pd.read_csv(feat_path, index_col=0)
    diff_pairs = pd.read_csv(diff_pairs_path)
    same_pairs = pd.read_csv(same_pairs_path)
    
    feat.set_index('img_id', inplace=True)    
        
    same_pairs_df = merge_features(same_pairs, feat)
    diff_pairs_df = merge_features(diff_pairs, feat)
    
    same_pairs_df = pd.concat([same_pairs_df, same_pairs], axis=1)
    same_pairs_df.drop(['img_id_A', 'img_id_B'], axis=1, inplace=True)
    
    diff_pairs_df = pd.concat([diff_pairs_df, diff_pairs], axis=1)
    diff_pairs_df.drop(['img_id_A', 'img_id_B'], axis=1, inplace=True)
    
    human_final_df = pd.concat([same_pairs_df, diff_pairs_df])
    
    human_final_df.to_csv('human_concat_final_df.csv')  
    
    same_pairs_df = sub_features(same_pairs, feat)
    diff_pairs_df = sub_features(diff_pairs, feat)
    
    same_pairs_df = pd.concat([same_pairs_df, same_pairs], axis=1)
    same_pairs_df.drop(['img_id_A', 'img_id_B'], axis=1, inplace=True)
    
    diff_pairs_df = pd.concat([diff_pairs_df, diff_pairs], axis=1)
    diff_pairs_df.drop(['img_id_A', 'img_id_B'], axis=1, inplace=True)
    
    human_final_df = pd.concat([same_pairs_df, diff_pairs_df])
    
    human_final_df.to_csv('human_subtract_final_df.csv')


# In[ ]:


# generating data with concatenation of features by using GSC features data set
# generating data with subtraction of features by using GSC features data set
# Opening and reading the csv file 
def genGSCDataSet():
    path = 'data/GSC-Dataset/GSC-Features-Data/'
    feat_path = os.path.join(path, 'GSC-Features.csv')
    
    diff_pairs_path = os.path.join(path, 'diffn_pairs.csv')
    same_pairs_path = os.path.join(path, 'same_pairs.csv')
    
    feat = pd.read_csv(feat_path, index_col=0)
    diff_pairs = pd.read_csv(diff_pairs_path)
    same_pairs = pd.read_csv(same_pairs_path)
    
    same_pairs_df = merge_features(same_pairs, feat)
        
    same_pairs_df.to_csv('gsc_concat_final_df.csv')
    
    print("same pairs df shape : {}".format(same_pairs_df.shape))
    
    # split diff into 55k dfs
    start = 0
    step = 55000 # depends on available ram
    max_ = diff_pairs.shape[0]
    
    for i in range(1, int(math.floor(diff_pairs.shape[0]/step))+1):
        
        stop = i * step
        
        if stop > max_:
            stop = max_ + 1
        
        print('stop : {}'.format(stop))
        
        diff_pairs_part = diff_pairs.iloc[start:stop]
        
        start = stop
    
        diff_pairs_df = merge_features(diff_pairs_part, feat)
    
        diff_pairs_df = pd.concat([diff_pairs_df, diff_pairs_part], axis=1)
        diff_pairs_df.drop(['img_id_A', 'img_id_B'], axis=1, inplace=True)
    
        print("diff pairs df shape : {}".format(diff_pairs_df.shape))
                
        with open('gsc_concat_final_df.csv', 'a') as f:
            diff_pairs_df.to_csv(f, header=False,index=False)
        
        del diff_pairs_part
        del diff_pairs_df
        
        
    same_pairs_df = sub_features(same_pairs, feat)
        
    same_pairs_df.to_csv('gsc_subtract_final_df.csv')
    
    print("same pairs df shape : {}".format(same_pairs_df.shape))
    
    # split diff into 55k dfs
    start = 0
    step = 55000 # depends on available ram
    max_ = diff_pairs.shape[0]
    
    for i in range(1, int(math.floor(diff_pairs.shape[0]/step))+1):
        
        stop = i * step
        
        if stop > max_:
            stop = max_ + 1
        
        print('stop : {}'.format(stop))
        
        diff_pairs_part = diff_pairs.iloc[start:stop]
        
        start = stop
    
        diff_pairs_df = merge_features(diff_pairs_part, feat)
    
        diff_pairs_df = pd.concat([diff_pairs_df, diff_pairs_part], axis=1)
        diff_pairs_df.drop(['img_id_A', 'img_id_B'], axis=1, inplace=True)
    
        print("diff pairs df shape : {}".format(diff_pairs_df.shape))
                
        with open('gsc_subtract_final_df.csv', 'a') as f:
            diff_pairs_df.to_csv(f, header=False,index=False)
        
        del diff_pairs_part
        del diff_pairs_df


# In[ ]:


# generating the human-observed and GSC data set final file
genHumanDataSet()
genGSCDataSet()


# In[ ]:


chunksize = 55000
for chunk in pd.read_csv('gsc_concat_final_df.csv', chunksize=chunksize, engine='python'):
    print("shape: {}".format(chunk.shape))


# ### Linear regression

# In[ ]:


# linear regression solution
def linear_regression(dataset='human', mode='concat'):
    
    data = pd.read_csv('{}_{}_final_df.csv'.format(dataset, mode), index_col=[1,2])
    data.drop('Unnamed: 0', axis=1, inplace=True)

    X = data.drop('target', axis=1).as_matrix()
    y = data['target']
    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.33)

    y = data['target'].values

    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3)

    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.33)
    
    
    class LinearRegression(object):
        def __init__(self, lr=0.1, n_iter=50):
            self.lr = lr
            self.n_iter = n_iter

        def fit(self, X, y):
            X = np.insert(X, 0, 1, axis=1)
            self.w = np.ones(X.shape[1])
            m = X.shape[0]

            for _ in range(self.n_iter):
                output = X.dot(self.w)
                errors = y - output
                self.w += self.lr / m * errors.dot(X)
            return self

        def predict(self, X):
            return np.insert(X, 0, 1, axis=1).dot(self.w)

        def score(self, X, y):
            return 1 - sum((self.predict(X) - y)**2) / sum((y - np.mean(y))**2)
    
    # tuning the hyperparameters with these values
    if dataset == 'human':
        lrs = [0.01, 0.001]
        n_iters = [5, 20, 50, 100, 250]
    if dataset == 'gsc':
        lrs = [0.01, 0.001]
        n_iters = [5, 20, 50, 100, 250]


    def train_linr_model(lr, n_iter, mode='test'):


        print('\n--Hyperparameters--')
        print('learning rate : {}'.format(lr))
        print('no of iterations : {}'.format(n_iter))

        model = LinearRegression(lr=lr, n_iter=n_iter)

        model.fit(X_train, y_train)

        if mode == 'val':

            y_preds = model.predict(X_val)

            rmse = mean_squared_error(y_val, y_preds) ** 0.5

            y_preds = np.round(y_preds)

            acc = accuracy_score(y_val, y_preds)
            
            # printing accuracy and Root mean square error for each hyperparameter tuning
            print('rmse : {}'.format(rmse))
            print('acc : {}'.format(acc))

        if mode == 'test':

            y_preds = model.predict(X_test)

            rmse = mean_squared_error(y_test, y_preds) ** 0.5

            y_preds = np.round(y_preds)

            acc = accuracy_score(y_test, y_preds)

        return acc, rmse
    
    
    scores_df = pd.DataFrame()
    for lr in lrs:
        for n_iter in n_iters:
            acc, rmse = train_linr_model(lr, n_iter, mode='val')
            scores = pd.DataFrame([acc, rmse], index=['acc', 'rmse'], columns=[(lr, n_iter)]).T
            scores_df = pd.concat([scores_df, scores])
    
    best_params = scores_df['acc'].idxmax()
    lr = best_params[0]
    n_iter = best_params[1]
    
    print('\n--Training with best hyper-parameters--\n')

    acc, rmse = train_linr_model(lr, n_iter)

    print('--Scores on test set--')
    print('accuracy : {}'.format(acc))
    print('RMSE : {}'.format(rmse))

    
    


# In[ ]:


# performing linear regression solution on Human Observed/ concat data set
linear_regression(dataset='human', mode='concat')


# In[ ]:


# performing linear regression solution on Human Observed/ subtract data set
linear_regression(dataset='human', mode='subtract')


# In[ ]:


# Run The code below for tuning hyperparameters for the GSC data set
# this code is added as a comment because it takes a lot of time to tune the hyperparameters for GSC data set
# linear_regression(dataset='gsc', mode='concat')


# In[ ]:


# Run The code below for tuning hyperparameters for the GSC data set
# this code is added as a comment because it takes a lot of time to tune the hyperparameters for GSC data set
# linear_regression(dataset='gsc', mode='subtract')


# ### Logistic Regression

# In[ ]:


def logistic_regression(dataset='human', mode='concat'):
    
    data = pd.read_csv('{}_{}_final_df.csv'.format(dataset, mode), index_col=[1,2])
    data.drop('Unnamed: 0', axis=1, inplace=True)

    X = data.drop('target', axis=1).as_matrix()
    y = data['target'].values
    
    # splitting the data set
    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.33)
    
    class LogisticRegression(object):
        def __init__(self, lr, num_iter=100, fit_intercept=True, verbose=False):
            self.lr = lr
            self.num_iter = num_iter
            self.fit_intercept = fit_intercept
            self.verbose = verbose

        def __add_intercept(self, X):
            intercept = np.ones((X.shape[0], 1))
            return np.concatenate((intercept, X), axis=1)

        def __sigmoid(self, z):
            return 1 / (1 + np.exp(-z))
        def __loss(self, h, y):
            return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

        def fit(self, X, y):
            if self.fit_intercept:
                X = self.__add_intercept(X)

            # weights initialization
            self.theta = np.zeros(X.shape[1])

            for i in range(self.num_iter):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                gradient = np.dot(X.T, (h - y)) / y.size
                self.theta -= self.lr * gradient

                if(self.verbose == True and i % 10000 == 0):
                    z = np.dot(X, self.theta)
                    h = self.__sigmoid(z)
                    print(f'loss: {self.__loss(h, y)} \t')

        def predict_prob(self, X):
            if self.fit_intercept:
                X = self.__add_intercept(X)

            return self.__sigmoid(np.dot(X, self.theta))

        def predict(self, X, threshold=0.5):
            return self.predict_prob(X) >= threshold

    if dataset == 'human':
        lrs = [0.01, 0.001]
        n_iters = [5, 20, 50, 100, 250]
        
    if dataset == 'gsc':
        lrs = [0.01, 0.001]
        n_iters = [5, 20, 50, 100, 250]



    def train_logr_model(lr, n_iter, mode='test'):


        print('\n--Hyperparameters--')
        print('learning rate : {}'.format(lr))
        print('no of iterations : {}'.format(n_iter))

        model = LogisticRegression(lr=lr, num_iter=n_iter)

        model.fit(X_train, y_train)

        if mode == 'val':

            y_preds = model.predict(X_val)

            rmse = mean_squared_error(y_val, y_preds) ** 0.5

            acc = accuracy_score(y_val, y_preds)

            print('rmse : {}'.format(rmse))
            print('acc : {}'.format(acc))

        if mode == 'test':

            y_preds = model.predict(X_test)

            rmse = mean_squared_error(y_test, y_preds) ** 0.5

            acc = accuracy_score(y_test, y_preds)

        return acc, rmse
    
    
    scores_df = pd.DataFrame()
    for lr in lrs:
        for n_iter in n_iters:
            acc, rmse = train_logr_model(lr, n_iter, mode='val')
            scores = pd.DataFrame([acc, rmse], index=['acc', 'rmse'], columns=[(lr, n_iter)]).T
            scores_df = pd.concat([scores_df, scores])
    
    best_params = scores_df['acc'].idxmax()
    lr = best_params[0]
    n_iter = best_params[1]
    
    print('\n--Training with best hyper-parameters--\n')

    acc, rmse = train_logr_model(lr, n_iter, mode='test')

    print('--Scores on test set--')
    print('accuracy : {}'.format(acc))
    print('RMSE : {}'.format(rmse))

    
    


# In[ ]:


# performing logistic regression solution on Human Observed/ concat data set
logistic_regression(dataset='human', mode='concat')


# In[ ]:


# performing logistic regression solution on Human Observed/ subtract data set
logistic_regression(mode='subtract')


# In[ ]:


# Run The code below for tuning hyperparameters for the GSC data set
# this code is added as a comment because it takes a lot of time to tune the hyperparameters for GSC data set
# logistic_regression(dataset='gsc', mode='concat')


# In[ ]:


# Run The code below for tuning hyperparameters for the GSC data set
# this code is added as a comment because it takes a lot of time to tune the hyperparameters for GSC data set
# logistic_regression(dataset='gsc', mode='subtract')


# ### neural network

# In[ ]:


# importing packages
# A package is a collection/directory of python modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


# In[ ]:


# reading the final created human observed file
data = pd.read_csv('human_final_df.csv', index_col=[1,2], engine='c')
data.drop('Unnamed: 0', axis=1, inplace=True)
y = data['target'].values

# splitting the file
X_train, X_valtest, y_train, y_valtest = train_test_split(data, y, test_size=0.3)

X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.33)


# In[ ]:


X_train.to_csv('train_data.csv')
X_val.to_csv('val_data.csv')
X_test.to_csv('test_data.csv')


# In[ ]:


class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): path to csv file
        """
        self.data = pd.read_csv(csv_path)
        self.labels = np.asarray(self.data['target'])

    def __getitem__(self, index):
        single_label = self.labels[index]
        features = np.asarray(self.data.iloc[index, :-1])
        
        features_as_tensor = torch.tensor(features)
        return (features_as_tensor, single_label)

    def __len__(self):
        return len(self.data.index)


# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)
    
net = Net()
print(net)


# In[ ]:


# create a stochastic gradient descent optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# create a loss function
criterion = nn.NLLLoss()


# In[ ]:


datasetTrain = CustomDatasetFromCSV('train_data.csv')
datasetVal = CustomDatasetFromCSV('val_data.csv')
datasetTest = CustomDatasetFromCSV('test_data.csv')

trBatchSize = 16
epochs = 10

dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=24, pin_memory=True)
dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=24, pin_memory=True)
dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, shuffle=False, num_workers=24, pin_memory=True)


# In[ ]:


# run the main training loop
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(dataLoaderTrain):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        net_out = net(data)
        loss = criterion(net_out, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(dataLoaderTrain.dataset),
                           100. * batch_idx / len(dataLoaderTrain), loss.data[0]))

