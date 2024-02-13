import sys
sys.path.append('./../')
import pandas as pd
import numpy as np
import os, random, itertools
# from utils import *
import sklearn.datasets as datasets
from functools import partial

class dataGenerate(object):
    """
    A class of functions for generating jsonl datasets for classification tasks.
    """
    def __init__(self, seed = 123):
        self.seed = 123

    def df2jsonl(self, df, filename, integer = False, 
                 context = False, feature_names = None, target_names = None, init = '', end = ''):
        jsonl = '\n'.join(df.apply(func = partial(self.data2text, 
                                                  integer = integer, 
                                                  context = context, 
                                                  feature_names = feature_names, 
                                                  target_names = target_names, 
                                                  init = init, 
                                                  end = end), axis = 1).tolist())
        fpath = os.path.join('data', filename)
        with open(fpath, 'w') as f:
            f.write(jsonl)
        return fpath
            
    def array2prompts(self, X, integer = False,
                     context = False, feature_names = None, target_names = None, init = '', end = ''):
        return list(map(partial(self.data2text, 
                                integer = integer, 
                                label = False,
                                context = context, 
                                feature_names = feature_names, 
                                target_names = target_names, 
                                init = init, 
                                end = end
                               ), X))
    
    def data_split(self, X, y):
        n = X.shape[0]
        idx = np.arange(n)
        random.shuffle(idx)
        train_idx, valid_idx, test_idx = idx[:int(.6*n)], idx[int(.6*n):int(.8*n)], idx[int(.8*n):]
        X_train, X_valid, X_test = X[train_idx], X[valid_idx], X[test_idx]
        y_train, y_valid, y_test = y[train_idx], y[valid_idx], y[test_idx]
        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def gridX_generate(self, X, resolution = 50):
        # lb = np.apply_along_axis(min, 0, X)
        # ub = np.apply_along_axis(max, 0, X)
        
        # lb, ub = lb - rang*0.2, ub + rang*0.2

        # X_grid = np.linspace(lb, ub, resolution).T
        # X_grid = np.array(list(itertools.product(*X_grid)))

        # h = 0.02
        lb = np.min(X, axis=0)[0]
        ub = np.max(X, axis=0)[0]
        rang = ub - lb
        h = rang/resolution
        xx, yy = np.meshgrid(np.arange(lb, ub, h),
                            np.arange(lb, ub, h))
        X_grid = np.c_[xx.ravel(), yy.ravel()]

        grid_prompts = self.array2prompts(X_grid)
        return pd.DataFrame(X_grid), grid_prompts