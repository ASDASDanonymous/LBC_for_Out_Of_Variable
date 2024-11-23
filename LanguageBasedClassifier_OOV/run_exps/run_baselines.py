import sys
sys.path.append('')
import numpy as np
import argparse
from utils.helper import log
from utils.classification_data_generator import DataGenerator
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
# from sklearn.model_selection import HalvingRandomSearchCV
import warnings
warnings.filterwarnings("ignore")
import os
import torch
import pandas as pd
import random
import matplotlib.pyplot as plt
from utils.prepare_data import prepare_data
from models.baselines import clf_model, param_grids
import utils.configs as cfgs
from utils.classificationDataGen import dataGenerate
from utils.train_nn_generator import DataNN
from utils.helper import add_ending_symbol

parser = argparse.ArgumentParser(description='Baselines')
parser.add_argument("-d", "--data", default=-1, type=int)
parser.add_argument("-c", "--corrupted", default=0, type=float)
parser.add_argument("-t", "--task", default='accuracy', type=str)
parser.add_argument("-m", "--mixup", default=0, type=int)

######  Load configurations ##############
args = parser.parse_args()


# model_names = ['majorguess', 'logreg', 'knn', 'tree', 'nn', 'svm', 'rf', 'xgboost']
model_names = ['tree', 'knn','logreg','svm','xgboost']

if args.mixup:
    raise NotImplementedError
    model_names = ['nnmixup']#['random', 'logreg', 'knn', 'tree', 'nn', 'svm', 'rf', 'xgboost', 'nn_mixup']
# model_names = ['nn']
# eval_prefix = 'results/evals/rerun_all_acc_clf_baselines'
# csv_prefix = 'results/csv/rerun_all_acc_clf_baselines'
eval_prefix = ''
csv_prefix = ''

done_keys = []
sorted_done_key = 0
######## Data
# data_lst = cfgs.synthetic_data_ids.values()
data_lst =  cfgs.openml_data_ids.keys()
data_ids = [args.data] if args.data > -1 else data_lst
lst_ids = []
for k in data_ids:
    if k not in done_keys and k > sorted_done_key:
        lst_ids.append(k)
data_ids = np.sort(lst_ids)

def load_baseline_data(did):
    fname = f'.../{did}_dev_test.npy'
    # if not os.path.isfile(fname):
    #     print('prepare data', did)
    #     prepare_data(did,0)
    
    ith_fname = f'.../data/{did}.npy'
    data = np.load(ith_fname, allow_pickle=True)
    data = data.item()

    # Indexes to exclude (OOV)
    exclude_indices = []

    # Filter out the columns (features) to exclude
    X_train = np.delete(data['X_norm_train'], exclude_indices, axis=1)
    X_val = np.delete(data['X_norm_val'], exclude_indices, axis=1)
    X_test = np.delete(data['X_norm_test'], exclude_indices, axis=1)
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
    return X_train, y_train, X_val, y_val, X_test, y_test


from sklearn.metrics import f1_score, roc_auc_score

def get_acc_f1_auc(clf, X_train, y_train, X_val, y_val, X_test, y_test):
    
    clf.fit(X_train, y_train)
        
    # Predictions
    y_pred_test = clf.predict(X_test)
    y_pred_val = clf.predict(X_val)

    # Probabilities
    y_pred_test_prob = clf.predict_proba(X_test)[:,1]
    y_pred_val_prob = clf.predict_proba(X_val)[:,1]

    # Accuracy
    val_acc = round((y_pred_val == y_val).mean() * 100, 2)
    test_acc = round((y_pred_test == y_test).mean() * 100, 2)

    # F1 Score
    val_f1 = round(f1_score(y_val, y_pred_val, average='weighted'),2)
    test_f1 = round(f1_score(y_test, y_pred_test, average='weighted'),2)

    # AUC Score
    val_auc = round(roc_auc_score(y_val, y_pred_val_prob),2)
    test_auc = round(roc_auc_score(y_test, y_pred_test_prob),2)
    
    return val_acc, test_acc, test_f1, test_auc


############## Running ####################
log_fpath = f'{eval_prefix}_{args.task}.txt'
csv_fpath = f'{csv_prefix}_{args.task}.csv'
logf = open(log_fpath, 'a+')

results = []
for data_id in data_ids:
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_baseline_data(int(data_id))
        X = np.concatenate([X_train, X_val], axis=0)
        y = np.concatenate([y_train, y_val], axis=0)
        data_accuracies = []
        for clf_name in model_names:
            if clf_name in ['random', 'majorguess']:
                best_clf = clf_model[clf_name]
            else:
                base_estimator = clf_model[clf_name]
                print("We are here...")
                search = HalvingGridSearchCV(base_estimator, param_grids[clf_name], random_state=0).fit(X, y)
                best_clf = search.best_estimator_

            best_val = -1
            best_test = -1

            X_train, y_train, X_val, y_val, X_test, y_test = load_baseline_data(int(data_id))

            val_acc, test_acc, test_f1, test_auc = get_acc_f1_auc(best_clf, X_train, y_train, X_val, y_val, X_test, y_test)
            log(logf, f"{data_id} {clf_name} Binary: Acc={test_acc}, F1={test_f1}, AUC={test_auc}")
            
            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc
            acc = best_test
            message = f"{data_id} {clf_name} {acc}"
            # Print the summary for each classifier and dataset
            print(message)
            data_accuracies.append(acc)
        results.append(data_accuracies)
    except Exception as e:
        print('Error ', data_id, e)


np.savetxt(csv_fpath, np.asarray(results))


