import os
import sys
from typing import List
sys.path.append("./../")
import numpy as np
# from baselines import clf_model
import configs as cfgs
from helper import log
from classification_data_generator import DataGenerator, dfprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from feature_names import df2jsonl_feat_name, df2jsonl_feat_name_icl
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import pdb

# ####### Data
def prepare_data(did, OOV):
    data_gen = DataGenerator(did)
    d_n = {'Blood':'blood', 'Breast_Cancer':'breast_cancer', 'Creditcard':'creditcard', 'German':'german', 'ILPD':'ilpd', 'Loan':'loan', 'Salary':'salary', 'Steel_Plate':'steel_plate'}

    y, y_class, X_raw, X_norm, att_names = dfprocessing(did, d_n=d_n)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    # from IPython import embed; embed()
    count = 0
    for dev_index, test_index in sss.split(X_raw, y):
        assert count ==0
        X_raw_dev, X_raw_test = X_raw[dev_index], X_raw[test_index]
        y_dev, y_test = y[dev_index], y[test_index]
        X_norm_dev, X_norm_test = X_norm[dev_index], X_norm[test_index]
        count +=1

    # n_dev = len(dev_index)
    # train_index  = train_index[:int(.8*n_dev)]
    # val_index =  train_index[int(.8*n_dev):]
    # y_train, y_val, y_test =  y[train_index], y[val_index], y[test_index]
    # X_raw_train, X_raw_val, X_raw_test = X_raw[train_index], X_raw[val_index], X_raw[test_index]
    # X_norm_train, X_norm_val, X_norm_test = X_norm[train_index], X_norm[val_index], X_norm[test_index]
    # data = {'y_train': y_train, 'y_val': y_val, 'y_test': y_test, 'X_raw_train': X_raw_train, 'X_raw_test': X_raw_test, 'X_raw_val': X_raw_val, 'X_norm_test': X_norm_test, 'X_norm_val': X_norm_val, 'X_norm_train': X_norm_train, 'att_names': att_names}
    data = {'y_dev': y_dev, 'y_test': y_test, 'X_raw_dev': X_raw_dev, 'X_raw_test': X_raw_test, 'X_norm_test': X_norm_test, 'X_norm_dev': X_norm_dev}
   
    np.save(os.path.join(os.path.dirname(__file__),f'../data/{did}_dev_test_split'), data)

    # convert to prompt
    count = 0
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    for train_index, val_index in sss.split(X_raw_dev, y_dev):
        assert count ==0
        train_index, val_index = train_index, val_index
        count += 1
    y_train, y_val = y_dev[train_index], y_dev[val_index]
    X_raw_train, X_raw_val = X_raw_dev[train_index], X_raw_dev[val_index]
    X_norm_train, X_norm_val = X_norm_dev[train_index], X_norm_dev[val_index]

    # save datasets
    data_i = {'y_train': y_train, 'y_val': y_val, 'y_test': y_test, 'X_raw_train': X_raw_train, 'X_raw_test': X_raw_test,
        'X_raw_val': X_raw_val, 'X_norm_test': X_norm_test, 'X_norm_val': X_norm_val, 'X_norm_train': X_norm_train, 'att_names': att_names}

    np.save(os.path.join(os.path.dirname(__file__), f'../data/{did}'), data_i)

    train_df, val_df, test_df = pd.DataFrame(X_raw_train), pd.DataFrame(X_raw_val), pd.DataFrame(X_raw_test)
    def get_quartiles(df:pd.DataFrame):
        QUARTILES = (0.25, 0.5, 0.75)
        qdf = df.copy()
        cols = []
        for idx_column in range(qdf.shape[1]):
            if qdf.iloc[:,idx_column].dtype != 'object':
                cols.append(qdf.iloc[:,idx_column].quantile(QUARTILES))
            else:
                cols.append([pd.NA, pd.NA, pd.NA])
        return qdf
    
    train_df = train_df.apply(pd.to_numeric, errors='ignore')
    val_df = val_df.apply(pd.to_numeric, errors='ignore')
    test_df = test_df.apply(pd.to_numeric, errors='ignore')
    train_quartiles = get_quartiles(train_df)
    test_quartiles = get_quartiles(test_df)
    #Choose OOV and remove selected OOV in train data
    
    n_cols = X_raw_train.shape[1]
    n_cols_to_remove = int(n_cols * (OOV / 100))
    # cols_to_remove = np.random.choice(n_cols, n_cols_to_remove, replace=False) #random
    cols_to_remove = [] #determine
    if OOV:
        train_df = pd.DataFrame(X_raw_train)
        train_df.iloc[:, cols_to_remove] = None #all value is changed to None
        print("Deleted columns indices ({}%):".format(OOV), cols_to_remove)
        print(train_df.head())


    train_df['y'], val_df['y'], test_df['y'] = y_train, y_val, y_test
    dfs = {'train': train_df, 'val': val_df, 'test': test_df}

    # target_names = att_names[-1] if did > 10 else None
    # feature_names = att_names[:-1] if did > 10 else None
    fname = f"{did}"
    jsonl_files = {}
    icl = True
    
    df2jsonl_func = df2jsonl_feat_name_icl if icl else df2jsonl_feat_name
    
    for mode in ['train', 'val', 'test']:
        fname_prefixes:List[str] = [fname, mode, str(icl)]
        json_name = '_'.join(fname_prefixes + ["feature_names.jsonl"])
        jsonl_files[mode] = df2jsonl_func(dfs[mode], json_name, did, train_quartiles, test_quartiles)
        # if did == 'Blood':
        #     json_name = f'{fname}_{mode}_feature_names.jsonl'
        #     jsonl_files[mode] = df2jsonl_feat_name(dfs[mode], json_name, did, train_quartiles, test_quartiles)
        # elif did == 'Breast_Cancer':
        #     json_name = f'{fname}_{mode}_feature_names.jsonl'
        #     jsonl_files[mode] = df2jsonl_feat_name(dfs[mode], json_name, did, train_quartiles, test_quartiles)    
        # elif did == 'Creditcard':
        #     json_name = f'{fname}_{mode}_feature_names.jsonl'
        #     jsonl_files[mode] = df2jsonl_feat_name(dfs[mode], json_name, did,train_quartiles, test_quartiles)
        # elif did == 'German':
        #     json_name = f'{fname}_{mode}_feature_names.jsonl'
        #     jsonl_files[mode] = df2jsonl_feat_name(dfs[mode], json_name, did,train_quartiles, test_quartiles)
        # elif did == 'ILPD':
        #     json_name = f'{fname}_{mode}_feature_names.jsonl'
        #     jsonl_files[mode] = df2jsonl_feat_name(dfs[mode], json_name, did,train_quartiles, test_quartiles)
        # elif did == 'Loan':
        #     json_name = f'{fname}_{mode}_feature_names.jsonl'
        #     jsonl_files[mode] = df2jsonl_feat_name(dfs[mode], json_name, did,train_quartiles, test_quartiles)
        # elif did == 'Salary':
        #     json_name = f'{fname}_{mode}_feature_names.jsonl'
        #     jsonl_files[mode] = df2jsonl_feat_name(dfs[mode], json_name, did,train_quartiles, test_quartiles)
        # elif did == 'Steel_Plate':
        #     json_name = f'{fname}_{mode}_feature_names.jsonl'
        #     jsonl_files[mode] = df2jsonl_feat_name(dfs[mode], json_name, did,train_quartiles, test_quartiles)

    print('Done', did)

if __name__ == '__main__':
    for did in ['Creditcard', 'Loan', 'Steel_Plate']:
        prepare_data(did,0) #prepare_data(did, the percent(%) of OOV)
        