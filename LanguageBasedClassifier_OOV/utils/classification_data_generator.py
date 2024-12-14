import sys
sys.path.append('./../')
import os, random, itertools
from attr import attributes
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from functools import partial
import openai, openml
from einops import rearrange, repeat
from scipy.io import arff

import configs as cfgs

def data_split(X, y):
	if len(set(y)) == 2:
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
	else:
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
	# n = X.shape[0]
	# idx = np.arange(n)
	# random.shuffle(idx)
	# train_idx, valid_idx, test_idx = idx[:int(.6*n)], idx[int(.6*n):int(.8*n)], idx[int(.8*n):]
	# X_train, X_valid, X_test = X[train_idx], X[valid_idx], X[test_idx]
	# y_train, y_valid, y_test = y[train_idx], y[valid_idx], y[test_idx]
	return X_train, X_valid, X_test, y_train, y_valid, y_test
	
import pandas as pd
from scipy.io import arff
import os

def origin2df(dataname):
    base_path = os.path.join(os.path.dirname(__file__), '..', 'origin_dataset')
    file_path = os.path.join(base_path,dataname)

    if os.path.exists(file_path + '.arff'):
        path = file_path + '.arff'
        data, meta = arff.loadarff(path)
        df = pd.DataFrame(data)
    elif os.path.exists(file_path + '.csv') or os.path.exists(file_path + '.data'):
        path = file_path + '.csv'
        df = pd.read_csv(path)
    else:
        print(os.path.abspath(file_path))
        print(file_path)
        raise FileNotFoundError(f"No file found for {dataname} with .arff or .csv extension.")

    df.dropna(how='any', inplace=True)
    for col in df.select_dtypes([object]):
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    X, y = df.iloc[:5000, :-1], df.iloc[:5000, -1]

    y = y.astype("category")
    attribute_names = list(df.columns)[:-1]
    categorical_indicator = [df[c].dtype == 'object' for c in attribute_names]

    return X, y, categorical_indicator, attribute_names


def dfprocessing(did=-1, d_n=None, ignore_cat=False, convert_cat=False):
	X, y, categorical_indicator, attribute_names = origin2df(d_n[did])
	# preprocess
	Xy = pd.concat([X, y], axis=1, ignore_index=True)  # X & y concatenated together
	Xy.to_csv()
	if ignore_cat:
		# non-cat
		non_categorial_indices = np.where(np.array(categorical_indicator) == False)[
			0]  # find where categorical columns are
		Xy = Xy.iloc[:,
				[*non_categorial_indices, -1]]  # Slice columns -- ignore categorical X columns and add y column (-1)
		attribute_names = [attribute_names[i] for i in non_categorial_indices]

	Xy = Xy[Xy.iloc[:, -1].notna()]  # remove all the rows whose labels are NaN

	y_after_NaN_removal = Xy.iloc[:, -1]
	# Xy.dropna(axis=1, inplace=True)  # drop all the columns with missing entries
	# print(Xy)
	# Xy.dropna(inplace=True)  # drop all the rows with missing entries
	# print(Xy)
	X = Xy.iloc[:, :-1] # add this line bc nan error

	assert ((Xy.iloc[:, -1] == y_after_NaN_removal).all())

	X_raw, y = Xy.iloc[:, :-1], Xy.iloc[:, -1]
	# fine the categorical
	categorial_indices = np.where(np.array(categorical_indicator) == True)[0]
	scaler = StandardScaler()
	if len(categorial_indices) > 0:
		enc = OneHotEncoder(handle_unknown='ignore')
		# Slice columns -- ignore categorical X columns and add y column (-1)
		X_cat = X.iloc[:, [*categorial_indices]]
		X_cat_new = pd.DataFrame(enc.fit_transform(X_cat).toarray())
		X_cat_new = X_cat_new.values
		noncat_indices = np.where(np.array(categorical_indicator) == False)[0]
	
		if len(noncat_indices) > 0:
			X_noncat = X.iloc[:, [*noncat_indices]]
			X_noncat = scaler.fit_transform(X_noncat)
			X_norm = np.concatenate([X_noncat, X_cat_new], axis=1)
		else:
			X_norm = X_cat_new
	
	else:
			X_norm = scaler.fit_transform(X_raw)

	y = y.cat.codes.values

	return y, y_after_NaN_removal, X_raw.values, X_norm, attribute_names

# +
class DataGenerator(object):
	"""
	A class of functions for generating jsonl datasets for classification tasks.
	"""
	def __init__(self, did, seed = 123):
		self.seed = seed
		self.did = did
		self.fname = f'{did}'
		self.scaler = StandardScaler()

	def preprocess_data(self, data,  normalized=False, corruption_level=0, outliers=None):
		X, y = data['data'], data['target']
		if normalized:
			X = self.scaler.fit_transform(X)
		
		X_train, X_valid, X_test, y_train, y_valid, y_test = data_split(X, y)
		if outliers is not None:
			X_out, y_out = outliers
			X_train = np.concatenate([X_train, X_out], axis = 0)
			y_train = np.concatenate([y_train, y_out], axis = 0)
		if corruption_level > 0:
			# corrupt here
			n = len(y_train)
			m = int(n * corruption_level)
			inds = random.sample(range(1, n), m)
			for i in inds:
				y_train[i] = 1 - y_train[i] #binary
		
		train_df, val_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)
		train_df['y'], val_df['y'], test_df['y'] = y_train, y_valid, y_test   

		return train_df, val_df, test_df

	# def prepare_prompts(self, fnames, dfs, context=False, init=None, end=None, feature_names=None, target_names=None):
	# 	X_test = dfs['test'].values[:, :-1]
	# 	jsonl_files = {}
	# 	for mode in ['train', 'val']:
	# 		jsonl_files[mode] = df2jsonl(dfs[mode], fnames[mode],
	# 					context = context, 
	# 					feature_names = feature_names, 
	# 					target_names = target_names, 
	# 					init = init, 
	# 					end = end)
	# 	test_prompts = array2prompts(X_test,
	# 		context = context, 
	# 		feature_names = feature_names,
	# 		target_names = target_names, 
	# 		init = init, 
	# 		end = end)
		
	# 	return jsonl_files, test_prompts