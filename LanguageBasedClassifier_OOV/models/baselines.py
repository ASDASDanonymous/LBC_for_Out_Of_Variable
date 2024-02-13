from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn import tree
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import  BaseEstimator
from collections import Counter
import numpy as np
import torch


def get_optimizer(model, optim):
    if optim == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            momentum=0.9, 
            lr=0.1, 
            weight_decay = 0.0001
            )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=0.001, 
            weight_decay = 0.0001
            )

    return optimizer


def get_lambda(alpha=1.0):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        if np.random.rand() <= 0.5:
            lam = 1.
        else:
            lam = 0.
    return lam
    

clf_model = {
    'svm': SVC(gamma='auto',class_weight='balanced',probability=True),
    'logreg': LogisticRegression(random_state=0),
    'knn': KNeighborsClassifier(n_neighbors=5),
    'tree': tree.DecisionTreeClassifier(),
    'xgboost': XGBClassifier(verbosity = 0, silent=True),
}

clf_model_teachers = {
    'svm': {'poly': make_pipeline(StandardScaler(), SVC(kernel='poly')),'rbf': make_pipeline(StandardScaler(),  SVC(kernel='rbf')), 'sigmoid': make_pipeline(StandardScaler(), SVC(kernel='sigmoid'))},
    'logreg': {'C=1.0':LogisticRegression(random_state=0, C=1.0)},
    'knn': {'K=1':KNeighborsClassifier(n_neighbors=1), 'K=3':KNeighborsClassifier(n_neighbors=3),'K=5': KNeighborsClassifier(n_neighbors=5)},
    'tree': {'d=3':tree.DecisionTreeClassifier(max_depth=3), 'd=5':tree.DecisionTreeClassifier(max_depth=5)},
    'xgboost': {'XGB':XGBClassifier(verbosity = 0, silent=True)},
}

param_grids = {
    'svm': {'kernel': ('linear', 'rbf'), 'C': [1, 10, 100]},
    'tree': {'criterion': ("gini", "entropy"), 'max_depth': [3, 5, 20]},
    'logreg': {'C': [1, 10, 100]},
    'knn': {'n_neighbors': [5, 1, 3], 'p': [1, 2]},
    'xgboost': {'max_depth': [3, 5, 10]}
}


def predict(clf, x_train, y_train, x_test):
    clf = clf.fit(x_train, y_train)
    return clf.predict(x_test)
