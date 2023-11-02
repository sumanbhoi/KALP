from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import warnings
import dill
from collections import Counter
warnings.filterwarnings('ignore')

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

# use the same metric from DMNC
def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

# def transform_split(X, Y):
#     x_train, x_eval, y_train, y_eval = train_test_split(X, Y, train_size=2/3, random_state=1203)
#     x_eval, x_test, y_eval, y_test = train_test_split(x_eval, y_eval, test_size=0.5, random_state=1203)
#     return x_train, x_eval, x_test, y_train, y_eval, y_test


def regression_metrics(y_gt, y_pred):
    # mse = mean_squared_error(y_gt, y_pred)
    def rmseval(y_gt, y_pred):
        all_mse = []
        for b in range(len(y_gt)):
            all_mse.append(mean_squared_error(y_gt[b], y_pred[b]))
        return np.mean(all_mse)

    def maeval(y_gt, y_pred):
        all_mae = []
        for b in range(len(y_gt)):
            all_mae.append(mean_absolute_error(y_gt[b], y_pred[b]))
        return np.mean(all_mae)

    def mapeval(y_gt, y_pred):
        all_mape = []
        for b in range(len(y_gt)):
            all_mape.append(mean_absolute_percentage_error(y_gt[b], y_pred[b]))
        return np.mean(all_mape)

    rmse = rmseval(y_gt, y_pred)
    mae = maeval(y_gt, y_pred)
    mape = mapeval(y_gt, y_pred)

    return rmse, mae, mape
