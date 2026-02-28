import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error,r2_score


def MAE(pred, true):
    return mean_absolute_error(true,pred)

def MSE(pred, true):
    return mean_squared_error(true,pred)

def RMSE(pred, true):
    return np.sqrt(mean_squared_error(true, pred))

def MAPE(pred, true):
    return mean_absolute_percentage_error(true,pred)

def R2(pred, true):
    return r2_score(true,pred)



def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    r2 = R2(pred, true)

    return mae, mse, rmse, mape, r2
