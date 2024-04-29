import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tools.eval_measures import mse, rmse, meanabs
from statsmodels.tsa.stattools import acf


def evaluate(train, test, forecasts):

    errors = forecasts - test

    mean_squared_error = mse(test, forecasts)
    root_mean_squared_error = rmse(test, forecasts)
    mean_absolute_error = meanabs(test, forecasts)

    mpe = np.mean((errors / test) * 100)
    mape = np.mean(np.abs(errors / test) * 100)

    naive_errors = train.shift(1).dropna() - train.dropna()
    denom = np.mean(np.abs(naive_errors))

    mase = mean_absolute_error / denom
        
    autocorrelation = acf(errors, fft=True, nlags=1)
    acf1 = autocorrelation[1]

    print("MSE:", mean_squared_error)
    print("RMSE:", root_mean_squared_error)
    print("MAE:", mean_absolute_error)
    print("MPE:", mpe)
    print("MAPE:", mape)
    print("MASE:", mase)
    print("ACF1:", acf1)



