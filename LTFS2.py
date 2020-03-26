import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

os.chdir("D:\\Narendra\\LTFS2\\")
print(os.getcwd())

train_data = pd.read_csv("Train.csv")
test_data = pd.read_csv("Test.csv")

train_data['date_index'] = pd.to_datetime(train_data['application_date'])
train_data.set_index('date_index')

train_summary = train_data.groupby(['segment', 'date_index'])['case_count'].agg([('sum', 'sum')])

train_x = train_summary.loc[1, 'sum']

plot_acf(train_x.values)
plot_pacf(train_x.values)


def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
 
# invert differenced forecast
def inverse_difference(last_ob, value):
	return value + last_ob


de_seasonal_train_x = pd.Series(difference(train_x, 365))
plot_acf(de_seasonal_train_x)
plot_pacf(de_seasonal_train_x)

n = len(de_seasonal_train_x)

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(de_seasonal_train_x[:400], order = (1,0,1))
results_ARMA = model.fit(disp = 5)
print(results_ARMA.summary())

predictions = pd.Series(model.predict(de_seasonal_train_x))
predictions.plot()

results_ARMA.plot_predict()
results_ARMA.resid