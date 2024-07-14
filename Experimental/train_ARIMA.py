'''
# line plot of time series
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
import numpy 

# load dataset
series = read_csv('SPY_intraday.csv', header=0, index_col=0)
close = series[['close']]

# display first few rows
print(series.head(20))
# line plot of dataset
#close.plot()
#pyplot.show()

# split the dataset
split_point = round(len(series) * 0.8)
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
#dataset.to_csv('dataset.csv', index=False)
#validation.to_csv('validation.csv', index=False)

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return numpy.array(diff)
    
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
    
# seasonal difference
X = dataset.values
#days_in_year = 365
#differenced = difference(X, days_in_year)
# fit model
#model = ARIMA(differenced, order=(7,0,1))
model = ARIMA(X, order=(7,0,1))

model_fit = model.fit()
# print summary of fit model
print(model_fit.summary())
'''

from pmdarima import auto_arima
import pandas as pd
from matplotlib import pyplot
import numpy as np

stockprices = pd.read_csv('SPY_daily.csv')
stockprices = stockprices.sort_values(by=["timestamp"], ascending=False)
#stockprices.timestamp = pd.to_datetime(stockprices.timestamp)
#stockprices = stockprices.set_index('timestamp')

lag_features = ["high", "low", "volume"]
stockprices_rolled_3d = stockprices[lag_features].rolling(window=7, min_periods=0)
stockprices_mean_3d = stockprices_rolled_3d.mean().shift(1).reset_index().astype(np.float32)
stockprices_std_3d = stockprices_rolled_3d.std()
stockprices_rolled_7d = stockprices[lag_features].rolling(window=7, min_periods=0)
stockprices_mean_7d = stockprices_rolled_7d.mean().shift(1).reset_index().astype(np.float32)
stockprices_std_7d = stockprices_rolled_7d.std()
stockprices_rolled_30d = stockprices[lag_features].rolling(window=7, min_periods=0)
stockprices_mean_30d = stockprices_rolled_30d.mean().shift(1).reset_index().astype(np.float32)
stockprices_std_30d = stockprices_rolled_30d.std()

stockprices = pd.DataFrame({'timestamp':stockprices.timestamp, 'open':stockprices.open, 'high':stockprices.high, 'low':stockprices.low, 'close':stockprices.close, 'high_mean_3d':stockprices_mean_3d["high"], 'low_mean_3d':stockprices_mean_3d["low"], 'volume_mean_3d':stockprices_mean_3d["volume"], 'high_mean_7d':stockprices_mean_7d["high"], 'low_mean_7d':stockprices_mean_7d["low"], 'volume_mean_7d':stockprices_mean_7d["volume"], 'high_mean_30d':stockprices_mean_30d["high"], 'low_mean_30d':stockprices_mean_30d["low"], 'volume_mean_30d':stockprices_mean_30d["volume"], 'high_std_3d':stockprices_std_3d["high"], 'low_std_3d':stockprices_std_3d["low"], 'volume_std_3d':stockprices_std_3d["volume"], 'high_std_7d':stockprices_std_7d["high"], 'low_std_7d':stockprices_std_7d["low"], 'volume_std_7d':stockprices_std_7d["volume"],'high_std_30d':stockprices_std_30d["high"], 'low_std_30d':stockprices_std_30d["low"], 'volume_std_30d':stockprices_std_30d["volume"]})


exo = ['high_mean_3d', 'low_mean_3d', 'volume_mean_3d', 'high_mean_7d', 'low_mean_7d', 'volume_mean_7d', 'high_mean_30d', 'low_mean_30d', 'volume_mean_30d',]

test_ratio = 0.2
training_ratio = 1 - test_ratio

train_size = int(training_ratio * len(stockprices))
test_size = int(test_ratio * len(stockprices))

print(f"train_size: {train_size}")
print(f"test_size: {test_size}")

df_train = stockprices[:train_size]
df_test = stockprices[train_size:]
print(df_train.head(20))

model = auto_arima(
	df_train["close"],
    exogenous=df_train[exo],
	trace=True,
	error_action="ignore",
	suppress_warnings=True)
    
forecast = model.predict(n_periods=len(df_test), exogenous=df_test[exo])  
print(forecast)

df_test["close"].plot()
df_train["close"].plot()
forecast.plot()
pyplot.show()