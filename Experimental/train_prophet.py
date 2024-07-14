import pandas as pd
import numpy as np
from prophet import Prophet
import plotly
import matplotlib.pyplot as plt
from prophet.plot import plot_plotly, plot_components_plotly

stockprices = pd.read_csv('ALAB_intraday.csv')
stockprices['timestamp'] = pd.to_datetime(stockprices['timestamp'])

'''
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
stockprices_days = stockprices['timestamp'].dt.day_name()
stockprices_date = stockprices['timestamp'].dt.date
stockprices_hour = stockprices['timestamp'].dt.hour

print(list(stockprices))

stockprices = pd.DataFrame({'timestamp':stockprices.timestamp, 'open':stockprices.open, 'high':stockprices.high, 'low':stockprices.low, 'close':stockprices.close, 'high_mean_3d':stockprices_mean_3d["high"], 'low_mean_3d':stockprices_mean_3d["low"], 'volume_mean_3d':stockprices_mean_3d["volume"], 'high_mean_7d':stockprices_mean_7d["high"], 'low_mean_7d':stockprices_mean_7d["low"], 'volume_mean_7d':stockprices_mean_7d["volume"], 'high_mean_30d':stockprices_mean_30d["high"], 'low_mean_30d':stockprices_mean_30d["low"], 'volume_mean_30d':stockprices_mean_30d["volume"], 'high_std_3d':stockprices_std_3d["high"], 'low_std_3d':stockprices_std_3d["low"], 'volume_std_3d':stockprices_std_3d["volume"], 'high_std_7d':stockprices_std_7d["high"], 'low_std_7d':stockprices_std_7d["low"], 'volume_std_7d':stockprices_std_7d["volume"],'high_std_30d':stockprices_std_30d["high"], 'low_std_30d':stockprices_std_30d["low"], 'volume_std_30d':stockprices_std_30d["volume"]}).fillna(0)

exo = ['high_mean_3d', 'low_mean_3d', 'volume_mean_3d', 'high_mean_7d', 'low_mean_7d', 'volume_mean_7d', 'high_mean_30d', 'low_mean_30d', 'volume_mean_30d', 'high_std_3d', 'low_std_3d', 'volume_std_3d', 'high_std_7d', 'low_std_7d', 'volume_std_7d', 'high_std_30d', 'low_std_30d', 'volume_std_30d',]
'''

#stockprices = stockprices[['timestamp', 'close']]
df = stockprices.rename(columns={'timestamp' : 'ds', 'close': 'y'})
m = Prophet(changepoint_prior_scale=0.01, weekly_seasonality=True)
# m.add_country_holidays(country_name='US')


####### For intraday only #########

df2 = df.copy()
df2['ds'] = pd.to_datetime(df2['ds'])

df2 = df2[df2['ds'].dt.hour <= 16]
df2 = df2[df2['ds'].dt.hour >= 9]
df2 = df2[df2['ds'].dt.dayofweek < 5]

###################################

print(df2.head(5))

'''
for feature in exo:
    m.add_regressor(feature)
print(exo)
'''

m.fit(df2[["ds","y"]])
#m.fit(df)

#m.add_country_holidays(country_name='US')
#m.fit(df2)

future = m.make_future_dataframe(periods=14, freq='D') # 'D' for day, 'M' for month

####### For intraday only #########


future = future[future['ds'].dt.hour <= 16]
future = future[future['ds'].dt.hour >= 9]
future = future[future['ds'].dt.dayofweek < 5]

###################################


print(future.tail())
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

m.plot(forecast)

m.plot_components(forecast)
plt.show()
plt.grid(visible=True)

#plot_plotly(m, forecast)
#plt.show()
#plot_components_plotly(m, forecast)
#plt.show()
'''
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2*np.pi*t)
plt.plot(t, s)

plt.title('About as simple as it gets, folks')
plt.show()
'''
