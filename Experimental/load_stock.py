import requests, csv
import pandas as pd

# %% Sources:
# https://medium.com/@dikshamadan/how-to-pull-data-from-an-api-and-save-as-a-csv-file-f1f14ad73e67
# https://www.alphavantage.co/documentation/
# https://neptune.ai/blog/predicting-stock-prices-using-machine-learning
# https://medium.com/@mudassar_lhr/how-to-download-the-csv-file-from-url-using-python-650eae6d3478

# %% Load API data from AlphaVantage
apikey = '9Q5Y150UTORQGB2Y'
symbol = 'ALAB'
outputsize = 'full'

# url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=SPY&datatype=json&apikey=' + apikey
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=' + symbol + '&interval=5min&outputsize=' + outputsize + '&datatype=csv&apikey=' + apikey
# https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&outputsize=full&apikey=demo
r = requests.get(url)
data = r.content
csv_file = open(f'{symbol}_intraday.csv', 'wb')
csv_file.write(data)
csv_file.close()

####### JSON Method (not recommended)

'''
data = r.json()
# print(data)

values = []
#headers = ['Order', 'Open', 'High', 'Low', 'Close', 'Volume']
headers = []
for i in data['Time Series (5min)']:
  headers.append(i)
  values.append(data['Time Series (5min)'][i])
  
df = pd.DataFrame(values)
df.set_axis(headers, axis=0)
df = df.rename(columns={'' : 'Order', '1. open': 'Open', '2. high': 'High', '3. low' : 'Low', '4. close' : 'Close', '5. volume' : 'Volume'}) 

name = data['Meta Data']['2. Symbol']
df.to_csv(f'stock_data_{name}_intraday.csv')
'''

