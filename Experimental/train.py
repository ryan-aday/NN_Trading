import neptune
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

'''
# %% Sources:
# https://medium.com/@dikshamadan/how-to-pull-data-from-an-api-and-save-as-a-csv-file-f1f14ad73e67
# https://www.alphavantage.co/documentation/
# https://neptune.ai/blog/predicting-stock-prices-using-machine-learning
# https://matplotlib.org/stable/tutorials/pyplot.html
# https://www.digitalocean.com/community/tutorials/standardscaler-function-in-python
# https://stackoverflow.com/questions/61914329/subtract-a-constant-from-a-column-in-a-pandas-dataframe
# https://stackoverflow.com/questions/2051744/how-to-invert-the-x-or-y-axis
'''
'''
run = neptune.init_run(
    project="Stock-Project/Stock-Alpha",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmOTNlZjM0Ny02MzUxLTRlMzQtODE5ZS0wYjc5ZGEzZTFiNTIifQ==",
)  # your credentials
'''
#########################################################################################################
# %% Train-Test split for time-series
#########################################################################################################
#stockprices = pd.read_csv('SPY_intraday.csv', index_col="timestamp")
stockprices = pd.read_csv('SPY_intraday.csv')
stockprices = stockprices.sort_values(by=["timestamp"], ascending=True)

# train_future = stockprices[train_size:][["timestamp"]]
# X_future = train_future["timestamp"]  + pd.DateOffset(days=1)

stockprices.timestamp = pd.to_datetime(stockprices.timestamp)
stockprices = stockprices.set_index('timestamp')
# stockprices['timestamp'] = pd.to_datetime(stockprices['timestamp'])
#stockprices["Order"] = 99 - stockprices["Order"]
#print(stockprices)
test_ratio = 0.2
training_ratio = 1 - test_ratio

train_size = int(training_ratio * len(stockprices))
test_size = int(test_ratio * len(stockprices))
print(f"train_size: {train_size}")
print(f"test_size: {test_size}")

train = stockprices[:train_size][["close"]]
test = stockprices[train_size:][["close"]]

window_size = 50
window_var = f"{window_size}day"

## Split the time-series data into training seq X and output value Y
def extract_seqX_outcomeY(data, N, offset):
    """
    Split time-series into training sequence X and outcome value Y
    Args:
        data - dataset
        N - window size, e.g., 50 for 50 days of historical stock prices
        offset - position to start the split
    """
    X, y = [], []

    for i in range(offset, len(data)):
        X.append(data[i - N : i])
        y.append(data[i])

    return np.array(X), np.array(y)
    
#### Calculate the metrics RMSE and MAPE ####
def calculate_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE)
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse


def calculate_mape(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) %
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape
    
'''    
def calculate_perf_metrics(var):
    ### RMSE
    rmse = calculate_rmse(
        np.array(stockprices[train_size:]["close"]),
        np.array(stockprices[train_size:][var]),
    )
    ### MAPE
    mape = calculate_mape(
        np.array(stockprices[train_size:]["close"]),
        np.array(stockprices[train_size:][var]),
    )

    ## Log to Neptune
    run["RMSE"] = rmse
    run["MAPE (%)"] = mape

    return rmse, mape    

def plot_stock_trend(var, cur_title, stockprices=stockprices):
    ax = stockprices[["close", var, "200*5mins"]].plot(figsize=(20, 10))
    plt.grid(False)
    plt.title(cur_title)
    plt.axis("tight")
    plt.ylabel("Stock Price ($)")
    #################plt.gca().invert_xaxis()

    ## Log to Neptune
    run["Plot of Stock Predictions"].upload(
        neptune.types.File.as_image(ax.get_figure())
    )
    

#########################################################################################################
### Simple MA (Not as good) ###
#########################################################################################################
# Initialize a Neptune run
run = neptune.init_run(
	project="Stock-Project/Stock-Alpha",
	api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmOTNlZjM0Ny02MzUxLTRlMzQtODE5ZS0wYjc5ZGEzZTFiNTIifQ==",
	name="SMA",
	description="stock-prediction-machine-learning",
	tags=["stockprediction", "MA_Simple", "neptune"],
)

stockprices[window_var] = stockprices["close"].rolling(window_size).mean()

### Include a 200-day SMA for reference
stockprices["200day"] = stockprices["close"].rolling(200).mean()

### Plot and performance metrics for SMA model
plot_stock_trend(var=window_var, cur_title="Simple Moving Averages")
rmse_sma, mape_sma = calculate_perf_metrics(var=window_var)

### Stop the run
run.stop()
'''
'''
#########################################################################################################
### Exponential MA ###
#########################################################################################################
# Initialize a Neptune run
run = neptune.init_run(
    project="Stock-Project/Stock-Alpha",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmOTNlZjM0Ny02MzUxLTRlMzQtODE5ZS0wYjc5ZGEzZTFiNTIifQ==",
    name="EMA",
    description="stock-prediction-machine-learning",
    tags=["stockprediction", "MA_Exponential", "neptune"],
)

window_ema_var = f"{window_var}_EMA"

# Calculate the <window_size>-day exponentially weighted moving average
stockprices[window_ema_var] = (
    stockprices["close"].ewm(span=window_size, adjust=False).mean()
)
stockprices["200*5mins"] = stockprices["close"].rolling(200).mean()

### Plot and performance metrics for EMA model
plot_stock_trend(
    var=window_ema_var, cur_title="Exponential Moving Averages")
rmse_ema, mape_ema = calculate_perf_metrics(var=window_ema_var)

### Stop the run
run.stop()
'''
#########################################################################################################
### Predict Prices ###
#########################################################################################################

layer_units = 64
optimizer = "adam"
cur_epochs = 20
cur_batch_size = 32 # Good practice to keep it at a power of 2

'''
import keras
layer_units = 50
initial_learning_rate = 0.01
optimizer = keras.optimizers.SGD(learning_rate=initial_learning_rate)
cur_epochs = 100
cur_batch_size = 64
'''

cur_LSTM_args = {
    "units": layer_units,
    "optimizer": optimizer,
    "batch_size": cur_batch_size,
    "epochs": cur_epochs,
}

# Initialize a Neptune run
run = neptune.init_run(
    project="Stock-Project/Stock-Alpha",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmOTNlZjM0Ny02MzUxLTRlMzQtODE5ZS0wYjc5ZGEzZTFiNTIifQ==",
    name="LSTM",
    description="stock-prediction-machine-learning",
    tags=["stockprediction", "LSTM", "neptune"],
)
run["LSTM_args"] = cur_LSTM_args

# Scale our dataset
scaler = StandardScaler()
scaled_data = scaler.fit_transform(stockprices[["close"]])
scaled_data_train = scaled_data[: train.shape[0]]

# We use past <window_size> days’ stock prices for our training to predict the 51th day's closing price.
X_train, y_train = extract_seqX_outcomeY(scaled_data_train, window_size, window_size)

### Setup Neptune's Keras integration ###
from neptune.integrations.tensorflow_keras import NeptuneCallback
from keras.layers import Activation, Dense, Input, LSTM
from keras import Model
import matplotlib.dates as mdates ## Import required library

neptune_callback = NeptuneCallback(run=run)

### Build a LSTM model and log training progress to Neptune ###

def Run_LSTM(X_train, layer_units=50):
    inp = Input(shape=(X_train.shape[1], 1))

    x = LSTM(units=layer_units, return_sequences=True)(inp)
    x = LSTM(units=layer_units)(x)
    out = Dense(1, activation="linear")(x)
    model = Model(inp, out)

    # Compile the LSTM neural net
    model.compile(loss="mean_squared_error", optimizer="adam")

    return model


model = Run_LSTM(X_train, layer_units=layer_units)

history = model.fit(
    X_train,
    y_train,
    epochs=cur_epochs,
    batch_size=cur_batch_size,
    verbose=1,
    validation_split=0.1,
    shuffle=True,
    callbacks=[neptune_callback],
)

# predict stock prices using past window_size stock prices
def preprocess_testdat(data=stockprices, scaler=scaler, window_size=window_size, test=test):
    raw = data["close"][len(data) - len(test) - window_size:].values
    raw = raw.reshape(-1,1)
    raw = scaler.transform(raw)

    X_test = [raw[i-window_size:i, 0] for i in range(window_size, raw.shape[0])]
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_test

X_test = preprocess_testdat()
#train_future = stockprices[train_size:][["timestamp"]]
#X_future = train_future["timestamp"]  + pd.DateOffset(days=1)
#X_future = preprocess_testdat(train_future)

predicted_price_ = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price_)

#future_price_ = model.predict(X_future)
#future_price = scaler.inverse_transform(future_price_)

# Plot predicted price vs actual closing price
test["Predictions_lstm"] = predicted_price
#test["Future_lstm"] = future_price

# Evaluate performance
rmse_lstm = calculate_rmse(np.array(test["close"]), np.array(test["Predictions_lstm"]))
mape_lstm = calculate_mape(np.array(test["close"]), np.array(test["Predictions_lstm"]))

### Log to Neptune
run["RMSE"] = rmse_lstm
run["MAPE (%)"] = mape_lstm

### Plot prediction and true trends and log to Neptune
def plot_stock_trend_lstm(train, test):
    fig = plt.figure(figsize = (20,10))
    plt.plot(np.asarray(train.index), np.asarray(train["close"]), label = "Train Closing Price")
    plt.plot(np.asarray(test.index), np.asarray(test["close"]), label = "Test Closing Price")
    plt.plot(np.asarray(test.index), np.asarray(test["Predictions_lstm"]), label = "Predicted Closing Price")
    plt.title("LSTM Model")
    plt.xlabel("Date")
    plt.ylabel("Stock Price ($)")
    plt.legend(loc="upper left")
    # plt.gca().invert_xaxis()
    
    
    locator = mdates.AutoDateLocator()
    #formatter = mdates.ConciseDateFormatter(locator)
    #formatter.scaled[1/(24*60)] = '%M:%S'
    #plt.gca().xaxis.set_major_locator(formatter)

    #formatter = AutoDateFormatter(locator)
    hours = mdates.HourLocator(interval=12) ## 12 months apart & show last date
    plt.gca().xaxis.set_major_locator(locator) ## Set months as major locator
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M')) ##Display format - update here to change
    ## Log image to Neptune
    run["Plot of Stock Predictions"].upload(neptune.types.File.as_image(fig))

plot_stock_trend_lstm(train, test)




### Stop the run after logging
run.stop()

