# NN_Trading
Creating deep learning and regression models to trade with. Keeping this public for all to access &amp; improve!

## Thanks:
 - Tom Sawada for the LSTM layer suggestion

## Updates:
7/21/2024:
 - Added plots to bin.py and dl.py to also show historical closing price data.
 - Remade the bin.py script such that sentiment was binary classfied, then predicted for the regression model to predict future closing prices
 - Adam is the only algorithm now in dl.py, all other optimizers found to be way too inaccurate
 - Rewrote the dl.py script to use LSTM layers instead of Dense layers to improve accuracy (made sense due to time basis)
 - Added dl_gru.py script to try to improve upon dl.py accuracy.
 - Tried hybrid model (not pushed) with GRU then LSTM layers and vice versa. Less accurate than both strict models.
 - Added new references to tqdm for progress bar and holidays for holiday dates.
 - Set os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' for floating-point calcs to not be rounded off (and to remove the annoying warning)
 - max_workers = 64 for performance improvements w/o bogging other environmental processes. Found through trial & error.
 - Note: Ran parameter tuning for dl.py and dl_gru.py. Parameters are optimized for the most part. Layer units also optimized through brute-forcing, and current scripts have the most stable outputs with the unit amounts.


## SFI_AI
Sentiment & Financial Indicator (SFI) AI models.

Aggregates all financial data and news articles for the past 1000 days, then uses features from the financial data and sentiment analysis to determine whether the ticker is likely to go up or down in price the following day.

Both the bin.py and dl.py scripts compare different model/optimizer algorithms w/ different parameters to train, then select the best model/optimizer to provide the model with the best accuracy. Both are also parallelized for improved efficiency (max_workers = 64).

NOTE: The holiday feature designates which days are holidays, to better characterize stock price behavior.
NOTE: Currently, to have predictions that are meaningful to project for the future 2 weeks, the future predictions use the best model w/ the closing prices and features of the last 30 days. This is a "quirk", and a more permanent fix may be necessary.

### Before running:
Run the following command to install/update the necessary libraries:

pip install requirements -r

### Data Aggregation (Both scripts)
Uses nltk & Hugging Face (Transformer) for sentiment analysis, Yahoo Finance & requests, newspaper for related news articles, BeautifulSoup (bs4) for document parsing, ta for certain financial indicators.

#### Financial Indicators
Some of the more specialized indicators were researched here: https://www.investopedia.com/top-7-technical-analysis-tools-4773275

RSI, Bollinger Bands, OBV, A/D Line, ADI, Aroon Indicator, and Stochastic Indicator all use the ta library.

 - Time Features (Hour, Day of Week, Minute)
 - Daily Open stock price
 - Daily Close stock price
 - Moving Averages: SMA_5. SMA_10, SMA_12, SMA_26
 - MACD
 - RSI
 - Bollinger Bands (Lower, Mid, Upper)
 - On-Balance Volume (OBV)
 - Accumulation/Distribution (A/D) Line
 - Average Directional Index (ADI)
 - Aroon Indicator (Up, Down)
 - Stochiastic Oscillator

Other good sources:
 - https://stackoverflow.com/questions/61976027/scraping-yahoo-finance-at-regular-intervals
 - https://howtotrade.com/trading-strategies/triple-moving-average-crossover/

### bin.py: Binary Classification
Uses sklearn, transformers to apply binary classification models with varying parameters to optimize each model for improved accuracy.

Models used:
 - Logistic Regression
 - Random Forest
 - Gradient Boosting
 - SVD (Singular Value Decomposition)

Least processing time for decent accuracy (usually around ~75% for the optimized models). Expect GradientBoosting to provide some of the more accurate results.

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(),
        'GradientBoosting': GradientBoostingClassifier(),
        'SVC with SVD': Pipeline([
            ('svd', TruncatedSVD(random_state=42)),
            ('svc', SVC(kernel='rbf'))
        ])
    }

    param_grids = {
        'LogisticRegression': {'C': [0.01, 0.1, 1, 10]},
        'RandomForest': {'n_estimators': [50, 100, 150], 'max_depth': [5, 10, 15]},
        'GradientBoosting': {'n_estimators': [50, 100, 150, 200, 250], 'learning_rate': [0.01, 0.1, 0.15, 0.2]},
        'SVC with SVD': {
            'svd__n_components': [5, 10, 15],
            'svc__C': [0.1, 1, 10],
            'svc__gamma': ['scale', 'auto']
        }
    }

The script now includes various regression models to predict the next two weeks closing price.
The regression models:
 - Random Forest
 - Linear Regression (Linear regression does not have hyperparameters to tune in this context)
 - SVR
 - ARIMA

The features used for future regression predictions are semi-normally distributed from the historical ticker dataset.
The sentiment analysis index (0 for low sentiment, 1 for good sentiment) is used to determine variability: adjusted_std = std * (1 - sentiment_mean)/2 * 0.5
The 0.5 factor was introduced to reduce observed heavily volatile price movement in all stocks.

The output plot has the actual & historical closing price data and the predicted price data. Since the regression model is extremely accurate, the actual historical data plot line has been made more opaque and thicker to differentiate between the curves.

Example output plot:
![Figure_10](https://github.com/user-attachments/assets/cfde7168-fd1f-4c17-9676-ef087e2e0cdc)


### dl.py: Deep Learning
Uses sklearn, transformers, tensorflow to develop a deep learning network with parameters to optimize each provided model for improved accuracy.

Models used:
 - Adam (Adaptive Moment Estimation), optimizer and shown parameters chosen through trial and error

Deep Learning stucture is a single input layer, 2 hidden LSTM layers w/ relu activation functions.

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], 1)))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))

    optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

    param_grid = {
        'learning_rate': [0.005],
        'beta_1': [0.90],
        'beta_2': [0.999],
        'epsilon': [1e-8]
    }

Example output plot:
![Figure_1](https://github.com/user-attachments/assets/0118fb60-6296-46c4-a6f0-5eefdb8bfab3)

Good sources:
 - https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/
 - https://datascience.stackexchange.com/questions/38955/how-does-the-validation-split-parameter-of-keras-fit-function-work
 - https://www.rdocumentation.org/packages/keras/versions/2.0.5/topics/fit
 - https://stackoverflow.com/questions/75162883/keras-loss-value-very-high-and-not-decreasing
 - https://www.geeksforgeeks.org/hidden-layer-perceptron-in-tensorflow/
 - https://keras.io/api/optimizers/lion/
 - https://datascience.stackexchange.com/questions/57738/can-tanh-be-used-as-an-output-for-a-binary-classifier
 - https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc
 - https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
 - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
 - https://datascience.stackexchange.com/questions/38955/how-does-the-validation-split-parameter-of-keras-fit-function-work

### dl_gru.py: Deep Learning w/ GRU Layers
Uses sklearn, transformers, tensorflow to develop a deep learning network with parameters to optimize each provided model for improved accuracy.

Models used:
 - Adam (Adaptive Moment Estimation), optimizer and shown parameters chosen through trial and error

Deep Learning stucture is a single input layer, 2 hidden GRU layers w/ relu activation functions.

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], 1)))
    model.add(GRU(128, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))

    optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

    param_grid = {
        'learning_rate': [0.000625],
        'beta_1': [0.8, 0.85],
        'beta_2': [0.9999],
        'epsilon': [1.75e-6, 2e-6]
        
    }

Example output plot:
![Figure_9](https://github.com/user-attachments/assets/b9060af9-2efa-4e63-a885-79fa906a57e5)

Good sources:
 - https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU

## Evaluation and Analysis of an LSTM and GRU Based Stock Investment Strategy
Interesting paper that led to my rabbit hole away from the Dense model.

https://www.researchgate.net/publication/379175870_Comparative_Analysis_of_LSTM_GRU_and_ARIMA_Models_for_Stock_Market_Price_Prediction

## NeptuneAI (Experimental)
Experimental repository for trying ARIMA, Prophet, and NeptuneAI models to predict stock prices. The most promising was ARIMA theoretically.

Prophet (from Meta) was easier to set up, but less consistently accurate when adding additional parameters.

### load_stock.py
Loads stock price intraday information from AlphaVantage. Requires API key, and only permits a maximum of 30 pulls.
Would not recommend unless you do not mind paying for the subscription.

AlphaVantage documentation: https://www.alphavantage.co/documentation/

### train_ARIMA.py
Extrapolates moving day averages to plot out predictions for stock tickers from a provided .CSV sheet. Not very accurate for very volatile stocks.
   
    stockprices_rolled_3d = stockprices[lag_features].rolling(window=7, min_periods=0)
    stockprices_mean_3d = stockprices_rolled_3d.mean().shift(1).reset_index().astype(np.float32)
    stockprices_std_3d = stockprices_rolled_3d.std()
    stockprices_rolled_7d = stockprices[lag_features].rolling(window=7, min_periods=0)
    stockprices_mean_7d = stockprices_rolled_7d.mean().shift(1).reset_index().astype(np.float32)
    stockprices_std_7d = stockprices_rolled_7d.std()
    stockprices_rolled_30d = stockprices[lag_features].rolling(window=7, min_periods=0)
    stockprices_mean_30d = stockprices_rolled_30d.mean().shift(1).reset_index().astype(np.float32)
    stockprices_std_30d = stockprices_rolled_30d.std()

### train_prophet.py
Plots out predictions for stock tickers from a provided .CSV sheet using Meta's proprietary Prophet algorithm. Barely any setup needed for decent accuracy, but adding more hyperparameters reduces accuracy drastically.
   
