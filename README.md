# NN_Trading
Creating NN models to trade with. Keeping this public for all to access &amp; improve!

## SFI_AI
Sentiment & Financial Indicator (SFI) AI models.

Aggregates all financial data and news articles for the past 1000 days, then uses features from the financial data and sentiment analysis to determine whether the ticker is likely to go up or down in price the following day.

Both the bin.py and dl.py scripts compare different model/optimizer algorithms w/ different parameters to train, then select the best model/optimizer to provide the model with the best accuracy. Both are also parallelized for improved efficiency (max_workers = 16).

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

### dl.py: Deep Learning
Uses sklearn, transformers to apply binary classification models with varying parameters to optimize each model for improved accuracy.

Models used:
 - Adam (Adaptive Moment Estimation)
 - Lion
 - Stochiastic Gradient Descent

Deep Learning stucture is a single input layer, 2 hidden layer  w/ relu activation functions.

    model = Sequential()
    model.add(Dense(1024, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    optimizers = [
        ('Adam', {
            'learning_rate': [0.001, 0.0001],
            'beta_1': [0.9, 0.85],
            'beta_2': [0.999, 0.9999],
            'epsilon': [1e-7, 1e-8, 1e-9]
        }),
        ('SGD', {
            'learning_rate': [0.01, 0.1],
            'momentum': [0.8, 0.9]
        }),
        # Placeholder, assuming Lion has similar params
        ('Lion', {
            'learning_rate': [0.001, 0.01],
            'beta_1': [0.9, 0.95],
            'beta_2': [0.999, 0.995],
            'epsilon': [1e-7, 1e-8],
            'amsgrad': [True, False]
        })
    ]

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
Plots out predictions for stock tickers from a provided .CSV sheet using the proprietary Prophet algorithm. Barely any setup needed for decent accuracy, but adding more hyperparameters reduces accuracy drastically.
   
