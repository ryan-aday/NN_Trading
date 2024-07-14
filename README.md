# NN_Trading
Creating NN models to trade with. Keeping this public for all to access &amp; improve!

## NeptuneAI (Experimental)
Experimental repository for trying ARIMA, Prophet, and NeptuneAI models to predict stock prices. The most promising was ARIMA theoretically.

Prophet (from Meta) was easier to set up, but less consistently accurate when adding additional parameters.

### load_stock.py
Loads stock price intraday information from AlphaVantage. Requires 

## SFI_AI
Sentiment & Financial Indicator (SFI) AI models.

Aggregates all financial data and news articles for the past 1000 days, then uses features from the financial data and sentiment analysis to determine whether the ticker is likely to go up or down in price the following day.

Both the bin.py and dl.py scripts compare different model/optimizer algorithms w/ different parameters to train, then select the best model/optimizer to provide the model with the best accuracy. Both are also parallelized for improved efficiency (max_workers = 16).

### Data Aggregation (Both scripts)
Uses nltk for sentiment analysis, Yahoo Finance & requests, newspaper for related news articles, BeautifulSoup (bs4) for document parsing, ta for certain financial indicators.

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
