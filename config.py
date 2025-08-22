# config.py

# --- Data Splitting Settings ---
LOOKBACK_MINUTES = 8
PREDICTION_MINUTES = 2

# --- Feature Engineering Settings ---
# A list of features you want to calculate.
# This makes it easy to add or remove features.
ACTIVE_FEATURES = [
    'volatility',
    'volume',
    'vwap_change'
]

# --- Labeling Settings ---
# The threshold for defining a momentum move.
# 0.75 means the price must close in the top or bottom 25% of the range.
LABEL_THRESHOLD = 0.75

# --- Model Settings ---
# Choose your model: 'XGBoost', 'RandomForest', or 'LogisticRegression'
MODEL_TYPE = 'XGBoost'

# XGBoost specific hyperparameters you might want to tweak
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1
}

# --- File Path Settings ---
RAW_DATA_DIR = 'spikehistory/'
PROCESSED_DATA_PATH = 'processed/features_and_labels.csv'
MODEL_SAVE_PATH = 'models/trained_model.pkl'