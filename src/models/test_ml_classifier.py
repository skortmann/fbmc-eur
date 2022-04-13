import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
from sklearn.metrics import accuracy_score

from data_processing import data_processing

list_of_filters = ['scheduled_exchanges_', 'residual_load_', 'net_position_', 'day_ahead_prices_']
df_total = pd.read_excel(f"./data/dataframes/df_total_1h_CWE.xlsx", index_col=0, parse_dates=True,   engine='openpyxl')

X_train, X_test, y_train, y_test, features_name_, class_names_ = data_processing(df_total, list_of_filters, from_year=2015, to_year=2022)

import lazypredict
from lazypredict.Supervised import LazyClassifier

clf = LazyClassifier(verbose=0, ignore_warnings=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)