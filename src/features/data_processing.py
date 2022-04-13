from pyexpat import XML_PARAM_ENTITY_PARSING_ALWAYS
from catboost import train
import pandas as pd
import numpy as np

from datetime import date

from sklearn.model_selection import  train_test_split, TimeSeriesSplit

from sklearn.preprocessing import StandardScaler, MinMaxScaler

def data_processing(df_total:pd.DataFrame, 
    list_of_filters:list, 
    from_year:int, to_year:int,
    binary_labels: bool = False, 
    continuous_labels: bool = False, 
    binary_weekdays:bool = False, 
    need_scaling: bool = False):

    ## TODO
    # Variablen: Gesamte residuale Last

    # df_total = df_total[date(year=2015, month=1, day=1).isoformat():date(year=2021, month=12, day=31).isoformat()]
    filter_col = [col for col in df_total if col.startswith(tuple(list_of_filters))]
    filter_col.extend(['mean_price', 'price_spread_total'])
    df_total = df_total[filter_col]
    # df_total = df_total.drop(columns=['day_ahead_prices_DE_AT_LU', 'scheduled_exchanges_AT_DE_LU', 'scheduled_exchanges_DE_AT_LU_FR', 'scheduled_exchanges_DE_AT_LU_NL', 'residual_load_DE_AT_LU'])
    print(df_total.columns)
    df_total = df_total.dropna(how="any", axis=1)

    # correlation_matrix(df_total)

    df_total["hour"] = df_total.index.hour
    df_total["season"] = df_total.index.month%12 // 3 + 1
    if binary_weekdays:
        names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for i, x in enumerate(names):
            df_total[x] = (df_total.index.get_level_values(0).weekday == i).astype(int)

    X = df_total.loc[:, df_total.columns != 'price_spread_total']
    features_name = X.columns

    y = df_total.loc[:, 'price_spread_total']
    class_names = ["Full convergence", "Moderate convergence", "Divergence"]

    # if binary_labels:
    #     y = pd.cut(df_total['price_spread_total'], bins=[0,10,np.Infinity],
    #     labels=["Konvergenz", "Divergenz"])
    #     y = y.cat.codes + 1
    #     y = y.astype('bool')
    # elif continuous_labels:
    #     y = y
    # else:
    y = pd.cut(df_total['price_spread_total'], bins=[0,1,10,np.Infinity], labels=["Full convergence", "Moderate convergence", "Divergence"])
    y = y.cat.codes + 1


    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

    # tscv = TimeSeriesSplit(n_splits=2, test_size=8760)
    # all_splits = list(tscv.split(X,y))
    # train_0, test_0 = all_splits[0]

    # X_train = X.iloc[train_0]
    # X_test = X.iloc[test_0]
    # y_train = y.iloc[train_0]
    # y_test = y.iloc[test_0]

    print(features_name)
    print(class_names)

    print(np.shape(X_train))
    print(np.shape(y_train))
    print(np.shape(X_test))
    print(np.shape(y_test))

    X_train = np.nan_to_num(X_train).astype(np.float32)
    X_test = np.nan_to_num(X_test).astype(np.float32)
    y_train = np.nan_to_num(y_train).astype(np.float32)
    y_test = np.nan_to_num(y_test).astype(np.float32)

    if need_scaling:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train, y_train)
        X_test = scaler.fit_transform(X_test, y_test)

    return X_train, X_test, y_train, y_test, features_name, class_names

# df_total = pd.read_excel(f"./data/dataframes/df_total_1h_CWE.xlsx", index_col=0, parse_dates=True)
# print(df_total.columns)
# data_processing(df_total=df_total, list_of_filters = ['scheduled_exchanges_', 'residual_load_', 'net_positions_', 'day_ahead_prices_'], from_year=2015, to_year=2022)