import pandas as pd
import numpy as np
from time import time

import matplotlib.pyplot as plt
from sqlalchemy import true
from torch import frobenius_norm
plt.style.use(['science'])
from matplotlib import cm, colors
# import matplotlib
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

import color_blinded
plt.cm.register_cmap('sunset', color_blinded.tol_cmap('sunset'))

from sklearn.linear_model import LogisticRegression, LinearRegression, QuantileRegressor
from sklearn.svm import l1_min_c

from data_processing import data_processing

def regression_model(regressor, df_total: pd.DataFrame(), 
    list_of_filters: list, 
    from_year:int = 2015, to_year:int = 2022):

    X_train, X_test, y_train, y_test, features_name, class_names = data_processing(df_total=df_total, need_scaling=False, binary_weekdays=False,
        list_of_filters=list_of_filters, from_year=from_year, to_year=to_year)

    start = time()
    clf = regressor
    clf = clf.fit(X_train, y_train)

    # TODO: Modell evaluieren
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    accuracy_train = clf.score(X_train, y_train)
    accuracy_test = clf.score(X_test, y_test)
    print(f"Score aus dem Trainingsset: {accuracy_train}")
    print(f"Score aus dem Testset: {accuracy_test}")

    coef_matrix = pd.DataFrame()
    coef_matrix.index = features_name
    print(coef_matrix)
    print(clf.coef_.ravel())
    coef_matrix["Coef"] = clf.coef_.ravel()

    max_best_features = coef_matrix["Coef"].nlargest(5)
    min_best_features = coef_matrix["Coef"].nsmallest(5)
    print(coef_matrix)
    coef_matrix_reduced = pd.concat([max_best_features, min_best_features], axis=0)

    print("here")

    coef_matrix.plot(kind="barh", figsize=(9, 7))
    plt.title(f"Coefficients for regression on price spread from {from_year} to {to_year}")
    plt.suptitle(f"{regressor}")
    plt.axvline(x=0, color=".5")
    plt.subplots_adjust(left=0.3)
    plt.show()

    coef_matrix_reduced.plot(kind="barh", figsize=(9, 7))
    plt.title(f"Coefficients for regression on price spread from {from_year} to {to_year}")
    plt.suptitle(f"{regressor}")
    plt.axvline(x=0, color=".5")
    plt.subplots_adjust(left=0.3)
    plt.show()
    # plt.close('all')

    from sklearn.metrics import median_absolute_error, mean_absolute_percentage_error

    mae = median_absolute_error(y_train, y_pred_train)
    print(f"MAE on training set: {mae:.2f}")
    y_pred = clf.predict(X_test)
    mae = median_absolute_error(y_test, y_pred_test)
    print(f"MAE on testing set: {mae:.2f}")

    mae = mean_absolute_percentage_error(y_train, y_pred_train)
    print(f"MAPE on training set: {mae:.2f}")
    y_pred = clf.predict(X_test)
    mae = mean_absolute_percentage_error(y_test, y_pred_test)
    print(f"MAPE on testing set: {mae:.2f}")

    return coef_matrix

df_total = pd.read_excel(f"./data/dataframes/df_total_1h_CWE.xlsx", index_col=0, parse_dates=True)
print(df_total.columns)
regression_model(df_total=df_total, regressor=LogisticRegression(), list_of_filters = ['scheduled_exchanges_', 'residual_load_', 'net_positions_', 'day_ahead_prices_'])