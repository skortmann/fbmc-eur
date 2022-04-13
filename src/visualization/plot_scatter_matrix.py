from cProfile import label
import os
from typing import Optional
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from bidding_zones import BIDDING_ZONES_CWE
from helper import corr

import matplotlib.pyplot as plt
# plt.style.use('science')
import seaborn as sns

def scatter_matrix_residual_load(df_total: pd.DataFrame, remove_outlier: bool = True):

    df_total = df_total.resample("1h").sum()

    if remove_outlier:
        Q1 = df_total.quantile(0.25)
        Q3 = df_total.quantile(0.75)
        IQR = Q3 - Q1
        df = df_total[~((df_total < (Q1 - 1.5 * IQR)) |(df_total > (Q3 + 1.5 * IQR))).any(axis=1)]

    filter_col = [col for col in df_total if col.startswith(('residual_load'))]
    df = df_total[filter_col]
    df["sum_residual_load"] = df.sum(axis=1, skipna=True, numeric_only=True)
    df.dropna(subset=["sum_residual_load"], inplace=True)

    df["year"] = df.index.year.copy()
    df = df.replace(to_replace=0, value=np.nan)

    sns.pairplot(df, hue="year", diag_kind="hist",
        plot_kws = {'alpha': 0.8, 's': 2, 'edgecolor': 'k'})
    plt.savefig(f"./plots/scatter_matrix/scatter_residual_load.png")

    return

###########################################################################################
###########################SECOND#FUNCTION#################################################
###########################################################################################

def scatter_matrix_price_spread(df_total: pd.DataFrame, remove_outlier: bool = True, column:str = 'day_ahead_prices_', bins: list=[0,10,50,100,np.Infinity]):

    df_total = df_total.resample("1h").sum()

    if remove_outlier:
        Q1 = df_total.quantile(0.25)
        Q3 = df_total.quantile(0.75)
        IQR = Q3 - Q1
        df = df_total[~((df_total < (Q1 - 1.5 * IQR)) |(df_total > (Q3 + 1.5 * IQR))).any(axis=1)]

    filter_col = [col for col in df_total if col.startswith((column))]
    df = df_total[filter_col]
    df["sum"] = df.sum(axis=1, skipna=True, numeric_only=True)
    df.dropna(subset=["sum"], inplace=True)

    df = df.replace(to_replace=0, value=np.nan)
    df = df.dropna(axis=1, thresh=int(len(df.columns)/2))

    df = pd.concat([df, df_total["price_spread_total"]], axis=1)
    df["price_spread_total"] = pd.cut(df["price_spread_total"], bins=bins)
    df = df.dropna(subset=["price_spread_total"])

    sns.pairplot(df, hue="price_spread_total", diag_kind="hist")
    plt.savefig(f"./plots/scatter_matrix/scatter_residual_load_price_spread_total.png")

    return

###########################################################################################
###########################THIRD#FUNCTION##################################################
###########################################################################################

def scatter_matrix_mean_price_year(df_total: pd.DataFrame, remove_outlier: bool = True):

    df_total = df_total.resample("1h").sum()

    if remove_outlier:
        Q1 = df_total.quantile(0.25)
        Q3 = df_total.quantile(0.75)
        IQR = Q3 - Q1
        df = df_total[~((df_total < (Q1 - 1.5 * IQR)) |(df_total > (Q3 + 1.5 * IQR))).any(axis=1)]

    filter_col = [col for col in df_total if col.startswith(('price_spread'))]
    df = df_total[filter_col]
    df = pd.concat([df, df_total["mean_price"]], axis=1)
    df["year"] = df.index.year
    df["mean_price"] = df["mean_price"].replace(to_replace=0, value=np.nan)
    df = df.dropna(axis=1, thresh=int(len(df.columns)/2))
    
    sns.pairplot(df, hue="year", diag_kind="hist", dropna=True, 
        plot_kws = {'alpha': 0.8, 's': 2, 'edgecolor': 'k'})
    plt.savefig(f"./plots/scatter_matrix/scatter_mean_price_year.png")

    return

###########################################################################################
###########################FOURTH#FUNCTION#################################################
###########################################################################################

def scatter_matrix_mean_price_price_spread(df_total: pd.DataFrame, remove_outlier: bool = True):

    df_total = df_total.resample("1h").sum()

    if remove_outlier:
        Q1 = df_total.quantile(0.25)
        Q3 = df_total.quantile(0.75)
        IQR = Q3 - Q1
        df = df_total[~((df_total < (Q1 - 1.5 * IQR)) |(df_total > (Q3 + 1.5 * IQR))).any(axis=1)]

    filter_col = [col for col in df_total if col.startswith(('price_spread'))]
    df = df_total[filter_col]
    df = pd.concat([df, df_total["mean_price"]], axis=1)
    df["mean_price"] = df["mean_price"].replace(to_replace=0, value=np.nan)
    df = df.dropna(axis=1, thresh=int(len(df.columns)/2))

    df["price_spread_total"] = pd.cut(df["price_spread_total"], bins=[0,1,5,10,20,50,100,np.Infinity])
    df = df.dropna(subset=["price_spread_total"])

    for column in df:
        if df[column].dtype == "float64":
            df[column]=pd.to_numeric(df[column], downcast="float")
        if df[column].dtype == "int64":
            df[column]=pd.to_numeric(df[column], downcast="integer") 
    
    sns.pairplot(df, hue="price_spread_total")
    plt.savefig(f"./plots/scatter_matrix/scatter_mean_price_price_spread.png")

    return

###########################################################################################
###########################FIFTH#FUNCTION##################################################
###########################################################################################

def pairgrid(df_total: pd.DataFrame, country_code: str, remove_outlier: bool = True):

    filter_col = [f'cluster_time_series_{country_code}', f'day_ahead_prices_{country_code}', f'residual_load_{country_code}',  
         f'net_positions_{country_code}']
    df = df_total[filter_col]

    if country_code == 'DE_LU':
        date_to_split = date(year=2020, month=1, day=1).isoformat()
        from_date = datetime.strptime(date_to_split, "%Y-%m-%d")
        df  = df[from_date:]

    if country_code == 'DE_AT_LU':
        date_to_split = date(year=2018, month=10, day=1).isoformat()
        to_date = datetime.strptime(date_to_split, "%Y-%m-%d") - timedelta(hours=1)
        df  = df[:to_date]

    if remove_outlier:
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

    df = pd.concat([df, df_total["mean_price"], df_total["price_spread_total"]], axis=1)
    df["mean_price"] = df["mean_price"].replace(to_replace=0, value=np.nan)
    df = df.dropna(axis=1, thresh=int(len(df.columns)/2))

    sns.pairplot(df, hue=f'cluster_time_series_{country_code}')
    plt.savefig(f"./plots/scatter_matrix/scatter_plot_{country_code}.png")

    return

# df_total = pd.read_excel(f"./data/dataframes/df_total_reduced_1h_with_nan_12_cluster.xlsx", index_col="Date")
# scatter_matrix(df_total=df_total, country_code='DE_LU', country_code_to='FR', remove_outlier=False)
# pairgrid(df_total=df_total, country_code='DE_LU', remove_outlier=False)

# df = pd.read_excel(f"./data/dataframes/df_total_1h.xlsx", index_col=0, parse_dates=True)
# df=df[pd.Timestamp('2020-11-20'):pd.Timestamp('2022-01-02')]

# from create_dataframe import create_dataframe_day_ahead_prices, create_dataframe_residual_load
# df_day_ahead_prices = create_dataframe_day_ahead_prices(countries_day_ahead_prices=list(BIDDING_ZONES_CWE.keys()))
# df_residual_load = create_dataframe_residual_load(countries_load=list(BIDDING_ZONES_CWE.keys()))
# df_total = pd.concat([df_day_ahead_prices, df_residual_load], axis=1)

# countries = list(BIDDING_ZONES_CWE.keys())

# scatter_matrix_price_spread(df_total=df_total)