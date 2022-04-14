import pandas as pd
import numpy as np
from datetime import date, datetime, time, timedelta
from pandas._libs.tslibs import timestamps

from pandas.io.parsers import read_csv
from bidding_zones import *

from src.data.create_dataframe import *

import matplotlib.pyplot as plt
plt.style.use('science')
import color_blinded
plt.cm.register_cmap('rainbow_discrete', color_blinded.tol_cmap('rainbow_discrete'))
plt.cm.register_cmap('sunset', color_blinded.tol_cmap('sunset'))

#muting unnecessary warnings if needed
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

def plot_boxplot_marketsplit(df_total, year:int, month:int, day:int):

    date_to_split = date(year=year, month=month, day=day).isoformat()
    to_date = datetime.strptime(date_to_split, "%Y-%m-%d")
    from_date = datetime.strptime(date_to_split, "%Y-%m-%d")+timedelta(1)

    # create train test partition
    train = df_total[:to_date]
    test  = df_total[from_date:]
    print('Train Dataset:',train.shape)
    print('Test Dataset:',test.shape)

    first_date = datetime.strftime(df_total.index[0], "%Y-%m-%d")
    last_date = datetime.strftime(df_total.index[-1], "%Y-%m-%d")
    to_date_str = datetime.strftime(to_date, "%Y-%m-%d")
    from_date_str = datetime.strftime(from_date, "%Y-%m-%d")

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,5))
    sns.boxplot(data=train, ax=ax[0])
    ax[0].set(xlabel="Country",
        ylabel="Day-Ahead-Prices",
        title=f"Day-Ahead-Prices\n{first_date} - {to_date_str}")
    sns.boxplot(data=test, ax=ax[1])
    ax[1].set(xlabel="Country",
        ylabel="Day-Ahead-Prices",
        title=f"Day-Ahead-Prices\n{from_date_str} - {last_date}")
    fig.tight_layout()
    # plt.show()

    return

def plot_boxplot_cluster(df_total: pd.DataFrame, country_code : str, remove_outlier: bool = True):

    # filter_col = [col for col in df_total if col.endswith('DE_LU')]
    filter_col = [f'cluster_time_series_{country_code}', f'day_ahead_prices_{country_code}', f'net_positions_{country_code}']
    df = df_total[filter_col]

    if country_code == 'DE_LU':
        date_to_split = date(year=2018, month=10, day=1).isoformat()
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

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(16,4))
    sns.boxplot(x=df[f'cluster_time_series_{country_code}'], y=df[f'day_ahead_prices_{country_code}'], ax=ax[0])
    ax[0].set(xlabel="Cluster",
        ylabel="Day-Ahead-Prices",
        title=f"bidding zone {country_code}")
    sns.boxplot(x=df[f'cluster_time_series_{country_code}'], y=df[f'net_positions_{country_code}'], ax=ax[1])
    ax[1].set(xlabel="Cluster",
        ylabel="Net positions",
        title=f"bidding zone {country_code}")
    fig.tight_layout()
    plt.savefig(f"./plots/boxplots/clustering_{country_code}_{remove_outlier}.png")
    # plt.show()
    plt.close('all')

    return

def correlation_matrix(df_total:pd.DataFrame):

    # Compute the correlation matrix
    corrmat = df_total.corr()
    print(corrmat)

    k_best_features = corrmat["price_spread_total"].abs().nlargest(5)
    list(k_best_features.index)

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corrmat, dtype=bool))

    import seaborn as sns

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 11))

    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corrmat, square=True, annot=True, vmin=-1, vmax=1, linewidths=.5, cmap="sunset", cbar_kws={'shrink': 0.5})
    plt.tight_layout()
    plt.title(f"Correlation matrix from 2015-1-1 to 2020-11-17")
    plt.savefig(f"./plots/correlation_matrix/ALEGrO/correlation_matrix_day_ahead_prices_before", dpi=1200)
    # plt.show()

    corrmat_reduced = df_total[list(k_best_features.index)].corr()

    plt.figure(figsize=(10, 10))
    sns.heatmap(corrmat_reduced, square=True, annot=True, vmin=-1, vmax=1, linewidths=.5, cmap="sunset", cbar_kws={'shrink': 0.5})
    plt.tight_layout()
    plt.title(f"Correlation matrix from 2015-1-1 to 2020-11-17")
    plt.savefig(f"./plots/correlation_matrix/ALEGrO/k_best_features_correlation_matrix_day_ahead_prices_before", dpi=1200)
    # plt.show()

    plt.close('all')
    return

# # df_total = pd.read_excel(f"./data/dataframes/df_total_reduced_1h_with_nan.xlsx", index_col="Date")
# # plot_boxplot_cluster(df_total=df_total, country_code='DE_LU', remove_outlier=True)

# from create_dataframe import create_dataframe_day_ahead_prices, create_dataframe_residual_load, create_dataframe_scheduled_exchanges
# df_day_ahead_prices = create_dataframe_day_ahead_prices(countries_day_ahead_prices=list(BIDDING_ZONES_CWE.keys()))
# df_residual_load = create_dataframe_residual_load(countries_load=list(BIDDING_ZONES_CWE.keys()))
# df_scheduled_exchanges = create_dataframe_scheduled_exchanges(countries_scheduled_exchanges=list(BIDDING_ZONES_CWE.keys()))
# df_total = pd.concat([df_day_ahead_prices, df_residual_load, df_scheduled_exchanges], axis=1)

# countries = list(BIDDING_ZONES_CWE.keys())

# # scatter_matrix_price_spread(df_total=df_total)

# df_pure = df_total

# # for from_year in range(2015,2022):

# # 2020-11-18
# df_total = df_pure[date(year=2015, month=1, day=1).isoformat():date(year=2020, month=11, day=17).isoformat()]
# filter_col = [col for col in df_total if col.startswith(('scheduled_exchanges_', 'price_spread_total'))]
# df_total = df_total[filter_col]
# correlation_matrix(df_total)