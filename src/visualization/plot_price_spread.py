import pandas as pd
import numpy as np
import joypy
import datetime

import matplotlib.pyplot as plt
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

from src.data.create_dataframe import create_dataframe_day_ahead_prices
import bidding_zones

def plot_mean_joyplot(df_total:pd.DataFrame, countries:list):

    filter_col = [col for col in df_total if col.startswith("price_spread")]
    filter_col.append("mean_price")
    df_total = df_total[filter_col]

    Q1 = df_total.quantile(0.25)
    Q3 = df_total.quantile(0.75)
    IQR = Q3 - Q1
    df_total = df_total[~((df_total < (Q1 - 1.5 * IQR)) |(df_total > (Q3 + 1.5 * IQR))).any(axis=1)]

    fig, axes = joypy.joyplot(df_total, by=df_total.index.year, column="mean_price",
                            labels=df_total.index.year.unique(), range_style='own', 
                            grid="y", linewidth=1, legend=False, fade=True, figsize=(21,8),
                            title="mean price spread 2014-2022 \nfor CWE region",
                            colormap="sunset")
    plt.savefig(f"./plots/price_spread/mean_price")
    plt.close('all')

    return

def plot_price_spread_joyplot(df_total:pd.DataFrame, countries:list):

    for country in countries:
        filter_col = ["mean_price", f"price_spread_{country}"]
        df_total = df_total[filter_col]
        df_total = df_total[df_total[f'price_spread_{country}'].notna()]

        Q1 = df_total.quantile(0.25)
        Q3 = df_total.quantile(0.75)
        IQR = Q3 - Q1
        df_total = df_total[~((df_total < (Q1 - 1.5 * IQR)) |(df_total > (Q3 + 1.5 * IQR))).any(axis=1)]

        fig, axes = joypy.joyplot(df_total, by=df_total.index.year, column=f"price_spread_{country}",
                                labels=df_total.index.year.unique(), range_style='own', 
                                grid="y", linewidth=1, legend=False, fade=True, figsize=(21,8),
                                title=f"price spread for {country}",
                                colormap="sunset")
        plt.savefig(f"./plots/price_spread/price_spread_{country}")
        plt.close('all')

        return

def compute_price_convergence_month(df_total:pd.DataFrame):

    df = df_total["price_spread_total"]
    df = df.replace(to_replace=0.0, value=np.nan).dropna()
    bins = [0,1,5,10,20,50,100,np.Infinity]
    series = df.groupby([(df.index.year), (df.index.month)]).value_counts(ascending=True, bins=bins, normalize=True)
    print(series)
    # series.index = pd.to_datetime([f'{y}-{m}' for y,m in series.index], format='%Y-%m')

    fig, ax = plt.subplots(figsize=(21,8))
    series.unstack().plot.bar(stacked=True, ax=ax, cmap="sunset")
    ax.legend()
    ax.set_title('Share of hours with price spread between countries in CWE region')
    ax.set_xlabel('time')
    ax.set_ylim(0,1)
    ax.set_ylabel('fraction of hours with price spread')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', title="price spread [€] between", bbox_to_anchor=(0.5, -0.15),
            fancybox=True, shadow=True, ncol=4)
    plt.savefig("./plots/price_spread/number_hours_price_spread.pdf", dpi=1200)
    # plt.show()
    plt.close('all')

    return series

def compute_price_convergence_relative_price_spread(df_total:pd.DataFrame):

    df = df_total["relative_price_spread"]
    df = df.replace(to_replace=0.0, value=np.nan).dropna()
    bins = [-np.inf,0,10,25,50,100,np.inf]
    series = df.groupby([(df.index.year), (df.index.month)]).value_counts(ascending=True, bins=bins, normalize=True)
    series = series.multiply(100)
    print(series)

    fig, ax = plt.subplots(figsize=(21,8))
    series.unstack().plot.bar(stacked=True, ax=ax, cmap="sunset")
    ax.legend()
    # ax.set_title('Share of hours with price spread between countries in CWE region')
    ax.set_xlabel('time', fontsize=20)
    ax.set_ylim(0,100)
    ax.set_ylabel('share of hours with price spread $[\%]$', fontsize=20)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', title="relative price spread [\%] between", bbox_to_anchor=(0.5, -0.25),
            fancybox=True, shadow=True, ncol=6, fontsize=12)
    fig.subplots_adjust(bottom=0.4)
    plt.savefig("./plots/price_spread/number_hours_relative_price_spread_year.pdf", dpi=1200)
    plt.show()
    plt.close('all')

    return series

def compute_price_convergence_year(df_total:pd.DataFrame):
    
    df = df_total["price_spread_total"]
    df = df.replace(to_replace=0.0, value=np.nan).dropna()
    bins = [0,1,5,10,20,50,100,np.Infinity]
    series = df.groupby((df.index.year)).value_counts(ascending=True, bins=bins, normalize=True)
    print(series)

    fig, ax = plt.subplots(figsize=(21,8))
    series.unstack().plot.bar(stacked=True, ax=ax, cmap="sunset")
    ax.legend()
    ax.set_title('Share of hours with price spread between countries in CWE region')
    ax.set_xlabel('time')
    ax.set_ylim(0,1)
    ax.set_ylabel('fraction of hours with price spread')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', title="price spread [€] between", bbox_to_anchor=(0.5, -0.15),
            fancybox=True, shadow=True, ncol=4)
    plt.savefig("./plots/price_spread/number_hours_price_spread_year.png", dpi=1200)
    plt.show()
    plt.close('all')

    return series

# df_total = create_dataframe_day_ahead_prices(countries_day_ahead_prices=list(bidding_zones.BIDDING_ZONES_CWE.keys()))
# df_total.to_csv("price_spread.csv")
# df_total = pd.read_excel(f"./data/dataframes/df_total_1h.xlsx", index_col=0, parse_dates=True)
# print(df_total)

# plot_mean_joyplot(df_total=df_total, countries=list(BIDDING_ZONES_CWE.keys()))
# plot_price_spread_joyplot(df_total=df_total, countries=list(BIDDING_ZONES_CWE.keys()))
# price_convergence = compute_price_convergence_month(df_total)
# price_convergence = compute_price_convergence_year(df_total)

# price_convergence = compute_price_convergence_relative_price_spread(df_total=df_total)
# print(price_convergence)