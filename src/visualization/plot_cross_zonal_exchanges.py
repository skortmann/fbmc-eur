import pandas as pd
import numpy as np
from datetime import date

from bidding_zones import BIDDING_ZONES_CWE
from data.create_dataframe import create_dataframe_scheduled_exchanges

import matplotlib.pyplot as plt
from matplotlib import cm
plt.style.use('science')
import matplotlib
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })
matplotlib.rcParams.update({
    'font.size' : 12
})

import color_blinded
plt.cm.register_cmap('rainbow_discrete', color_blinded.tol_cmap('rainbow_discrete'))
plt.cm.register_cmap('sunset', color_blinded.tol_cmap('sunset'))

def plot_cross_zonal_exchanges(df_total:pd.DataFrame, countries:list):

    filter_col = [col for col in df_total if col.startswith(('scheduled_exchanges'))]
    df = df_total[filter_col]

    df_DE_BE = df
    df_DE_BE = df_DE_BE.resample("M").sum()

    df = df.sum(axis=1, skipna=True)

    df_1=df.resample("M").mean()
    df_2=df.resample("M").min()
    df_3=df.resample("M").max()
    df_4=df.resample("M").quantile(0.05)
    df_5=df.resample("M").quantile(0.95)

    df_total = pd.concat([df_1, df_2, df_3, df_4, df_5], axis=1)
    df_total = df_total.rename(columns={df_total.columns[0]: "mean", df_total.columns[1]: "min", df_total.columns[2]: "max", 
        df_total.columns[3]: "5\% quantile", df_total.columns[4]: "95\% quantile"})

        ## PLOT 1

    fig, ax = plt.subplots(figsize=(21,8))
    linewidths = [4, 4, 4, 2, 2]
    style = ['-', '--', '--', ':', ':']
    color = ['#CC4D38', 'k', 'k', '#CC4D38', '#CC4D38']
    for col, style, lw, color in zip(df_total.columns, style, linewidths, color):
        df_total[col].plot(style=style, lw=lw, ax=ax, color=color)
    ax.grid(True)
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
            fancybox=True, shadow=True, ncol=5)

    ax.axvline(pd.Timestamp('2015-05-20'), color='#9D9EA0', linestyle="--", lw=4)
    ax.text(x= pd.Timestamp('2015-05-20'), y=.85, transform=ax.get_xaxis_transform(),
        s="Introduction of flow based market coupling\nbetween bidding zones in CWE region", color='black', 
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    ax.axvline(pd.Timestamp('2018-10-1'), color='#9D9EA0', linestyle="--", lw=4)
    ax.text(x= pd.Timestamp('2018-10-1'), y=.85, transform=ax.get_xaxis_transform(),
        s="Market splitting\nbetween bidding zones DE/LU-AT", color='black', 
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    ax.axvline(pd.Timestamp('2020-11-18'), color='#9D9EA0', linestyle="--", lw=4)
    ax.text(x= pd.Timestamp('2020-11-18'), y=.85, transform=ax.get_xaxis_transform(),
        s="Opening of ALEGrO interconnector\nbetween bidding zones DE/LU-BE", color='black', 
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))    
        
    ax.set_title('Mean, maximum and minimum monthly cross border trading\n between countries in CWE region')
    ax.set_xlabel('time')
    ax.set_ylabel('cross border volume [MWh]')

    ax.set_ylim(0,)
    
    plt.savefig("./plots/cross_border/cross-border-flows_summary.png", dpi=1200)

        ## PLOT 2

    df = df.resample("M").sum()
    ma = df.rolling(6).mean()
    mstd = df.rolling(6).std()
    
    fig, ax = plt.subplots(figsize=(21,8))

    ax.plot(df.index, df, color="k", linewidth=2, label="actual aggregated")
    ax.plot(ma.index, ma, color='#CC4D38', linestyle=":", linewidth=2, label="rolling mean")
    ax.fill_between(mstd.index, ma - 2 * mstd, ma + 2 * mstd, color=["#9D9EA0"], alpha=0.2)

    ax.axvline(pd.Timestamp('2015-05-20'), color='#9D9EA0', linestyle="--", lw=4)
    ax.text(x= pd.Timestamp('2015-05-20'), y=.85, transform=ax.get_xaxis_transform(),
        s="Introduction of flow based market coupling\nbetween bidding zones in CWE region", color='black', 
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    ax.axvline(pd.Timestamp('2018-10-1'), color='#9D9EA0', linestyle="--", lw=4)
    ax.text(x= pd.Timestamp('2018-10-1'), y=.85, transform=ax.get_xaxis_transform(),
        s="Market splitting\nbetween bidding zones DE/LU-AT", color='black', 
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    ax.axvline(pd.Timestamp('2020-11-18'), color='#9D9EA0', linestyle="--", lw=4)
    ax.text(x= pd.Timestamp('2020-11-18'), y=.85, transform=ax.get_xaxis_transform(),
        s="Opening of ALEGrO interconnector\nbetween bidding zones DE/LU-BE", color='black', 
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    ax.set_title("Aggregated monthly values of CWE cross border trading")
    ax.grid(True)
    ax.set_ylim(0, 16000000)
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
            fancybox=True, shadow=True, ncol=4)

    ax.set_ylim(0)
    ax.set_xlim(df.index[0], df.index[-1])
    ax.set_xlabel('time')
    ax.set_ylabel('cross border volume [MWh]')
    # ax.get_yaxis().get_major_formatter().set_scientific(False)
    
    plt.savefig("./plots/cross_border/cross-border-flows_monthly.png", dpi=1200)

        ## PLOT 3

    fig, ax = plt.subplots(figsize=(21,8))
    df_DE_BE.plot.area(stacked=True, ax=ax, colormap="rainbow_discrete", linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
            fancybox=True, shadow=True, ncol=4, fontsize=12)

    # ax.set_title("Aggregated monthly values of CWE cross border trading")

    ax.axvline(pd.Timestamp('2015-05-20'), color='#9D9EA0', linestyle="--", lw=4)
    ax.text(x= pd.Timestamp('2015-05-20'), y=.85, transform=ax.get_xaxis_transform(),
        s="Introduction of flow based market coupling\nbetween bidding zones in CWE region", color='black', 
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    ax.axvline(pd.Timestamp('2018-10-1'), color='#9D9EA0', linestyle="--", lw=4)
    ax.text(x= pd.Timestamp('2018-10-1'), y=.85, transform=ax.get_xaxis_transform(),
        s="Market splitting\nbetween bidding zones DE/LU-AT", color='black', 
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    ax.axvline(pd.Timestamp('2020-11-18'), color='#9D9EA0', linestyle="--", lw=4)
    ax.text(x= pd.Timestamp('2020-11-18'), y=.85, transform=ax.get_xaxis_transform(),
        s="Opening of ALEGrO interconnector\nbetween bidding zones DE/LU-BE", color='black', 
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    ax.grid(True)
    ax.set_ylim(0, 16000000)
    ax.set_xlabel('time', fontsize=20)
    ax.set_ylabel('cross border volume [MWh]', fontsize=20)
    
    plt.savefig("./plots/cross_border/cross-border-flows_monthly_seperated.pdf", dpi=1200)
    # plt.show()

        ## PLOT 4

    fig, ax = plt.subplots(figsize=(21,8))
    df_DE_BE.plot.area(stacked=True, ax=ax, colormap="rainbow_discrete", linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
            fancybox=True, shadow=True, ncol=4)

    ax.set_title("Aggregated monthly values of CWE cross border trading")

    ax.grid(True)
    ax.set_ylim(0, 16000000)
    ax.set_xlabel('time')
    ax.set_ylabel('cross border volume [MWh]')
    
    plt.savefig("./plots/cross_border/cross-border-flows_monthly_seperated_no_text.png", dpi=1200)
#     plt.show()
    plt.close('all')

    return

# df_total = pd.read_excel(f"./data/dataframes/df_total_1h_CWE.xlsx", index_col=0, parse_dates=True)
# df_total = create_dataframe_scheduled_exchanges(countries_scheduled_exchanges=list(BIDDING_ZONES_CWE.keys()))
# plot_cross_zonal_exchanges(df_total, countries=list(BIDDING_ZONES_CWE.keys()))