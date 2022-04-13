import pandas as pd
import numpy as np

from data.create_dataframe import create_dataframe_generation, create_dataframe_load, create_dataframe_residual_load, create_dataframe_net_positons

import matplotlib.pyplot as plt
plt.style.use(['science'])
from matplotlib import cm, colors
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })
import matplotlib.pyplot as plt

import color_blinded
plt.cm.register_cmap('rainbow_discrete', color_blinded.tol_cmap('rainbow_discrete'))

def plot_dispatch(countries: str):

    df_generation = create_dataframe_generation(countries_generation=[countries])
    df_generation = df_generation.resample('1M').sum()

    df_load = create_dataframe_load(countries_load=[countries])
    df_load = df_load.resample('1M').sum()

    df_residual_load = create_dataframe_residual_load(countries_load=[countries])
    df_residual_load = df_residual_load[f"residual_load_{countries}"].resample('1M').sum()

    df_net_positions = create_dataframe_net_positons(countries_net_positons=[countries])
    df_net_positions = df_net_positions.resample('1M').sum()

    df_generation = df_generation[df_generation.index[0]:]
    df_load = df_load[df_generation.index[0]:]
    df_residual_load = df_residual_load[df_generation.index[0]:]
    df_net_positions = df_net_positions[df_generation.index[0]:]


    column_names = ['Wind Offshore', 'Wind Onshore', 'Solar', 'Other renewable', 'Geothermal', 'Biomass', 
        'Hydro Pumped Storage', 'Hydro Run-of-river and poundage','Hydro Water Reservoir', 'Waste',
       'Nuclear', 'Fossil Brown coal/Lignite',  'Fossil Hard coal', 'Fossil Coal-derived gas',
       'Fossil Gas', 'Fossil Oil', 'Other']

    df_generation = df_generation.reindex(columns=column_names)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(21,10))

    axes = df_generation.plot.area(ax=axes, cmap="rainbow_discrete")
    df_load.plot.line(ax=axes, color={f"load_{countries}": "k"}, linewidth=2, linestyle='dashed')
    df_residual_load.plot.line(ax=axes, color={f"residual_load_{countries}": "r"}, linewidth=2, linestyle='dashed')
    df_net_positions.plot.line(ax=axes, linewidth=2)

    plt.ylim(bottom=min(df_net_positions.min()))
    # plt.xlim(df_generation.index[1], df_generation.index[-3])
    axes.set_title('Aggregated generation output per market time unit and per production type in France')
    axes.set_xlabel('time')
    axes.set_ylabel('generation [MW]')

    box = axes.get_position()
    axes.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])

    # Put a legend below current axis
    lgd = axes.legend(loc='upper center', title="production type", bbox_to_anchor=(0.5, -0.1),
            fancybox=True, shadow=True, ncol=4)

    axes.grid(True)

    fig.subplots_adjust(bottom=0.2)
    fig.tight_layout()

    plt.savefig(f"./plots/dispatch/dispatch_{countries}.png", dpi=1200, bbox_inches='tight')
    # plt.show()

    plt.close('all')

    return
