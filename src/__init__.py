__author__ = 'Steffen Kortmann'
__email__ = 'steffen.kortmann@rwth-aachen.de'
__version__ = '0.1'

import logging
import pandas as pd
import numpy as np
import timing

from data.create_folder import *
from data.import_data import *
from bidding_zones import *
from visualization.plot_scatter_matrix import *
from data.create_dataframe import *
from visualization.plot_statistical_analysis import *
from models.time_series_clustering import *
from models.k_means_clustering import *
from models.decision_tree import *
from visualization.plot_dispatch import *
from visualization.plot_price_spread import *
from visualization.plot_cross_zonal_exchanges import *
from features.data_processing import *

def import_data(countries:list):
        import_load(countries_load=countries)
        import_generation(countries_generation=countries)
        import_day_ahead_prices(countries_day_ahead_prices=countries)
        import_scheduled_exchanges(countries_scheduled_exchanges=countries)
        import_net_positions(countries_net_positions=countries)
        import_net_transfer_capacity_day_ahead(countries_net_transfer_capacity_day_ahead=countries)
        import_wind_solar_forecast(countries_wind_solar_forecast=countries)
        return

def create_dataframe(countries:list):
        df_day_ahead_prices = create_dataframe_day_ahead_prices(countries_day_ahead_prices=countries)
        df_generation = create_dataframe_generation(countries_generation=countries)
        df_load = create_dataframe_load(countries_load=countries)
        df_scheduled_exchanges = create_dataframe_scheduled_exchanges(countries_scheduled_exchanges=countries)
        df_residual_load = create_dataframe_residual_load(countries_load=countries)
        df_net_positions = create_dataframe_net_positons(countries_net_positons=countries)
        df_net_transfer_capacity_day_ahead = create_dataframe_net_transfer_capacity_day_ahead(countries_net_transfer_capacity_day_ahead=countries)

        df_total = pd.DataFrame()
        df_total = df_total.join([df_day_ahead_prices, df_residual_load, df_scheduled_exchanges, df_net_positions], how='outer')
        df_total.index = pd.to_datetime(df_total.index)
        df_total.to_excel(f"./data/processed/df_total_15min.xlsx")

        df_total = df_total.resample('1h').sum()

        df_total.replace(to_replace=0, value=np.nan, inplace=True)
        df_total.to_excel(f"./data/processed/df_total_1h.xlsx")
        df_total.to_csv(f"./data/processed/df_total_1h.csv")

        return df_total

def main(countries: list, update_data: bool = False, create_data: bool = False, clustering: bool = False):

    if update_data:
        import_data(countries=countries)
    if create_data:
        df_total = create_dataframe(countries=countries)
    else:
        df_total = pd.read_excel(f"./data/processed/df_total_1h.xlsx", index_col=0, parse_dates=True)

    if clustering:
        for country in countries:
            df_residual_load = time_series_clustering(country_code = country, plot_knee=False,
            plot_cluster_comparions=False, plot_cluster_euclidean_k_means=True)
            df_total = df_total.join(df_residual_load[f"cluster_time_series_{country}"], how='outer')
            df_total[f"cluster_k_means_{country}"] = k_means_clustering(df_total, country)

    df_total.to_excel(f"./data/processed/df_total_1h.xlsx")
    df_total.to_csv(f"./data/processed/df_total_1h.csv")

    for country in countries:
        plot_boxplot_cluster(df_total=df_total, country_code=country, remove_outlier=True)
        plot_dispatch(countries=country)
        pairgrid(df_total, country)

    # plot_boxplot(df_day_ahead_prices, year=2018, month=9, day=30)
    plot_mean_joyplot(df_total=df_total, countries=countries)
    plot_price_spread_joyplot(df_total=df_total, countries=countries)
    plot_cross_zonal_exchanges(df_total=df_total, countries=countries)
    decision_tree(df_total=df_total, from_year=2019, to_year=2022)
    scatter_matrix_residual_load(df_total=df_total)

    price_convergence = compute_price_convergence_month(df_total=df_total)
    price_convergence.to_excel(f"./data/processed/price_convergence.xlsx")

    df_total.groupby(by=df_total.index.year).describe(percentiles=[0.05, 0.95]).to_csv("./data/processed/data_description.csv")
    df_total.to_excel(f"./data/processed/df_total_1h.xlsx")
    df_total.to_csv(f"./data/processed/df_total_1h.csv")

    return

if __name__ == "__main__":
    main(countries = list(BIDDING_ZONES_CWE.keys()), update_data=True, create_data=True, clustering=True)