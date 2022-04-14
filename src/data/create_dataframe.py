from numpy.ma import count
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta, tzinfo
import os
import logging

from pandas.io.parsers import read_csv
from bidding_zones import *

def create_dataframe_day_ahead_prices(countries_day_ahead_prices: list, remove_outlier: bool = False) -> pd.DataFrame:

    logging.info('Creating dataframe for day ahead prices')
    df_total = pd.DataFrame()

    for country in countries_day_ahead_prices:
        day_ahead_prices = pd.read_csv(f'./data/raw/day_ahead_prices/day_ahead_prices_{country}.csv')
        day_ahead_prices.set_index('Unnamed: 0', inplace=True)
        day_ahead_prices.index = pd.to_datetime(day_ahead_prices.index, utc=True)
        day_ahead_prices.rename(columns={"0":f"{country}"}, inplace=True)
        df_total = df_total.join(day_ahead_prices, how='outer')

    df_total.index.names = ['Date']
    df_total.index = pd.to_datetime(df_total.index, utc=True)
    df_total.index = df_total.index.tz_localize(None)
    df_total = df_total.sort_index()

    df_total["mean_price"] = df_total.mean(axis=1, skipna=True, numeric_only=True)
    df_total["price_spread_total"] = df_total.max(axis=1, skipna=True, numeric_only=True) - df_total.min(axis=1, skipna=True, numeric_only=True)
    for country in countries_day_ahead_prices:
        df_total[f"price_spread_{country}"] = (
            np.subtract(df_total["mean_price"].to_numpy(),df_total[f"day_ahead_prices_{country}"].to_numpy()))
        df_total[f"SDM_{country}"] = (
            np.square(np.subtract(df_total["mean_price"].to_numpy(),df_total[f"day_ahead_prices_{country}"].to_numpy())))

    df_total["relative_price_spread"] = np.divide(df_total["price_spread_total"], df_total["mean_price"])*100
    # print(df_total[df_total < 0.0].count())
    # df_total["price_spread_category"] = pd.cut(df_total["price_spread_total"], bins=[0,1,5,10,20,50,100,np.Infinity])

    # df_total.dropna(how='all', axis=0, inplace=True)
    # df_total.to_csv("day_ahead_prices_total.csv")

    if remove_outlier:
        Q1 = df_total.quantile(0.25)
        Q3 = df_total.quantile(0.75)
        IQR = Q3 - Q1
        df_total = df_total[~((df_total < (Q1 - 1.5 * IQR)) |(df_total > (Q3 + 1.5 * IQR))).any(axis=1)]

    df_total.to_csv("./data/processed/day_ahead_prices.csv")
    print("Finished day ahead prices")

    return df_total

def create_dataframe_generation(countries_generation: list, remove_outlier: bool = False) -> pd.DataFrame:
    df_total = pd.DataFrame()

    for country in countries_generation:
        if country in list(BIDDING_ZONES_NON_DEFAULT.keys()):  
            generation = pd.read_csv(f'./data/raw/generation/generation_{country}.csv', index_col=0)
            generation.index = pd.to_datetime(generation.index, utc=True)
        ## TODO
        # write specific function to import dataframes with wrong header
        elif country in list(BIDDING_ZONES_WRONG_HEADER.keys()):  
            generation = pd.read_csv(f'./data/raw/generation/generation_{country}.csv', index_col=0)
            generation.index = pd.to_datetime(generation.index, utc=True)
        else:
            generation = pd.read_csv(f'./data/raw/generation/generation_{country}.csv', index_col=0, header=[0,1])
            generation.drop(["Actual Consumption"], axis=1, level=1, inplace=True)
            generation.index = pd.to_datetime(generation.index, utc=True)
            generation = generation.droplevel(level=1, axis=1)
        df_total = df_total.join(generation, how='outer', lsuffix=f"_{country}")

    df_total.index.names = ['Date']
    df_total.index = pd.to_datetime(df_total.index, utc=True)
    df_total.index = df_total.index.tz_localize(None)
    df_total = df_total.sort_index()
    # df_total.dropna(how='all', axis=0, inplace=True)
    # df_total.to_csv("generation_total.csv")

    if remove_outlier:
        Q1 = df_total.quantile(0.25)
        Q3 = df_total.quantile(0.75)
        IQR = Q3 - Q1
        df_total = df_total[~((df_total < (Q1 - 1.5 * IQR)) |(df_total > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    df_total.to_csv("./data/processed/generation.csv")
    print("Finished generation")

    return df_total

def create_dataframe_load(countries_load: list, remove_outlier: bool = False) -> pd.DataFrame:
    df_total = pd.DataFrame()

    for country in countries_load:
        load = pd.read_csv(f'./data/raw/load/load_{country}.csv')
        load.set_index('Unnamed: 0', inplace=True)
        load.index = pd.to_datetime(load.index, utc=True)
        load.rename(columns={"0":f"{country}"}, inplace=True)
        df_total = df_total.join(load, how='outer')

    df_total.index.names = ['Date']
    df_total.index = pd.to_datetime(df_total.index, utc=True)
    df_total.index = df_total.index.tz_localize(None)
    df_total = df_total.sort_index()
    # df_total.dropna(how='all', axis=0, inplace=True)
    # df_total.to_csv("day_ahead_prices_total.csv")

    if remove_outlier:
        Q1 = df_total.quantile(0.25)
        Q3 = df_total.quantile(0.75)
        IQR = Q3 - Q1
        df_total = df_total[~((df_total < (Q1 - 1.5 * IQR)) |(df_total > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    df_total.to_csv("./data/processed/load.csv")
    print("Finished load")

    return df_total

def create_dataframe_scheduled_exchanges(countries_scheduled_exchanges: list, remove_outlier: bool = False) -> pd.DataFrame:
    df_total = pd.DataFrame()

    for country_code_from in countries_scheduled_exchanges:
        for country_code_to in countries_scheduled_exchanges:
            if country_code_from == country_code_to : continue
            try:
                scheduled_exchanges = pd.read_csv(f'./data/raw/scheduled_exchanges/scheduled_exchanges_{country_code_from}_{country_code_to}.csv')
                scheduled_exchanges.set_index('Unnamed: 0', inplace=True)
                scheduled_exchanges.index = pd.to_datetime(scheduled_exchanges.index, utc=True)
                scheduled_exchanges.rename(columns={scheduled_exchanges.columns[0]:f"scheduled_exchanges_{country_code_from}_to_{country_code_to}"}, inplace=True)
                df_total = df_total.join(scheduled_exchanges, how='outer')
            except:
                continue

    df_total.index.names = ['Date']
    df_total.index = pd.to_datetime(df_total.index, utc=True)
    df_total.index = df_total.index.tz_localize(None)
    df_total = df_total.sort_index()
    # df_total.dropna(how='all', axis=0, inplace=True)
    # df_total.to_csv("day_ahead_prices_total.csv")

    if remove_outlier:
        Q1 = df_total.quantile(0.25)
        Q3 = df_total.quantile(0.75)
        IQR = Q3 - Q1
        df_total = df_total[~((df_total < (Q1 - 1.5 * IQR)) |(df_total > (Q3 + 1.5 * IQR))).any(axis=1)]

    df_total.to_csv("./data/processed/scheduled_exchanges.csv")
    print("Finished scheduled exchanges")
    
    return df_total

def create_dataframe_residual_load(countries_load: list, remove_outlier: bool = False) -> pd.DataFrame:
    df_total = pd.DataFrame()

    for country in countries_load:
        load = pd.read_csv(f'./data/raw/load/load_{country}.csv')
        load.set_index('Unnamed: 0', inplace=True)
        load.index = pd.to_datetime(load.index, utc=True)

        if country in list(BIDDING_ZONES_NON_DEFAULT.keys()):
            generation = pd.read_csv(f'./data/raw/generation/generation_{country}.csv', index_col=0)
            generation.index = pd.to_datetime(generation.index, utc=True)
        ## TODO
        # write specific function to import dataframes with wrong header
        elif country in list(BIDDING_ZONES_WRONG_HEADER.keys()):  
            generation = pd.read_csv(f'./data/raw/generation/generation_{country}.csv', index_col=0)
            generation.index = pd.to_datetime(generation.index, utc=True)
        else:
            generation = pd.read_csv(f'./data/raw/generation/generation_{country}.csv', index_col=0, header=[0,1])
            generation.drop(["Actual Consumption"], axis=1, level=1, inplace=True)
            generation.index = pd.to_datetime(generation.index, utc=True)
            generation = generation.droplevel(level=1, axis=1)

        non_fuel_cost = ['Wind Offshore', 'Wind Onshore', 'Solar', 'Other renewable', 'Geothermal', 'Biomass', 
        'Hydro Pumped Storage', 'Hydro Run-of-river and poundage','Hydro Water Reservoir', 'Waste']

        filter_col = [col for col in generation if col.startswith(('Wind', 'Solar', 'Hydro Run-of-river'))]
        generation[f"non_dispatchable_{country}"] = generation[filter_col].sum(axis=1)
        globals()[f"residual_load_{country}"] = load.join(generation[f"non_dispatchable_{country}"], how='outer')
        globals()[f"residual_load_{country}"][f"load_{country}"].interpolate(method='time', limit_direction='forward', axis=0, inplace=True)
        globals()[f"residual_load_{country}"][f"load_{country}"].interpolate(method='time', limit_direction='backward', axis=0, inplace=True)
        globals()[f"residual_load_{country}"][f"non_dispatchable_{country}"].interpolate(method='time', limit_direction='forward', axis=0, inplace=True)
        globals()[f"residual_load_{country}"][f"non_dispatchable_{country}"].interpolate(method='time', limit_direction='backward', axis=0, inplace=True)
        globals()[f"residual_load_{country}"][f"residual_load_{country}"] = globals()[f"residual_load_{country}"][f"load_{country}"] - globals()[f"residual_load_{country}"][f"non_dispatchable_{country}"]
        globals()[f"residual_load_{country}"].to_csv(f"./data/raw/residual_load/residual_load_{country}.csv")
        df_total = df_total.join(globals()[f"residual_load_{country}"], how='outer') 

    df_total.index.names = ['Date']
    df_total.index = pd.to_datetime(df_total.index, utc=True)
    df_total.index = df_total.index.tz_localize(None)
    df_total = df_total.sort_index()
    df_total["sum_residual_load"] = df_total.sum(axis=1, skipna=True, numeric_only=True)
    # df_total.dropna(how='all', axis=0, inplace=True)
    # df_total.to_csv("day_ahead_prices_total.csv")

    if remove_outlier:
        Q1 = df_total.quantile(0.25)
        Q3 = df_total.quantile(0.75)
        IQR = Q3 - Q1
        df_total[~((df_total < (Q1 - 1.5 * IQR)) |(df_total > (Q3 + 1.5 * IQR))).any(axis=1)] = np.nan

    df_total.to_csv("./data/processed/residual_load.csv")
    print("Finished residual loads")
    
    return df_total

def create_dataframe_net_positons(countries_net_positons: list, remove_outlier: bool = False) -> pd.DataFrame:
    df_total = pd.DataFrame()

    for country in countries_net_positons:
        net_positons = pd.read_csv(f'./data/raw/net_positions/net_positions_{country}.csv')
        net_positons.set_index('Unnamed: 0', inplace=True)
        net_positons.index = pd.to_datetime(net_positons.index, utc=True)
        net_positons.rename(columns={"0":f"{country}"}, inplace=True)
        df_total = df_total.join(net_positons, how='outer')

    df_total.index.names = ['Date']
    df_total.index = pd.to_datetime(df_total.index, utc=True)
    df_total.index = df_total.index.tz_localize(None)
    df_total = df_total.sort_index()
    # df_total.dropna(how='all', axis=0, inplace=True)
    # df_total.to_csv("day_ahead_prices_total.csv")

    if remove_outlier:
        Q1 = df_total.quantile(0.25)
        Q3 = df_total.quantile(0.75)
        IQR = Q3 - Q1
        df_total = df_total[~((df_total < (Q1 - 1.5 * IQR)) |(df_total > (Q3 + 1.5 * IQR))).any(axis=1)]

    df_total.to_csv("./data/processed/net_positions.csv")
    print("Finished net positions")
    
    return df_total

def create_dataframe_net_transfer_capacity_day_ahead(countries_net_transfer_capacity_day_ahead: list, remove_outlier: bool = False) -> pd.DataFrame:
    df_total = pd.DataFrame()

    for country_code_from in countries_net_transfer_capacity_day_ahead:
        for country_code_to in countries_net_transfer_capacity_day_ahead:
            if country_code_from == country_code_to : continue
            try:
                net_transfer_capacity_day_ahead = pd.read_csv(f'./data/raw/net_transfer_capacity_day_ahead/net_transfer_capacity_day_ahead_{country_code_from}_{country_code_to}.csv')
                net_transfer_capacity_day_ahead.set_index('Unnamed: 0', inplace=True)
                net_transfer_capacity_day_ahead.index = pd.to_datetime(net_transfer_capacity_day_ahead.index, utc=True)
                net_transfer_capacity_day_ahead.rename(columns={net_transfer_capacity_day_ahead.columns[0]:f"net_transfer_capacity_day_ahead_{country_code_from}_{country_code_to}"}, inplace=True)
                df_total = df_total.join(net_transfer_capacity_day_ahead, how='outer')
            except:
                continue

    df_total.index.names = ['Date']
    df_total.index = pd.to_datetime(df_total.index, utc=True)
    df_total.index = df_total.index.tz_localize(None)
    df_total = df_total.sort_index()
    # df_total.dropna(how='all', axis=0, inplace=True)
    # df_total.to_csv("day_ahead_prices_total.csv")

    if remove_outlier:
        Q1 = df_total.quantile(0.25)
        Q3 = df_total.quantile(0.75)
        IQR = Q3 - Q1
        df_total = df_total[~((df_total < (Q1 - 1.5 * IQR)) |(df_total > (Q3 + 1.5 * IQR))).any(axis=1)]

    df_total.to_csv("./data/processed/net_transfer_capacity_day_ahead.csv")
    print("Finished net transfer capacity day ahead")
    
    return df_total