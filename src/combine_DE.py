import pandas as pd
import numpy as np

def combine_day_ahead_prices():

    day_ahead_prices_DE_AT_LU = pd.read_csv(f'./data/day_ahead_prices/day_ahead_prices_DE_AT_LU.csv')
    day_ahead_prices_DE_AT_LU.set_index('Unnamed: 0', inplace=True)
    day_ahead_prices_DE_AT_LU.index = pd.to_datetime(day_ahead_prices_DE_AT_LU.index, utc=True)
    day_ahead_prices_DE_AT_LU.rename(columns={day_ahead_prices_DE_AT_LU.columns[0]:f"day_ahead_prices_DE"}, inplace=True)

    day_ahead_prices_DE_LU = pd.read_csv(f'./data/day_ahead_prices/day_ahead_prices_DE_LU.csv')
    day_ahead_prices_DE_LU.set_index('Unnamed: 0', inplace=True)
    day_ahead_prices_DE_LU.index = pd.to_datetime(day_ahead_prices_DE_LU.index, utc=True)
    day_ahead_prices_DE_LU.rename(columns={day_ahead_prices_DE_LU.columns[0]:f"day_ahead_prices_DE"}, inplace=True)
    day_ahead_prices_combined = pd.concat([day_ahead_prices_DE_AT_LU, day_ahead_prices_DE_LU], axis=0)

    day_ahead_prices_combined.to_csv(f"./data/day_ahead_prices/day_ahead_prices_DE.csv")
    return

def combine_load():
    
    load_DE_AT_LU = pd.read_csv(f'./data/load/load_DE_AT_LU.csv')
    load_DE_AT_LU.set_index('Unnamed: 0', inplace=True)
    load_DE_AT_LU.index = pd.to_datetime(load_DE_AT_LU.index, utc=True)
    load_DE_AT_LU.rename(columns={load_DE_AT_LU.columns[0]:f"load_DE"}, inplace=True)

    load_DE_LU = pd.read_csv(f'./data/load/load_DE_LU.csv')
    load_DE_LU.set_index('Unnamed: 0', inplace=True)
    load_DE_LU.index = pd.to_datetime(load_DE_LU.index, utc=True)
    load_DE_LU.rename(columns={load_DE_LU.columns[0]:f"load_DE"}, inplace=True)
    load_combined = pd.concat([load_DE_AT_LU, load_DE_LU], axis=0)

    load_combined.to_csv(f"./data/load/load_DE.csv")
    return

def combine_generation():
    
    generation_DE_AT_LU = pd.read_csv(f'./data/generation/generation_DE_AT_LU.csv', header=[0,1])
    generation_DE_AT_LU.set_index(generation_DE_AT_LU.columns[0], inplace=True)
    generation_DE_AT_LU.index = pd.to_datetime(generation_DE_AT_LU.index, utc=True)

    generation_DE_LU = pd.read_csv(f'./data/generation/generation_DE_LU.csv', header=[0,1])
    generation_DE_LU.set_index(generation_DE_LU.columns[0], inplace=True)
    generation_DE_LU.index = pd.to_datetime(generation_DE_LU.index, utc=True)

    generation_combined = pd.concat([generation_DE_AT_LU, generation_DE_LU], axis=0)

    generation_combined.to_csv(f"./data/generation/generation_DE.csv")
    return

def combine_net_positions():
    
    net_positions_DE_AT_LU = pd.read_csv(f'./data/net_positions/net_positions_DE_AT_LU.csv')
    net_positions_DE_AT_LU.set_index('Unnamed: 0', inplace=True)
    net_positions_DE_AT_LU.index = pd.to_datetime(net_positions_DE_AT_LU.index, utc=True)
    net_positions_DE_AT_LU.rename(columns={net_positions_DE_AT_LU.columns[0]:f"net_positions_DE"}, inplace=True)

    net_positions_DE_LU = pd.read_csv(f'./data/net_positions/net_positions_DE_LU.csv')
    net_positions_DE_LU.set_index('Unnamed: 0', inplace=True)
    net_positions_DE_LU.index = pd.to_datetime(net_positions_DE_LU.index, utc=True)
    net_positions_DE_LU.rename(columns={net_positions_DE_LU.columns[0]:f"net_positions_De"}, inplace=True)
    net_positions_combined = pd.concat([net_positions_DE_AT_LU, net_positions_DE_LU], axis=0)

    net_positions_combined.to_csv(f"./data/net_positions/net_positions_DE.csv")
    return