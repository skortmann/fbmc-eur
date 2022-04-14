import os
import datetime
from datetime import datetime

import pandas as pd
from entsoe import EntsoePandasClient

from bidding_zones import *
from helper import TimeoutException
import src.data.parse_countries

client = EntsoePandasClient(api_key="e3547493-771f-47ae-9df6-d61cee2a5ffa")

start = pd.Timestamp('20140101', tz='Europe/Brussels')
end = pd.Timestamp(datetime.strftime(datetime.today(), "%Y%m%d"), tz='Europe/Brussels')

def import_load(countries_load: list) -> object:

    for country in countries_load:
        print(f"Current import of load for country-code: {country}")
        try:
            globals()[f"load_{country}"] = client.query_load(country, start=start, end=end)
            globals()[f"load_{country}"].index = pd.to_datetime(
                    globals()[f"load_{country}"].index)
            globals()[f"load_{country}"] = pd.DataFrame(
                    globals()[f"load_{country}"])
            globals()[f"load_{country}"].rename(
                columns={globals()[f"load_{country}"].columns[0]: 
                f"load_{country}"}, inplace=True)

            globals()[f"load_{country}"].to_csv(f"./data/raw/load/load_{country}.csv")

        except:
            print(f"failed importing load for {country}")
            continue
    return

def import_wind_solar_forecast(countries_wind_solar_forecast: list) -> object:

    for country in countries_wind_solar_forecast:
        print(f"Current import of load for country-code: {country}")
        try:
            globals()[f"wind_solar_forecast_{country}"] = client.query_wind_and_solar_forecast(country, start=start,end=end, psr_type=None)
            globals()[f"wind_solar_forecast_{country}"].index = pd.to_datetime(
                    globals()[f"wind_solar_forecast_{country}"].index)
            globals()[f"wind_solar_forecast_{country}"] = pd.DataFrame(
                    globals()[f"wind_solar_forecast_{country}"])
            globals()[f"wind_solar_forecast_{country}"].rename(
                columns={globals()[f"wind_solar_forecast_{country}"].columns[0]: 
                f"wind_solar_forecast_{country}"}, inplace=True)
            globals()[f"wind_solar_forecast_{country}"].add_suffix('_forecast')

            globals()[f"wind_solar_forecast_{country}"].to_csv(f"./data/raw/wind_solar_forecast/wind_solar_forecast_{country}.csv")

        except:
            print(f"failed importing wind_solar_forecast for {country}")
            continue
    return

def import_generation(countries_generation: list) -> object:

    list_of_no_generation_import = ["AT", "IT_SUD", "IT_CNOR", "NL"]
    for country in list_of_no_generation_import:
        src.data.parse_countries.parser(country_code=country)

    for country in countries_generation:
        if country in list_of_no_generation_import: continue
        print(f"Current import of generation for country-code: {country}")
        try:
            globals()[f"generation_{country}"] = client.query_generation(country, start=start, end=end, psr_type=None)
                
            globals()[f"generation_{country}"].index = pd.to_datetime(
                    globals()[f"generation_{country}"].index)
            globals()[f"generation_{country}"] = pd.DataFrame(
                    globals()[f"generation_{country}"])
            # globals()[f"generation_{country}"].rename(
            #     columns={globals()[f"generation_{country}"].columns[0]: 
            #     f"generation_{country}"}, inplace=True)

            globals()[f"generation_{country}"].to_csv(f"./data/raw/generation/generation_{country}.csv")

            # globals()[f"generation_non-dispatchable_{country}"] = globals()[f"generation_{country}"].loc[:, ['Solar', 'Wind Offshore', 'Wind Onshore']]
            # globals()[f"generation_non-dispatchable_{country}"].to_csv(f"./data/raw/generation/generation_non-dispatch-able_{country}.csv")
        except:
            print(f"failed importing generation for {country}")
            continue
    return


def import_day_ahead_prices(countries_day_ahead_prices: list) -> object:

    for country in countries_day_ahead_prices:
        print(f"Current import of day-ahead-prices for country-code: {country}")
        try:
            globals()[f"day_ahead_prices_{country}"] = client.query_day_ahead_prices(country, start=start, end=end)
            globals()[f"day_ahead_prices_{country}"].index = pd.to_datetime(
                    globals()[f"day_ahead_prices_{country}"].index)
            globals()[f"day_ahead_prices_{country}"] = pd.DataFrame(
                    globals()[f"day_ahead_prices_{country}"])
            globals()[f"day_ahead_prices_{country}"].rename(
                columns={globals()[f"day_ahead_prices_{country}"].columns[0]: 
                f"day_ahead_prices_{country}"}, inplace=True)

            globals()[f"day_ahead_prices_{country}"].to_csv(
                f"./data/raw/day_ahead_prices/day_ahead_prices_{country}.csv")

        except TimeoutException:
            print('Timeout after processing N records')
            print(f"failed importing day ahead prices for {country}")
            continue
        except Exception as e:
            print(e)
            print(f"failed importing day ahead prices for {country}")
            continue
    return


def import_scheduled_exchanges(countries_scheduled_exchanges: list) -> object:

    for country_code_from in countries_scheduled_exchanges:
        for country_code_to in countries_scheduled_exchanges:
            if country_code_from == country_code_to : continue
            print(f"Current import of scheduled import from country-code: {country_code_from}")
            print(f"Current export of scheduled export to country-code: {country_code_to}")
            try:

                globals()[
                    f"scheduled_exchanges_{country_code_from}_{country_code_to}"] = client.query_scheduled_exchanges(
                        country_code_from, country_code_to, start=start, end=end, dayahead=True)
                globals()[f"scheduled_exchanges_{country_code_from}_{country_code_to}"].index = pd.to_datetime(
                        globals()[f"scheduled_exchanges_{country_code_from}_{country_code_to}"].index)
                globals()[f"scheduled_exchanges_{country_code_from}_{country_code_to}"] = pd.DataFrame(
                        globals()[f"scheduled_exchanges_{country_code_from}_{country_code_to}"])
                globals()[f"scheduled_exchanges_{country_code_from}_{country_code_to}"].rename(
                    columns={globals()[f"scheduled_exchanges_{country_code_from}_{country_code_to}"].columns[0]: 
                    f"scheduled_exchanges_{country_code_from}-{country_code_to}"}, inplace=True)

                globals()[f"scheduled_exchanges_{country_code_from}_{country_code_to}"].to_csv(
                    f"./data/raw/scheduled_exchanges/scheduled_exchanges_{country_code_from}_{country_code_to}.csv")

            except:
                print(f"Failed at import from {country_code_from} and export to {country_code_to}")
                continue
    return

def import_net_positions(countries_net_positions: list) -> object:
    directory_net_positions = os.path.dirname("./data/raw/net_positions/")

    if not os.path.exists(directory_net_positions):
        os.makedirs(directory_net_positions)

    for country in countries_net_positions:
        print(f"Current import of net positions for country-code: {country}")
        try:
            globals()[f"net_positions_{country}"] = client.query_net_position_dayahead(country, start=start, end=end)
            globals()[f"net_positions_{country}"].index = pd.to_datetime(
                    globals()[f"net_positions_{country}"].index)
            globals()[f"net_positions_{country}"] = pd.DataFrame(
                    globals()[f"net_positions_{country}"])
            globals()[f"net_positions_{country}"].rename(
                columns={globals()[f"net_positions_{country}"].columns[0]: 
                f"net_positions_{country}"}, inplace=True)

            globals()[f"net_positions_{country}"].to_csv(
                f"./data/raw/net_positions/net_positions_{country}.csv")

        except Exception as e:
            print(e)
            print(f"failed importing net positions for {country}")
            continue
    return

def import_net_transfer_capacity_day_ahead(countries_net_transfer_capacity_day_ahead: list) -> object:

    for country_code_from in countries_net_transfer_capacity_day_ahead:
        for country_code_to in countries_net_transfer_capacity_day_ahead:
            if country_code_from == country_code_to : continue
            print(f"Current import of net transfer capacity from country-code: {country_code_from}")
            print(f"Current export of net transfer capacity to country-code: {country_code_to}")
            try:
                globals()[
                    f"net_transfer_capacity_day_ahead_{country_code_from}_{country_code_to}"] = client.query_net_transfer_capacity_yearahead(
                        country_code_from=country_code_from, country_code_to=country_code_to, start=start, end=end)
                globals()[f"net_transfer_capacity_day_ahead_{country_code_from}_{country_code_to}"].index = pd.to_datetime(
                        globals()[f"net_transfer_capacity_day_ahead_{country_code_from}_{country_code_to}"].index)
                globals()[f"net_transfer_capacity_day_ahead_{country_code_from}_{country_code_to}"] = pd.DataFrame(
                        globals()[f"net_transfer_capacity_day_ahead_{country_code_from}_{country_code_to}"])
                globals()[f"net_transfer_capacity_day_ahead_{country_code_from}_{country_code_to}"].rename(
                    columns={globals()[f"net_transfer_capacity_day_ahead_{country_code_from}_{country_code_to}"].columns[0]: 
                    f"net_transfer_capacity_day_ahead_{country_code_from}-{country_code_to}"}, inplace=True)

                globals()[f"net_transfer_capacity_day_ahead_{country_code_from}_{country_code_to}"].to_csv(
                    f"./data/raw/net_transfer_capacity_day_ahead/net_transfer_capacity_day_ahead_{country_code_from}_{country_code_to}.csv")
            except:
                print(f"Failed at import from {country_code_from} and export to {country_code_to}")
                continue
    return