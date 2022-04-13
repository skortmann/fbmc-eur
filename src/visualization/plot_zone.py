import geopandas as gpd
import pandas as pd
import folium
import matplotlib.pyplot as plt
plt.style.use('science')

import matplotlib
# matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size' : 30
})

import color_blinded
plt.cm.register_cmap('sunset', color_blinded.tol_cmap('sunset'))

import bidding_zones
from create_dataframe import create_dataframe_day_ahead_prices, create_dataframe_net_positons, create_dataframe_scheduled_exchanges
from datetime import date

from shapely.ops import unary_union

def plot_zonal_model(country_list: list):

    df_day_ahead_prices = create_dataframe_day_ahead_prices(countries_day_ahead_prices=country_list)

    df_day_ahead_prices = df_day_ahead_prices[date(year=2021, month=10, day=1).isoformat():date(year=2022, month=1, day=1).isoformat()]
    filter_col = [col for col in df_day_ahead_prices if col.startswith(('price_spread_'))]
    df_price_spread = df_day_ahead_prices[filter_col]

    filter_col = [col for col in df_day_ahead_prices if col.startswith(('day_ahead_prices_'))]
    df_day_ahead_prices = df_day_ahead_prices[filter_col]
    df_day_ahead_prices = df_day_ahead_prices.mean(axis=0, skipna=True, numeric_only=True)
    df_day_ahead_prices.index = df_day_ahead_prices.index.str.lstrip("day_ahead_prices_")

    print(df_day_ahead_prices)

    df_scheduled_exchanges = create_dataframe_scheduled_exchanges(countries_scheduled_exchanges=country_list)

    df_scheduled_exchanges = df_scheduled_exchanges[date(year=2021, month=10, day=1).isoformat():date(year=2022, month=1, day=1).isoformat()]
    # filter_col = [col for col in df_net_position if col.startswith(('net_position_'))]
    # df_net_position = df_net_position[filter_col]
    df_scheduled_exchanges = df_scheduled_exchanges.sum(axis=0, skipna=True, numeric_only=True)
    df_scheduled_exchanges.index = df_scheduled_exchanges.index.str.lstrip("scheduled_exchanges")

    print(df_scheduled_exchanges)

    df_net_position = create_dataframe_net_positons(countries_net_positons=country_list)

    df_net_position = df_net_position[date(year=2021, month=10, day=1).isoformat():date(year=2022, month=1, day=1).isoformat()]
    # filter_col = [col for col in df_net_position if col.startswith(('net_position_'))]
    # df_net_position = df_net_position[filter_col]
    df_net_position = df_net_position.sum(axis=0, skipna=True, numeric_only=True)
    df_net_position.index = df_net_position.index.str.lstrip("net_position_")

    print(df_net_position)

    for country in country_list:
        if country == "DE_AT_LU": continue
        try:
            globals()[f"countries_gdf_{country}"]= gpd.read_file(f"./zones/{country}.geojson")
        except:
            print(f"no country given {country}")
            continue

    gdf = gpd.GeoDataFrame(
        pd.concat([globals()[f"countries_gdf_{country}"]
        for country in country_list if country != "DE_AT_LU"], 
        ignore_index=True))

    print(gdf)

    gdf['coords'] = gdf['geometry'].apply(lambda x: x.representative_point().coords[:])
    gdf['coords'] = [coords[0] for coords in gdf['coords']]

    fig, axs = plt.subplots(figsize=(21, 8))
    gdf.plot(legend=True, ax=axs)
    plt.title("Bidding zones in SDAC market")
    for idx, row in gdf.iterrows():
        plt.annotate(text=row['id'], xy=row['coords'],
                    horizontalalignment='center',
                    fontsize=20)
    plt.savefig("./plots/maps/member_countries.pdf")
    # plt.show()
    plt.close('all')

    # Plot different geographic projections
    # fig, axs = plt.subplots(1, 3, figsize=(15, 12))
    # crs_list = [("WGS 84", "EPSG:4326"), ("Lambert", "EPSG:3347"), ("UTM 10N", "EPSG:32610")]
    # for n, (name, epsg) in enumerate(crs_list):
    #     gdf.to_crs(epsg).plot(ax=axs[n])
    #     axs[n].set_title(name)
    # plt.tight_layout()
    # plt.show()

    # gdf.to_crs("EPSG:4326")

    gdf["area"] = gdf.area
    gdf['centroid'] = gdf.centroid

    gdf = gdf.merge(df_day_ahead_prices.to_frame(), how="inner", left_on="zoneName", right_on=df_day_ahead_prices.index)
    gdf = gdf.rename(columns={0: "average_day_ahead_price"})
    gdf = gdf.merge(df_net_position.to_frame(), how="inner", left_on="zoneName", right_on=df_net_position.index)
    gdf = gdf.rename(columns={0: "sum_net_position"})

    first_point = gdf['centroid'].iloc[0]
    gdf['distance'] = gdf['centroid'].distance(first_point)

    gdf["x"] = gdf.centroid.map(lambda p: p.x)
    gdf["y"] = gdf.centroid.map(lambda p: p.y)

    # gdf = gdf.set_index("id")

    # gdf.plot(legend=True, figsize=(10,10))
    # plt.title("Member countries of flow based market coupling region")
    # # plt.show()
    # plt.close('all')


    fig, ax = plt.subplots(figsize=(21, 8))
    gdf.plot("average_day_ahead_price", legend=True, ax=ax, legend_kwds={'label': 'in [€]','shrink': 0.5}, cmap="sunset")
    # plt.title("Average day ahead price for SDAC including CWE for 2021")
    for idx, row in gdf.iterrows():
        plt.annotate(text=row['id'], xy=row['coords'],
                    horizontalalignment='center',
                    fontsize=20)
    # plt.show()
    plt.savefig('./plots/maps/average_price.pdf', dpi=1200)
    plt.close('all')

    # with plt.style.context(("seaborn", "ggplot")):
    #     gdf.plot("area", legend=True, figsize=(15,15), edgecolor="grey", color="white")
    #     plt.title("Interconnection map of CWE region")
    #     plt.savefig("./plots/maps/plain_connection_map_CWE.png", dpi=1200)
    #     plt.show()

    # fig, ax = plt.subplots(figsize=(21, 8))
    # gdf.plot("average_day_ahead_price", ax=ax, legend=True, legend_kwds={'label': "in [€]"}, cmap="sunset")

    # for idx, row in gdf.iterrows():
    #     plt.annotate(text=row['id'], xy=row['coords'],
    #                 horizontalalignment='center')
    # # for i in country_list:
    # #     for j in country_list:
    # #         if i == "DE_AT_LU": continue
    # #         if j == "DE_AT_LU": continue
    # #         if i == j: continue
    # #         plt.plot([gdf.loc[i, "x"], gdf.loc[j, "x"]], [gdf.loc[i, "y"], gdf.loc[j, "y"]], linewidth = 2, linestyle = "-", color="k")
    # #         plt.scatter([gdf.loc[i, "x"], gdf.loc[j, "x"]], [gdf.loc[i, "y"], gdf.loc[j, "y"]], alpha=0.1)
            
    # plt.title("Average day ahead price in FBMC region for 2021")
    # plt.tight_layout()
    # # plt.savefig("./plots/maps/average_day_ahead_prices_connection_map_CWE.pdf")
    # plt.show()
    # plt.close('all')

    fig, ax = plt.subplots(figsize=(21, 8))
    gdf.plot("sum_net_position", ax=ax, legend=True, legend_kwds={'label': 'in [MWh]','shrink': 0.5}, cmap="sunset")
    for idx, row in gdf.iterrows():
        plt.annotate(text=row['id'], xy=row['coords'],
                    horizontalalignment='center',
                    fontsize=20)

    # for i in country_list:
    #     for j in country_list:
    #         if i == "DE_AT_LU": continue
    #         if j == "DE_AT_LU": continue
    #         if i == j: continue
    #         plt.plot([gdf.loc[i, "x"], gdf.loc[j, "x"]], [gdf.loc[i, "y"], gdf.loc[j, "y"]], linewidth = 2, linestyle = "-", color="k")
    #         plt.scatter([gdf.loc[i, "x"], gdf.loc[j, "x"]], [gdf.loc[i, "y"], gdf.loc[j, "y"]], alpha=0.1)
            
    # plt.title("Sum of net position for FBMC region in 2021")
    plt.tight_layout()
    plt.savefig('./plots/maps/sum_net_positions.pdf', dpi=1200)
    # plt.show()
    plt.close('all')

    return

plot_zonal_model(country_list=bidding_zones.BIDDING_ZONES.keys())