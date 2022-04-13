import os

# create folder for data
directory = os.path.dirname("./data/")

if not os.path.exists(directory):
    os.makedirs(directory)

# create folder for dataframes
directory_dataframes = os.path.dirname("./data/processed/")

if not os.path.exists(directory_dataframes):
    os.makedirs(directory_dataframes)

# create folder for load
directory_load = os.path.dirname("./data/raw/load/")

if not os.path.exists(directory_load):
    os.makedirs(directory_load)

# create folder for residual load
directory_residual_load = os.path.dirname("./data/raw/residual_load/")

if not os.path.exists(directory_residual_load):
    os.makedirs(directory_residual_load)

# create folder for day ahead prices
directory_day_ahead_prices = os.path.dirname("./data/raw/day_ahead_prices/")

if not os.path.exists(directory_day_ahead_prices):
    os.makedirs(directory_day_ahead_prices)

# create folder for generation
directory_generation = os.path.dirname("./data/raw/generation/")

if not os.path.exists(directory_generation):
    os.makedirs(directory_generation)

#create folder for scheduled exchanges
directory_scheduled_exchanges = os.path.dirname("./data/raw/scheduled_exchanges/")

if not os.path.exists(directory_scheduled_exchanges):
    os.makedirs(directory_scheduled_exchanges)
        
#create folder for net transfer capacity day ahead
directory_net_transfer_capacity_day_ahead = os.path.dirname("./data/raw/net_transfer_capacity_day_ahead/")

if not os.path.exists(directory_net_transfer_capacity_day_ahead):
    os.makedirs(directory_net_transfer_capacity_day_ahead)

# create folder for times series cluster
times_series_cluster_dir = os.path.dirname("./plots/time_series_clustering/")

if not os.path.exists(times_series_cluster_dir):
    os.makedirs(times_series_cluster_dir)
   
# create folder for boxplot
boxplot_dir = os.path.dirname("./plots/boxplots/")

if not os.path.exists(boxplot_dir):
    os.makedirs(boxplot_dir)

# create folder for scatter matrix
scatter_matrix_dir = os.path.dirname("./plots/scatter_matrix/")

if not os.path.exists(scatter_matrix_dir):
    os.makedirs(scatter_matrix_dir)

# create folder for dispatch
dispatch_dir = os.path.dirname("./plots/dispatch/")

if not os.path.exists(dispatch_dir):
    os.makedirs(dispatch_dir)

# create folder for price spread
price_spread_dir = os.path.dirname("./plots/price_spread/")

if not os.path.exists(price_spread_dir):
    os.makedirs(price_spread_dir)


# create folder for cross border
cross_border_dir = os.path.dirname("./plots/cross_border/")

if not os.path.exists(cross_border_dir):
    os.makedirs(cross_border_dir)

# create folder for decision tree
decision_tree_dir = os.path.dirname("./plots/decision_tree/")

if not os.path.exists(decision_tree_dir):
    os.makedirs(decision_tree_dir)

# create folder for decision tree
random_forest_dir = os.path.dirname("./plots/random_forest/")

if not os.path.exists(random_forest_dir):
    os.makedirs(random_forest_dir)

# create folder for maps
maps_dir = os.path.dirname("./plots/maps/")

if not os.path.exists(maps_dir):
    os.makedirs(maps_dir)

# # create folder for description
# description_dir = os.path.dirname("./data/description/")

# if not os.path.exists(description_dir):
#     os.makedirs(description_dir)