import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.tools.datetimes import to_datetime

from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances_argmin

from kneed import DataGenerator, KneeLocator

from src.data.create_dataframe import create_dataframe_residual_load
from helper import split_years
from datetime import timedelta, date

def k_means_clustering(df_total: pd.DataFrame, country_code: str):

    # x = df_total[[f"day_ahead_prices_{country_code}", f"price_spread_{country_code}", f"SDM_{country_code}",
    #     f"net_positions_{country_code}", f"load_{country_code}", 
    #     f"non_dispatchable_{country_code}",	f"residual_load_{country_code}"]]

    x = df_total[[f"day_ahead_prices_{country_code}", f"price_spread_{country_code}",
        f"net_positions_{country_code}", f"residual_load_{country_code}"]]
    # Abweichungen etc.
    # Preisunterschiede, Summe Residuale Last

    x = np.nan_to_num(x)
    x_scaled = preprocessing.scale(x)

    kl = KneeLocator(range(1, 24), [(KMeans(n_clusters=i, init="k-means++").fit(x_scaled).inertia_) for i in range(1,24)], curve="convex", direction="decreasing")
    print("Correct ellbow point: " + str(kl.elbow))

    kmeans = KMeans(n_clusters=kl.elbow, init="k-means++").fit(x_scaled)

    idx = np.argsort(kmeans.cluster_centers_.sum(axis=1))
    lut = np.zeros_like(idx)
    lut[idx] = np.arange(kl.elbow)

    (unique, counts) = np.unique(kmeans.labels_, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print(frequencies)

    return lut[kmeans.labels_]