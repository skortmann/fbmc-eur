import pandas as pd
import numpy as np
from pandas.core.tools.datetimes import to_datetime

from tslearn.clustering import TimeSeriesKMeans, KShape
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler, TimeSeriesScalerMinMax

from src.data.create_dataframe import create_dataframe_residual_load
from helper import split_years
from datetime import timedelta, date

import matplotlib.pyplot as plt
from matplotlib import cm
plt.style.use('science')
# import matplotlib
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
#     'font.size' : 30
# })

import color_blinded
plt.cm.register_cmap('rainbow_discrete', color_blinded.tol_cmap('rainbow_discrete'))
plt.cm.register_cmap('sunset', color_blinded.tol_cmap('sunset'))

def time_series_clustering(country_code: str, split_by_year:bool = False, 
    plot_knee: bool=False, plot_cluster_euclidean_k_means: bool = True,
    plot_cluster_comparions: bool = False):

    print(f"Starting time series clustering for {country_code}")

    data = create_dataframe_residual_load(countries_load=[country_code], remove_outlier=False)
    data = data.resample('H').mean()

    data = data[date(year=2018, month=12, day=31).isoformat():date(year=2020, month=1, day=1).isoformat()]

    if split_by_year:
        list_df, number_of_years = split_years(data, country_code)
        data = list_df[1][f"residual_load_{country_code}"]

    from_date = data.index[0].round('D', nonexistent='shift_forward')+timedelta(1)
    to_date = data.index[-1].round('D', nonexistent='shift_backward')-timedelta(1)
    delta = to_date - from_date

    # create partition
    data  = data[from_date:]
    data  = data[:delta.days*24]

    # specify data
    data[f"residual_load_{country_code}"].replace(to_replace=0, value=np.nan, inplace=True)
    data[f"residual_load_{country_code}"].fillna(method="ffill", inplace=True)
    X_train = data[f"residual_load_{country_code}"].to_numpy()
    X_train = np.reshape(X_train, (delta.days,24,1))
    X_train = np.nan_to_num(X_train)

    # Transform time series
    # X_fit = TimeSeriesScalerMinMax().fit_transform(X_train)
    # X_fit = TimeSeriesScalerMeanVariance().fit_transform(X_train)


    # Make time series shorter
    X_train = TimeSeriesResampler(sz=24).fit_transform(X_train)
    sz = X_train.shape[1]

    from kneed import DataGenerator, KneeLocator

    kl = KneeLocator(range(1, 24), [(TimeSeriesKMeans(n_clusters=i, verbose=False).fit(X_train).inertia_) for i in range(1,24)], curve="convex", direction="decreasing")
    print("Correct ellbow point: " + str(kl.elbow))

    if plot_knee:
        kl.plot_knee()
        plt.savefig(f"./plots/time_series_clustering/plot_knee_{country_code}.png")
        # plt.show()

    # Euclidean k-means
    print("Euclidean k-means")
    # X = TimeSeriesScalerMinMax().fit_transform(X_train)
    km = TimeSeriesKMeans(n_clusters=kl.elbow, verbose=True)
    y_pred = km.fit_predict(X_train)

    ## TODO
    # Cluster der Größe nach sortieren

    idx = np.argsort(km.cluster_centers_.sum(axis=1), axis=0).flatten()
    print("Cluster according to their size")
    print(idx)
    lut = np.zeros_like(idx)
    lut[idx] = np.arange(kl.elbow)

    (unique, counts) = np.unique(km.labels_, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print(frequencies)

    data[f"cluster_time_series_{country_code}"] = np.repeat(lut[km.labels_] + 1, 24).T

    if plot_cluster_euclidean_k_means:
        plt.figure(figsize=(48,5))
        for yi in range(kl.elbow):
            plt.subplot(1, kl.elbow, yi + 1)
            for xx in X_train[y_pred == yi]:
                plt.plot(xx.ravel(), "k-", alpha=.2)
            plt.plot(km.cluster_centers_[yi].ravel(), "r-")

            # plot trend line
            pfit = np.polyfit(np.arange(len(km.cluster_centers_[yi].ravel())), km.cluster_centers_[yi].ravel(), 0)
            trend_line_model = np.poly1d(pfit)
            plt.plot(np.arange(len(km.cluster_centers_[yi].ravel())), trend_line_model(np.arange(len(km.cluster_centers_[yi].ravel()))), "b--") 
            
            plt.xlim(0, sz)
            plt.ylim(min(np.min(X_train),0), np.max(X_train))
            plt.xlabel("hour", fontsize=24)
            plt.ylabel("residual load [MW]", fontsize=24)
            plt.ylim(0,75000)

            plt.text(0.55, 0.15,'Cluster %d [%d]' % (yi +1, frequencies[yi][1]),
                    transform=plt.gca().transAxes)
            if yi == 1:
                plt.suptitle(f"Euclidean $k$-means time series clustering for bidding zone {country_code} in year 2019", fontsize=40)
        # plt.tight_layout()
        plt.savefig(f"./plots/time_series_clustering/time_series_clustering_{country_code}_{kl.elbow}_cluster_2019.pdf", dpi=1200)
        # plt.show()


    if plot_cluster_comparions:
        plt.figure(figsize=(16,12))
        for yi in range(kl.elbow):
            plt.subplot(3, kl.elbow, yi + 1)
            for xx in X_train[y_pred == yi]:
                plt.plot(xx.ravel(), "k-", alpha=.2)
            plt.plot(km.cluster_centers_[yi].ravel(), "r-")

            # print trend line
            pfit = np.polyfit(np.arange(len(km.cluster_centers_[yi].ravel())), km.cluster_centers_[yi].ravel(), 0)
            trend_line_model = np.poly1d(pfit)
            # print(trend_line_model)
            plt.plot(np.arange(len(km.cluster_centers_[yi].ravel())), trend_line_model(np.arange(len(km.cluster_centers_[yi].ravel()))), "b--", label="average") 
            # plt.legend(loc="upper right")
            
            plt.xlim(0, sz)
            plt.ylim(min(np.min(X_train),0), round(np.max(X_train)+10000,-5))
            # plt.ylim(-4,4)
            # plt.ylim(0,1)

            plt.text(0.55, 0.85,'Cluster %d' % (yi),
                    transform=plt.gca().transAxes)
            if yi == 1:
                plt.title("Euclidean $k$-means")

        # DBA-k-means
        print("DBA k-means")
        dba_km = TimeSeriesKMeans(n_clusters=kl.elbow,
                    n_init=2,
                    metric="dtw",
                    verbose=False,
                    max_iter_barycenter=10)
        y_pred = dba_km.fit_predict(X_train)

        labels_dba = dba_km.labels_

        (unique, counts) = np.unique(labels_dba, return_counts=True)
        frequencies_dba = np.asarray((unique, counts)).T
        print(frequencies_dba)

        for yi in range(kl.elbow):
            plt.subplot(3, kl.elbow, yi + kl.elbow + 1)
            for xx in X_train[y_pred == yi]:
                plt.plot(xx.ravel(), "k-", alpha=.2)
            plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
            plt.xlim(0, sz)
            # plt.ylim(0,1)
            # plt.ylim(0, round(np.max(X_train),-5))
            plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
                    transform=plt.gca().transAxes)
            if yi == 1:
                plt.title("DBA $k$-means")

        # kShape clustering
        print("kShape clustering")
        X = TimeSeriesScalerMeanVariance().fit_transform(X_train)
        ks = KShape(n_clusters=kl.elbow, random_state=0)
        y_pred = ks.fit_predict(X)

        labels_ks = ks.labels_

        (unique, counts) = np.unique(labels_ks, return_counts=True)
        frequencies_ks = np.asarray((unique, counts)).T
        print(frequencies_ks)

        for yi in range(kl.elbow):
            plt.subplot(3, kl.elbow, yi + kl.elbow + kl.elbow +1)
            for xx in X[y_pred == yi]:
                plt.plot(xx.ravel(), "k-", alpha=.2)
            plt.plot(ks.cluster_centers_[yi].ravel(), "r-")
            plt.xlim(0, sz)
            # plt.ylim(0, round(np.max(X_train),-5))
            # plt.ylim(0,1)
            plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
                    transform=plt.gca().transAxes)
            if yi == 1:
                plt.title("kShape clustering")

        plt.tight_layout()
        plt.savefig(f"./plots/time_series_clustering/comparison_clustering_algorithms_{country_code}.png")
        # plt.show()

    return data

# time_series_clustering(country_code="FR", split_by_year = False, 
#     plot_knee = False, plot_cluster_euclidean_k_means = True,
#     plot_cluster_comparions = False)