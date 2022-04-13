import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from datetime import date

# TODO: Importiere Decision Tree aus Scikit-Learn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree 

# TODO: Importiere Random Forest aus Scikit-Learn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# TODO: Importiere Gradient Boosting aus Scikit-Learn
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier

# TODO: Importiere Evaluationsmetriken
from sklearn.metrics import accuracy_score, classification_report, make_scorer

# TODO: Visualisierungstools
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split

# TODO: Visualisierungstools
from sklearn.tree import plot_tree, export_text, export_graphviz
from torch import rand

import graphviz 

from visualization.plot_statistical_analysis import correlation_matrix
from features.data_processing import data_processing


def decision_tree(df_total:pd.DataFrame,  
    list_of_filters:list =['scheduled_exchanges_', 'residual_load_', 'net_position_', 'day_ahead_prices_'], 
    from_year:int = 2015, to_year:int = 2022):

    X_train, X_test, y_train, y_test, features_name_, class_names_ = data_processing(df_total, list_of_filters, from_year, to_year)

    clf = DecisionTreeClassifier(max_depth=3, criterion='gini',
        splitter='best', min_samples_leaf=1, min_samples_split=2,
        ccp_alpha=0, min_impurity_decrease=0)
    clf = clf.fit(X_train, y_train)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print(f"Score aus dem Trainingsset: {accuracy_train}")
    print(f"Score aus dem Testset: {accuracy_test}")

    print(classification_report(y_test, clf.predict(X_test)))

    #Evaluate the Model and Print Performance Metrics
    from sklearn import metrics
    print('Accuracy:', np.round(metrics.accuracy_score(y_test,y_pred_test),4))
    print('Precision:', np.round(metrics.precision_score(y_test,
    y_pred_test,average = 'weighted'),4))
    print('Recall:', np.round(metrics.recall_score(y_test,y_pred_test,
    average = 'weighted'),4))
    print('F1 Score:', np.round(metrics.f1_score(y_test,y_pred_test,
    average = 'weighted'),4))
    print('Cohen Kappa Score:', np.round(metrics.cohen_kappa_score(y_test,
    y_pred_test),4))
    print('Matthews Corrcoef:', np.round(metrics.matthews_corrcoef(y_test,
    y_pred_test),4))
    print('\t\tClassification Report:\n', metrics.classification_report(y_pred_test,
    y_test))

    max_depth = clf.get_depth()

    max_depth_grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(),
        scoring=make_scorer(accuracy_score),
        param_grid=ParameterGrid(
            {"max_depth": [[max_depth] for max_depth in range(1, max_depth+1)]}
        )
    )

    max_depth_grid_search.fit(X_train, y_train)
    print(max_depth_grid_search.best_params_)

    best_max_depth_tree = max_depth_grid_search.best_estimator_
    best_max_depth = best_max_depth_tree.get_depth()

    print(best_max_depth)

    # fig, ax = plt.subplots()
    # plot_tree(
    #     best_max_depth_tree,
    #     feature_names=features_name_,
    #     class_names=class_names_,
    #     filled=True,
    #     ax=ax
    # )
    # plt.savefig("./plots/decision_tree/decision_tree_best.png")
    # plt.close('all')

    clf = DecisionTreeClassifier(max_depth=3, criterion='entropy',
        splitter='best', min_samples_leaf=1, min_samples_split=2,
        ccp_alpha=0, min_impurity_decrease=0)
    clf = clf.fit(X_train, y_train)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    print("With best maximum depth:")
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print(f"Score aus dem Trainingsset: {accuracy_train}")
    print(f"Score aus dem Testset: {accuracy_test}")

    print(classification_report(y_test, clf.predict(X_test)))

    #Evaluate the Model and Print Performance Metrics
    from sklearn import metrics
    print('Accuracy:', np.round(metrics.accuracy_score(y_test,y_pred_test),4))
    print('Precision:', np.round(metrics.precision_score(y_test,
    y_pred_test,average = 'weighted'),4))
    print('Recall:', np.round(metrics.recall_score(y_test,y_pred_test,
    average = 'weighted'),4))
    print('F1 Score:', np.round(metrics.f1_score(y_test,y_pred_test,
    average = 'weighted'),4))
    print('Cohen Kappa Score:', np.round(metrics.cohen_kappa_score(y_test,
    y_pred_test),4))
    print('Matthews Corrcoef:', np.round(metrics.matthews_corrcoef(y_test,
    y_pred_test),4))
    print('\t\tClassification Report:\n', metrics.classification_report(y_pred_test,
    y_test))

    # dot_data = export_graphviz(clf, out_file=None,
    #                   feature_names=features_name_,
    #                   class_names=class_names_,
    #                   filled=True, rounded=True,  
    #                   special_characters=True)  
    # graph = graphviz.Source(dot_data)
    # graph.render(f"price_spread_depth_best_decision_tree_december_2021_entropy")
    
    # graph.render(directory="./plots/decision_tree", 
    #     filename=f"price_spread_best_depth_decision_tree_{from_year}_{to_year}.gv", 
    #     outfile='.png')

    # ## reduce with pruning

    # ccp_alphas = clf.cost_complexity_pruning_path(X_train, y_train)["ccp_alphas"]

    # ccp_alphas_grid_search = GridSearchCV(
    #     estimator=DecisionTreeClassifier(),
    #     scoring=make_scorer(accuracy_score),
    #     param_grid=ParameterGrid({"ccp_alpha": [[alpha] for alpha in ccp_alphas]})
    # )

    # ccp_alphas_grid_search.fit(X_train, y_train)

    # print(ccp_alphas_grid_search.best_params_)

    # best_ccp_alpha_tree = ccp_alphas_grid_search.best_estimator_

    # print(classification_report(y_test, best_ccp_alpha_tree.predict(X_test)))

    # dot_data = export_graphviz(best_ccp_alpha_tree, out_file=None,
    #                   feature_names=features_name_,
    #                   class_names=class_names_,
    #                   filled=True, rounded=True,  
    #                   special_characters=True)  
    # graph = graphviz.Source(dot_data)
    # graph.render(f"price_spread_best_ccp_alpha_tree_{from_year}_{to_year}")
    # # graph.render(directory="./plots/decision_tree", 
    #     # filename=f"price_spread_best_ccp_alpha_tree_{from_year}_{to_year}.gv", 
    #     # outfile='.png')

    # import time
    # import numpy as np

    # from sklearn.inspection import permutation_importance

    # start_time = time.time()
    # result = permutation_importance(
    #     decision_tree, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    # )
    # elapsed_time = time.time() - start_time
    # print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    # tree_importances = pd.Series(result.importances_mean, index=features_name_)

    # fig, ax = plt.subplots()
    # tree_importances.plot.bar(yerr=result.importances_std, ax=ax)
    # ax.set_title("Feature importances using permutation on full model")
    # ax.set_ylabel("Mean accuracy decrease")
    # fig.tight_layout()
    # plt.show()
    
    # from sklearn.metrics import confusion_matrix
    # print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred_test))
    # from sklearn.metrics import confusion_matrix
    # from io import BytesIO #neded for plot
    # import seaborn as sns; sns.set()
    # import matplotlib.pyplot as plt
    # mat = confusion_matrix(y_test, y_pred_test)
    # sns.heatmap(mat.T, square = True, annot = True, fmt = 'd', cbar = False)
    # plt.xlabel('true label')
    # plt.ylabel('predicted label');
    # plt.savefig("Confusion.jpg")
    # # Save SVG in a fake file object.
    # f = BytesIO()
    # plt.savefig(f, format = "svg")

    return

def random_forest(df_total:pd.DataFrame, 
    list_of_filters:list = ['scheduled_exchanges_', 'residual_load_', 'net_position_', 'day_ahead_prices_'],
    from_year:int = 2021, to_year:int = 2022):

    X_train, X_test, y_train, y_test, features_name_, class_names_ = data_processing(df_total, list_of_filters, from_year, to_year)

    n_estimator=10
    random_forest = RandomForestClassifier(n_estimators=n_estimator)
    random_forest.fit(X_train, y_train) 

    y_pred_train = random_forest.predict(X_train)
    y_pred_test = random_forest.predict(X_test)

    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print(f"Score aus dem Trainingsset: {accuracy_train}")
    print(f"Score aus dem Testset: {accuracy_test}")

    i_tree = 0
    for tree_in_forest in random_forest.estimators_:
        if (i_tree < n_estimator):        
            dot_data = export_graphviz(tree_in_forest, out_file=None,
                        feature_names=features_name_,
                        class_names=class_names_,
                        filled=True, rounded=True,  
                        special_characters=True)
            graph = graphviz.Source(dot_data)
            graph.render(filename=f"price_spread_random_forest_{i_tree}_{from_year}_{to_year}",
                directory="plots/random_forest")
            i_tree = i_tree + 1

    # fig, axes = plt.subplots(nrows = 1,ncols = n_estimator,figsize = (10,2), dpi=900)
    # for index in range(0, n_estimator):
    #     plot_tree(random_forest.estimators_[index],
    #                 feature_names = features_name_, 
    #                 class_names=class_names_,
    #                 filled = True,
    #                 ax = axes[index])

    #     axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
    # fig.savefig('./plots/random_forest/rf_5trees.png')

    import time
    import numpy as np

    start_time = time.time()
    importances = random_forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in random_forest.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_importances = pd.Series(importances, index=features_name_)

    fig, ax = plt.subplots(figsize=(10,10))
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig("./plots/random_forest/feature_importance_mdi.png", dpi=1200)
    # plt.show()

    from sklearn.inspection import permutation_importance

    start_time = time.time()
    result = permutation_importance(
        random_forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_importances = pd.Series(result.importances_mean, index=features_name_)

    fig, ax = plt.subplots(figsize=(10,10))
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.savefig("./plots/random_forest/feature_importance_full_model.png", dpi=1200)
    # plt.show()

    return

# import bidding_zones
# countries = list(bidding_zones.BIDDING_ZONES_CWE.keys())
# from data.create_dataframe import *
# df_day_ahead_prices = create_dataframe_day_ahead_prices(countries_day_ahead_prices=countries)
# df_scheduled_exchanges = create_dataframe_scheduled_exchanges(countries_scheduled_exchanges=countries)
# df_residual_load = create_dataframe_residual_load(countries_load=countries)
# df_net_positions = create_dataframe_net_positons(countries_net_positons=countries)

# df_total = pd.DataFrame()
# df_total = df_total.join([df_day_ahead_prices, df_scheduled_exchanges, df_residual_load, df_net_positions], how='outer')
# df_total.index = pd.to_datetime(df_total.index)
# df_total = df_total.resample('1h').sum()
# df_total.to_excel(f"./data/dataframes/df_total_1h_CWE.xlsx")

# list_of_filters = ['scheduled_exchanges_', 'residual_load_', 'net_position_', 'day_ahead_prices_']
# df_total = pd.read_excel(f"./data/dataframes/df_total_1h_CWE.xlsx", index_col=0, parse_dates=True)
# liste = []
# for i in range(len(list_of_filters)):
#     liste.append(list_of_filters[i])
#     print(liste)
#     decision_tree(df_total, list_of_filters=liste, from_year=2019, to_year=2022)
#     random_forest(df_total, list_of_filters)

# decision_tree(df_total, list_of_filters=list_of_filters)
# random_forest(df_total=df_total, list_of_filters=list_of_filters)