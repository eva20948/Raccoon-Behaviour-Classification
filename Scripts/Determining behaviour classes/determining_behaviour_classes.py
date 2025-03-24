"""
Filename: determining_behaviour_classes.py
Author: Eva Reinhardt
Date: 2024-12-09
Version: 1.0
Description: The aim is to select behavior groups using different clustering algorithms. The
presentation uses PCA components, a table showing the assignment of behaviours to a cluster
according to the majority vote of the respective behavior and a second table in which it is broken
down and colour-coded how many data points per behaviour were assigned to which cluster (including
measurements kurtosis and enthropy).

Functions in this file:
apply_clustering_and_plot(): applying different clustering algorithms and prepare data to pass
        it to the plot functions

apply_weighted_clustering_and_plot(): applying different clustering algorithms using different
        weights for the predictors and prepare the data to pass it to the plot functions
"""

import re

import pandas as pd
import numpy as np

from tkinter import filedialog

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, MeanShift, Birch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.backends.backend_pdf import PdfPages

from raccoon_acc_setup import variables_simplefunctions as sim_func
from raccoon_acc_setup import importing_raw_data as im_raw
from raccoon_acc_setup import gui_functions as guif
from raccoon_acc_setup import plot_functions as plt_func

color_mapping = sim_func.COLOR_MAPPING_HTML

cluster_algorithms = {
    'KMeans': KMeans(random_state=42),
    # 'AgglomerativeClustering_single': AgglomerativeClustering(linkage='single'),
    'AgglomerativeClustering_ward': AgglomerativeClustering(linkage='ward'),
    # 'AgglomerativeClustering_complete': AgglomerativeClustering(linkage='complete'),
    # 'AgglomerativeClustering_average': AgglomerativeClustering(linkage='average'),
    # 'DBScan': DBSCAN()
    'SpectralClustering': SpectralClustering(affinity='nearest_neighbors', random_state=42),
    'GaussianMixture': GaussianMixture(n_components=5, random_state=42),
    # 'HDBSCAN': hdbscan.HDBSCAN(min_cluster_size=10),
    'MeanShift': MeanShift(),
    # 'AffinityPropagation': AffinityPropagation(random_state=42),
    'Birch': Birch()
}

filepaths_peter = [sim_func.IMPORT_PARAMETERS['Peter']['filepath_pred']]

filepaths_domi = [sim_func.IMPORT_PARAMETERS['Dominique']['filepath_pred']]

filepaths = [filepaths_peter, filepaths_domi]

def apply_clustering_and_plot(predictors_func: pd.DataFrame, labels_func: pd.DataFrame, cluster_algorithms_func: dict,
                              name_func: str):
    """
    Function to apply different clustering algorithms on the data. 

    @param predictors_func: Dataframe of all the predictor data.
    @param labels_func: the behavior data which is used to evaluate the clustering (one column of the input predictors data)
    @param cluster_algorithms_func: dict with names of cluster algorithms and respective function.
    @param name_func: whose dataset, 'Peter', 'Dominique' or 'Both datasets'
    """

    scaler = StandardScaler()
    scaled_predictors = pd.DataFrame(scaler.fit_transform(predictors_func), columns=predictors_func.columns)

    plt_func.pca_eval(scaled_predictors)

    # Reduce to 2 components for visualization
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(scaled_predictors)
    print('pca')

    num_clusters = range(1, 5)

    # iterating through the cluster algorithms
    for algo_name, algo in cluster_algorithms_func.items():
        print(algo_name)
        # iterating through the number of clusters if requested by algorithm 
        if 'n_clusters' in algo.get_params():
            for i_fu in num_clusters:
                print(i_fu)
                # clustering and plotting
                algo.set_params(n_clusters=i_fu + 2)
                clusters = algo.fit_predict(scaled_predictors)
                plt_func.plot_clustering_results(clusters, labels_func, scaled_predictors, pca_components, i_fu + 2,
                                                 algo_name, pdf, name_func=name_func)
        else:
            print('no n-clusters')
            # clustering and plotting if no number of clusters input needed 
            clusters = algo.fit_predict(scaled_predictors)
            plt_func.plot_clustering_results(clusters, labels_func, scaled_predictors, pca_components, len(clusters),
                                             algo_name, pdf, name_func=name_func)


def apply_weighted_clustering_and_plot(predictors_func: pd.DataFrame, labels_func: pd.DataFrame,
                                       weights_func: pd.DataFrame, name_func: str, option_gen_func: str):
    """
    Function to apply different clustering algorithms on the data. 

    @param predictors_func: Dataframe of all the predictor data.
    @param labels_func: the behavior data which is used to evaluate the clustering (one column of the input predictors data)
    @param weights_func: weights for different predictors.
    @param name_func: whose dataset, 'Peter', 'Dominique' or 'Both datasets'
    @param option_gen_func: denotes the chosen generalization
    """
    for w in range(weights_func.shape[0]):
        scaler = StandardScaler()
        scaled_predictors = pd.DataFrame(scaler.fit_transform(predictors_func), columns=predictors_func.columns)

        # adding weight
        scaled_predictors = scaled_predictors * list(weights_func.iloc[w])

        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(scaled_predictors.dropna(axis=1))
        print('pca')

        num_clusters = 5

        title_addition = '\nweights: '
        idx = weights_func.iloc[w][weights_func.iloc[w] != 1].index.tolist()
        for i_2 in idx:
            title_addition = title_addition + i_2 + ': ' + str(weights_func.iloc[w][i_2]) + '; '

        if option_gen_func == 'optimization':
            clusters = KMeans(n_clusters=num_clusters, random_state=42).fit_predict(scaled_predictors)
            plt_func.plot_clustering_results(clusters, labels_func, scaled_predictors, pca_components, num_clusters,
                                             'KMeans', pdf, add=title_addition, name_func=name_func)

            clusters = SpectralClustering(n_clusters=num_clusters, random_state=42,
                                          affinity='nearest_neighbors').fit_predict(scaled_predictors)
            plt_func.plot_clustering_results(clusters, labels_func, scaled_predictors, pca_components, num_clusters,
                                             'Spectral Clustering', pdf, add=title_addition, name_func=name_func)

            clusters = GaussianMixture(n_components=5, random_state=42).fit_predict(scaled_predictors)
            plt_func.plot_clustering_results(clusters, labels_func, scaled_predictors, pca_components, num_clusters,
                                             'Gaussian Mixture', pdf, add=title_addition, name_func=name_func)

        else:
            clusters = GaussianMixture(n_components=5, random_state=42).fit_predict(scaled_predictors)
            plt_func.plot_clustering_results_only_table(clusters, labels_func, scaled_predictors, pca_components,
                                                        num_clusters,
                                                        'Gaussian Mixture', pdf, add=title_addition,
                                                        name_func=name_func)


if __name__ == "__main__":


    option_gen = guif.choose_option(['optimized representation', 'optimization'])
    if option_gen == 'optimization':

        option5 = guif.choose_option(['Both', 'Single'])

        if option5 == 'Single':
            option3 = guif.choose_option(['Peter', 'Dominique'])
            if option3 == 'Peter':
                filepaths = [filepaths_peter]
            else:
                filepaths = [filepaths_domi]

        pred_com = pd.DataFrame()

        option = guif.choose_option(options=['different weights - set k', 'different k - not weighted'])
        option4 = guif.choose_option(['Remove outliers', 'Take the whole dataset'])
        option2 = guif.choose_option(options=['x and z as separate variables', 'xz combined and reduced features'])

        output_pdf_path = filedialog.asksaveasfilename(title="Save as")
        with PdfPages(output_pdf_path) as pdf:
            for filepaths_temp in filepaths:
                pred = pd.DataFrame()
                for i, filepath in enumerate(filepaths_temp):
                    pred_1 = pd.read_csv(filepath)
                    if 'Peter' in filepath:
                        pred_1 = im_raw.convert_beh(pred_1, 'Peter')
                        name = 'Peter'

                        replacement_dict = {}
                        words_to_remove = []
                    elif 'Dominique' in filepath:
                        name = 'Dominique'
                        if '5032' in filepath:
                            if 'behavior_Ottilie' in pred_1.columns:
                                pred_1 = pred_1.drop(['behavior_Ottilie'], axis=1)
                            pred_1 = pred_1.rename(columns={'behavior_Lisa': 'behavior'})
                        elif '5033' in filepath:
                            if 'behavior_Lisa' in pred_1.columns:
                                pred_1 = pred_1.drop(['behavior_Lisa'], axis=1)
                            pred_1 = pred_1.rename(columns={'behavior_Ottilie': 'behavior'})
                        pred_1 = im_raw.behavior_combi_domi(pred_1)

                    pred = pd.concat([pred, pred_1], ignore_index=True)

                # preparing the dataframe: fitlering out schütteln, filtering out outliers from Ndyn column if wanted
                pred = pred[~(pred['behavior'] == 'schütteln')]
                pred = pred[~((pred['behavior'] == 'schlafen') & (pred['Ndyn'] > 0.5))]

                if option4 == 'Remove outliers':
                    pred = sim_func.remove_outliers(pred)

                pred = pred.dropna(axis=1)

                pred = im_raw.convert_beh(pred, 'generalization')
                pred = im_raw.convert_beh(pred, 'translation')

                if option2 == 'xz combined and reduced features':
                    pred = sim_func.x_z_combination(pred)

                else:
                    pred = pred.drop(['XZmean', 'XZvar', 'XZmin', 'XZmax', 'XZmax - XZmin'], axis=1)

                filtered_columns = [col for col in pred.columns if 'behavior' not in col and 'datetime' not in col]

                predictors = pred[filtered_columns]
                labels_col = [col for col in pred.columns if 'behavior' in col]
                labels = pred[labels_col]

                if option == 'different k - not weighted':
                    apply_clustering_and_plot(predictors, labels, cluster_algorithms, name_func=name)
                else:
                    pattern = r'datetime|behavior'
                    filtered_columns = [col for col in pred.columns if not re.search(pattern, col)]
                    weights = pd.DataFrame(np.ones((7, len(filtered_columns)), dtype=int), columns=filtered_columns)
                    weights['Ndyn'][0] = 1.5
                    weights['Ndyn'][1] = 1.5
                    weights['Ndyn'][2] = 2
                    weights['Ndyn'][3] = 2
                    weights['Ndyn'][4] = 2
                    weights['Ndyn'][5] = 3.0
                    weights['Yvar'][1] = 1.5
                    weights['Yvar'][2] = 1.5
                    weights['Yvar'][3] = 2
                    weights['Yvar'][4] = 2
                    weights['Yvar'][5] = 2
                    weights['Odba'][0] = 1.5
                    weights['Odba'][1] = 2
                    weights['Odba'][2] = 1.5
                    weights['Odba'][3] = 2
                    weights['Odba'][4] = 2.5
                    weights['Odba'][5] = 3.0
                    # weights['Ymean'][1] = 1.5
                    # weights['Ymean'][2] = 1.5
                    # weights['Ymean'][3] = 2
                    # weights['Ymean'][4] = 2
                    # weights['Ymean'][5] = 2
                    # if 'XZmean' in weights.columns:
                    #     weights['XZmean'][0] = 1.5
                    #     weights['XZmean'][1] = 2
                    #     weights['XZmean'][2] = 1.5
                    #     weights['XZmean'][3] = 2
                    #     weights['XZmean'][4] = 2.5
                    #     weights['XZmean'][5] = 3.0
                    if 'XZvar' in weights.columns:
                        weights['XZvar'][1] = 1.5
                        weights['XZvar'][2] = 1
                        weights['XZvar'][3] = 1.5
                        weights['XZvar'][4] = 1
                        weights['XZvar'][5] = 1.5

                    weights = weights.drop([1, 3, 5], axis=0)
                    apply_weighted_clustering_and_plot(predictors, labels, weights, name_func=name,
                                                       option_gen_func=option_gen)

                pred_com = pd.concat([pred_com, pred], ignore_index=True)

            name = 'Both datasets'
            pred_com = pred_com.dropna(axis=1)

            filtered_columns = [col for col in pred_com.columns if 'behavior' not in col and 'datetime' not in col]

            labels_col = [col for col in pred_com.columns if 'behavior' in col]
            labels1 = pred_com[labels_col]

            predictors = pred_com[filtered_columns]
            labels_col = [col for col in pred_com.columns if 'generalization' in col]
            labels = pred_com[labels_col]

            if option == 'different k - not weighted':
                apply_clustering_and_plot(predictors, labels1, cluster_algorithms, name_func=name)
                apply_clustering_and_plot(predictors, labels, cluster_algorithms, name_func=name)
            else:
                pattern = r'datetime|behavior'
                filtered_columns = [col for col in pred_com.columns if not re.search(pattern, col)]
                weights = pd.DataFrame(np.ones((7, len(filtered_columns)), dtype=int), columns=filtered_columns)
                weights['Ndyn'][0] = 1.5
                weights['Ndyn'][1] = 1.5
                weights['Ndyn'][2] = 2
                weights['Ndyn'][3] = 2
                weights['Ndyn'][4] = 2
                weights['Ndyn'][5] = 3.0
                weights['Yvar'][1] = 1.5
                weights['Yvar'][2] = 1.5
                weights['Yvar'][3] = 2
                weights['Yvar'][4] = 2
                weights['Yvar'][5] = 2
                weights['Odba'][0] = 1.5
                weights['Odba'][1] = 2
                weights['Odba'][2] = 1.5
                weights['Odba'][3] = 2
                weights['Odba'][4] = 2.5
                weights['Odba'][5] = 3.0
                if 'XZvar' in weights.columns:
                    weights['XZvar'][1] = 1.5
                    weights['XZvar'][2] = 1
                    weights['XZvar'][3] = 1.5
                    weights['XZvar'][4] = 1
                    weights['XZvar'][5] = 1.5

                weights = weights.drop([1, 3, 5], axis=0)
                apply_weighted_clustering_and_plot(predictors, labels1, weights, name_func=name,
                                                   option_gen_func=option_gen)
                apply_weighted_clustering_and_plot(predictors, labels, weights, name_func=name,
                                                   option_gen_func=option_gen)

    else:

        pred_com = pd.DataFrame()
        output_pdf_path = filedialog.asksaveasfilename(title="Save as")
        with PdfPages(output_pdf_path) as pdf:
            for filepaths_temp in filepaths:
                pred = pd.DataFrame()
                for i, filepath in enumerate(filepaths_temp):
                    pred_1 = pd.read_csv(filepath)
                    if 'Peter' in filepath:
                        pred_1 = im_raw.convert_beh(pred_1, 'Peter')
                        name = 'Peter'

                        replacement_dict = {}
                        words_to_remove = []
                    elif 'Dominique' in filepath:
                        name = 'Dominique'
                        if '5032' in filepath:
                            if 'behavior_Ottilie' in pred_1.columns:
                                pred_1 = pred_1.drop(['behavior_Ottilie'], axis=1)
                            pred_1 = pred_1.rename(columns={'behavior_Lisa': 'behavior'})
                        elif '5033' in filepath:
                            if 'behavior_Lisa' in pred_1.columns:
                                pred_1 = pred_1.drop(['behavior_Lisa'], axis=1)
                            pred_1 = pred_1.rename(columns={'behavior_Ottilie': 'behavior'})
                        pred_1 = im_raw.behavior_combi_domi(pred_1)

                    pred = pd.concat([pred, pred_1], ignore_index=True)

                # preparing the dataframe: fitlering out schütteln, filtering out outliers from Ndyn column if wanted
                pred = pred[~(pred['behavior'] == 'schütteln')]
                pred = pred[~((pred['behavior'] == 'schlafen') & (pred['Ndyn'] > 0.5))]

                print(pred.head())

                pred = pred.dropna(axis=1)

                pred = im_raw.convert_beh(pred, 'generalization')
                pred = im_raw.convert_beh(pred, 'translation')

                pred = sim_func.x_z_combination(pred)

                pred_com = pd.concat([pred_com, pred], ignore_index=True)

            name = 'Both datasets'
            pred_com = pred_com.dropna(axis=1)

            filtered_columns = [col for col in pred_com.columns if 'behavior' not in col and 'datetime' not in col]

            labels_col = [col for col in pred_com.columns if 'behavior' in col]
            labels1 = pred_com[labels_col]

            predictors = pred_com[filtered_columns]
            labels_col = [col for col in pred_com.columns if 'generalization' in col]
            labels = pred_com[labels_col]

            pattern = r'datetime|behavior'
            filtered_columns = [col for col in pred_com.columns if not re.search(pattern, col)]
            weights = pd.DataFrame(np.ones((7, len(filtered_columns)), dtype=int), columns=filtered_columns)
            weights['Ndyn'][0] = 2.0
            weights['Yvar'][0] = 2.0
            weights['Odba'][0] = 2.5

            weights = weights.drop([1, 3, 5], axis=0)
            apply_weighted_clustering_and_plot(predictors, labels1, weights, name_func=name, option_gen_func=option_gen)
