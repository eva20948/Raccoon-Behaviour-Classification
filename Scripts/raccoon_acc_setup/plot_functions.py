"""
Filename: plot_functions.py
Author: Eva Reinhardt
Date: 2024-12-09
Version: 1.0
Description: This file contains functions to plot different tables,
diagrams and images.

The contained functions are:
plot_burst(): plotting the raw data

plot_clustering_results: Function to plot the clustering's results, including:
        a plot of the clusters using PCA components,
        a table enumerating the clustering per behavior,
        the silhouette score per k

        used predominantly in behavior_selection.py

pca_eval(): Function to evaluate the principal components. Output is a table
        representing the composition of the Principal Components

confusion_matrices_layered(): preparing the data for the layered model.
        conducting parameter optimization and cross validation for both layers.
        plotting the respective confusion matrices for the layers.

visualise_predictions_ml(): visualizing the predictions of a machine learning algorithm.
        Which behaviors are mislabelled, including confusion matrix and histogram for
        prediction probability.

plotting_proportions_predictions(): plotting the proportions of predictions probabilities for different predictions
        (most useful for unknown behavior)

output_hourly_contingents(): hourly behavior contingents are plotted as bar plots, included is the depiction of sunlight

plotting_vps(): plotting violinplots of different predictors sorted by the recorded behavior.

sankey_diagram(): plots sankey diagram
"""


from datetime import datetime
import calendar
import locale

import pandas as pd
import numpy as np

import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages

import plotly.graph_objects as go

import seaborn as sns

from scipy import stats
from scipy.stats import chisquare, entropy

from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from astral import LocationInfo
from astral.sun import sun

from . import variables_simplefunctions as sim_func
from . import machine_learning_functions as mlf




def plot_burst(burst: pd.DataFrame):
    """
    Plotting a burst (from acceleration data, same datetime). Plot for X, Y, Z.

    @param burst: DataFrame containing columns 'X', 'Y', 'Z', and behavior columns.
                    Logically, containing all rows belonging to one datetime.
    """

    # initiate plot

    fig, ax = plt.subplots()

    # specifying the used data
    x = np.array(burst['X'].astype(float))
    y = np.array(burst['Y'].astype(float))
    z = np.array(burst['Z'].astype(float))

    # plotting the different lines for x, y, and  z
    ax.plot(range(len(burst)), x, 'o-', linewidth=2)
    ax.plot(range(len(burst)), y, 'o-', linewidth=2.0)
    ax.plot(range(len(burst)), z, 'o-', linewidth=2)
    ax.axis([0, len(burst), -1, 1])

    # creating the title, containing the behavior per individual
    names = burst.filter(regex='behavior_')
    names = names.columns
    title = "; ".join([f"{name}: {burst[name].unique()[0]}" for name in names])

    ax.legend()
    ax.set_title(title)

    plt.show()


def plot_clustering_results(clusters: np.ndarray, labels_func: pd.DataFrame, scaled_predictors: pd.DataFrame,
                            pca_components: np.ndarray, k: int, cluster_al: str, pdf, add: str = '',
                            name_func: str = ''):
    """
    Function to plot the clustering's results, including:
        a plot of the clusters using PCA components,
        a table enumerating the clustering per behavior,
        the silhouette score per k
    If a pdf is opened before using the function, the plots are saved as a page to
    the pdf.

    @param clusters: the prediction of a clustering algorithm
    @param labels_func: the behavior data which is used to evaluate the clustering (one column of the input predictors data)
    @param scaled_predictors: the scaled predictors data (output from scaler)
    @param pca_components: PCA components used to plot the clusters (output from PCA analysis)
    @param k: number of clusters
    @param cluster_al: the used cluster algorithm
    @param pdf: pdf instance where the plots should be saved to
    @param add: additional information that will appear in the figure's title
    @param name_func: 'Peter' or 'Dominique'

    """

    # Add the cluster and behavior labels to the PCA components for visualization
    clustered_data = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])
    clustered_data['cluster'] = clusters
    if labels_func.shape[1] == 2:
        clustered_data['behavior'] = labels_func['behavior']
        clustered_data['behavior_generalization'] = labels_func['behavior_generalization']
    else:
        clustered_data['behavior'] = labels_func

    # calculating the silhouette score
    sc = silhouette_score(scaled_predictors, clusters)

    most_frequent = clustered_data.groupby('behavior')['cluster'].agg(lambda x: x.mode().iloc[0]).reset_index()
    most_frequent.columns = ['behavior', 'cluster']

    # Group by cluster and aggregate behaviors into a list
    grouped = most_frequent.groupby('cluster')['behavior'].agg(lambda x: '; '.join(x.unique())).reset_index()

    for j in range(grouped.shape[0]):
        parts = grouped['behavior'].iloc[j].split(';')
        print(parts)

        # Rejoin the parts, inserting a newline after every second comma
        new_string = ""
        for i_f in range(len(parts)):
            new_string += parts[i_f]
            if (i_f + 1) % 2 == 0 and i_f != len(parts) - 1:
                new_string += ';\n'
            elif i_f != len(parts) - 1:
                new_string += ';'
        grouped['behavior'].iloc[j] = new_string

    # Rename the columns for clarity
    grouped.columns = ['Cluster', 'Behaviors']

    # Plot the clusters
    fig = plt.figure(figsize=(25, 15))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 3], width_ratios=[2, 3])

    fig.suptitle(name_func + ', Used algorithm: ' + cluster_al + '\nUsed k: ' + str(k) + add, fontsize=18)

    axs0 = plt.subplot(gs[0, 0])
    for cluster in clustered_data['cluster'].unique():
        subset = clustered_data[clustered_data['cluster'] == cluster]
        axs0.scatter(subset['PCA1'], subset['PCA2'], label=f'Cluster {cluster}')

    axs0.legend()
    axs0.set_title('Clusters plotted using PCA components' + '\n Silhouette Score:  ' + str(sc))
    axs0.set_xlabel('PCA1')
    axs0.set_ylabel('PCA2')

    axs01 = plt.subplot(gs[0, 1])
    axs01.axis('tight')
    axs01.axis('off')

    table = axs01.table(cellText=grouped.values,
                        colLabels=grouped.columns,
                        rowLabels=None,
                        cellLoc='center', loc='center')

    # Determine the max number of lines in each row
    max_lines_per_row = []
    for i_f in range(1, len(grouped) + 1):
        max_lines = 1
        for j in range(len(grouped.columns)):
            text = table[(i_f, j)].get_text().get_text()
            num_lines = text.count(';') // 2 + 1
            max_lines = max(max_lines, num_lines)
        max_lines_per_row.append(max_lines)

    # Adjust the row heights uniformly based on the max lines in each row
    for i_f in range(1, len(grouped) + 1):
        for j in range(len(grouped.columns)):
            base_height = 0.07
            padding = 0.005
            adjusted_height = base_height + (max_lines_per_row[i_f - 1] - 1) * 0.05 + padding
            table[(i_f, j)].set_height(adjusted_height)

    # Customizing the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.2)

    # Analyze the clusters
    # Plotting the table
    axs1 = plt.subplot(gs[1, :])
    axs1.axis('tight')
    axs1.axis('off')
    table_data = clustered_data.groupby(['cluster', 'behavior']).size().unstack(fill_value=0)

    # Normalize the data by dividing by the sum of its column
    table_data_norm = table_data.div(table_data.sum(axis=0), axis=1)

    # Define a colormap from green to yellow to red
    cmap = LinearSegmentedColormap.from_list('custom_cmap', [(0.0, 'white'), (1.0, 'green')])

    column_en = []
    column_kurt = []
    column_chi = []
    column_g = []

    for i_f in range(len(table_data.columns)):

        c = table_data_norm.iloc[:, i_f]
        p = entropy(c, base=2)
        column_en.append((p / np.log2(len(c))).round(2))

        c_tot = table_data.iloc[:, i_f]
        stat, p = chisquare(c_tot)
        column_chi.append(p.round(2))

        expected_counts = [1 / sum(c_tot)] * len(c_tot)
        g_stat = 2 * np.sum(c_tot * np.log(c_tot / expected_counts))
        p = 1 - stats.chi2.cdf(g_stat, df=len(c_tot) - 1)
        column_g.append(p.round(2))

        c = c.sort_values()
        c = pd.concat([c, c[::-1]], ignore_index=True)
        p = stats.kurtosis(c)
        column_kurt.append((np.abs(p - 0.25) / 3).round(2))

    cmap_ks = LinearSegmentedColormap.from_list('custom_cmap',
                                                [(0.0, 'green'), (0.1, 'limegreen'), (0.55, 'yellow'), (1, 'red')])

    table_data_norm.loc[-2] = column_en
    table_data_norm.loc[-1] = column_kurt
    table_data_norm.index = table_data_norm.index + 2
    table_data_norm = table_data_norm.sort_index()

    # Insert column_ks as the first row in the table
    table_data.loc[-2] = column_en
    table_data.loc[-1] = column_kurt
    table_data.index = table_data.index + 2
    table_data = table_data.sort_index()

    for i_f in range(4, len(table_data)):
        table_data.iloc[i_f] = table_data.iloc[i_f].astype(int)

    # Reorder columns based on generalization
    if labels_func.shape[1] == 2:
        generalization_order = labels_func[['behavior', 'behavior_generalization']].drop_duplicates().sort_values \
            ('behavior_generalization')
        ordered_columns = generalization_order['behavior'].tolist()

        # Reorder the DataFrame based on the new column order
        table_data = table_data[ordered_columns]
        table_data_norm = table_data_norm[ordered_columns]

    table = axs1.table(cellText=table_data.values,
                       colLabels=table_data.columns,
                       rowLabels=['ent', 'kurt'] + list((range(table_data.shape[0] - 2))),
                       cellLoc='center', loc='center')

    # Customizing the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.2)

    # here, I am rotating the column labels and set the height so that the text is readable
    max_col_label_length = max(len(str(label)) for label in table_data.columns)

    # Calculate the height scale based on the maximum column label length
    header_height_scale = max(1.7, max_col_label_length / 2)

    # Rotate the column labels
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            if (name_func == 'Both datasets') & (labels_func.shape[1] == 1):
                original_text = cell.get_text().get_text()
                new_text = original_text.replace(',', '\n')
                cell.get_text().set_text(new_text)
                color = sim_func.COLOR_MAPPING_HTML.get(original_text)
                header_height_scale = 10
            else:
                group = labels_func[labels_func['behavior'] == cell.get_text().get_text()][
                    'behavior_generalization'].unique()
                color = dict(sim_func.COLOR_MAPPING_HTML).get(group[0])
            cell.set_facecolor(color)
            cell.get_text().set_rotation(90)
            cell.get_text().set_ha('center')
            cell.set_height(cell.get_height() * header_height_scale)

    # Apply color to the column_ks row based on the normalized values
    for j in range(len(column_en)):
        value = table_data_norm.iloc[0, j]
        color = cmap_ks(value)
        print(value)
        table[(1, j)].set_facecolor(color)
        value = table_data_norm.iloc[1, j]
        print(value)
        color = cmap_ks(value)
        table[(2, j)].set_facecolor(color)

    # Apply color to the rest of the cells based on the normalized values
    for i_f in range(2, len(table_data_norm.index)):
        for j in range(len(table_data_norm.columns)):
            value = table_data_norm.iloc[i_f, j]
            color = cmap(value)
            table[(i_f + 1, j)].set_facecolor(color)

    # Create the custom legend
    patches = [mpatches.Patch(color=color, label=label) for label, color in sim_func.COLOR_MAPPING_HTML.items()]
    plt.legend(handles=patches, loc='lower left', bbox_to_anchor=(1, 1))

    axs1.set_title \
        ('Table representing the different clusters and behaviors: \nks: Kolmogorov-Smirnov-Test; kurt: Kurtosis')

    plt.tight_layout()

    pdf.savefig(fig)
    plt.close(fig)


def plot_clustering_results_only_table(clusters: np.ndarray, labels_func: pd.DataFrame, scaled_predictors: pd.DataFrame,
                                       pca_components: np.ndarray, k: int, cluster_al: str, pdf, add: str = '',
                                       name_func: str = ''):
    """
    Function to plot clustering results, focusing on a transposed table
    that represents the assignment of behaviors to clusters. The table cells
    are colored by behavior and spacious for better readability.

    @param clusters: the prediction of a clustering algorithm
    @param labels_func: the behavior data used to evaluate the clustering
    @param scaled_predictors: the scaled predictors data (output from scaler)
    @param pca_components: PCA components (not used here but kept for compatibility)
    @param k: number of clusters
    @param cluster_al: the used cluster algorithm
    @param pdf: pdf instance where the plot should be saved to
    @param add: additional information that will appear in the figure's title
    @param name_func: 'Peter' or 'Dominique'
    """

    clustered_data = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])
    clustered_data['cluster'] = clusters
    if labels_func.shape[1] == 2:
        clustered_data['behavior'] = labels_func['behavior']
        clustered_data['behavior_generalization'] = labels_func['behavior_generalization']
    else:
        clustered_data['behavior'] = labels_func

    table_data = clustered_data.groupby(['cluster', 'behavior']).size().unstack(fill_value=0)

    # Normalize the data by dividing by the sum of its column
    table_data_norm = table_data.div(table_data.sum(axis=0), axis=1)

    # Define a colormap from green to yellow to red
    cmap = LinearSegmentedColormap.from_list('custom_cmap', [(0.0, 'white'), (1.0, 'green')])

    column_en = []
    column_kurt = []
    column_chi = []
    column_g = []

    for i_f in range(len(table_data.columns)):

        c = table_data_norm.iloc[:, i_f]
        p = entropy(c, base=2)
        column_en.append((p / np.log2(len(c))).round(2))

        c_tot = table_data.iloc[:, i_f]
        stat, p = chisquare(c_tot)
        column_chi.append(p.round(2))

        expected_counts = [1 / sum(c_tot)] * len(c_tot)
        g_stat = 2 * np.sum(c_tot * np.log(c_tot / expected_counts))
        p = 1 - stats.chi2.cdf(g_stat, df=len(c_tot) - 1)
        column_g.append(p.round(2))

        c = c.sort_values()
        c = pd.concat([c, c[::-1]], ignore_index=True)
        p = stats.kurtosis(c)
        column_kurt.append((np.abs(p - 0.25) / 3).round(2))

    cmap_ks = LinearSegmentedColormap.from_list('custom_cmap',
                                                [(0.0, 'green'), (0.1, 'limegreen'), (0.55, 'yellow'), (1, 'red')])

    table_data_norm.loc[-2] = column_en
    table_data_norm.loc[-1] = column_kurt
    table_data_norm.index = table_data_norm.index + 2
    table_data_norm = table_data_norm.sort_index()

    # Insert column_ks as the first row in the table
    table_data.loc[-2] = column_en
    table_data.loc[-1] = column_kurt
    table_data.index = table_data.index + 2
    table_data = table_data.sort_index()

    for i_f in range(4, len(table_data)):
        table_data.iloc[i_f] = table_data.iloc[i_f].astype(int)

    # Reorder columns based on generalization
    if labels_func.shape[1] == 2:
        generalization_order = labels_func[['behavior', 'behavior_generalization']].drop_duplicates().sort_values \
            ('behavior_generalization')
        ordered_columns = generalization_order['behavior'].tolist()

        # Reorder the DataFrame based on the new column order
        table_data = table_data[ordered_columns]
        table_data_norm = table_data_norm[ordered_columns]

    table_data = table_data.transpose()
    table_data_norm = table_data_norm.transpose()

    # Create a vertical layout for the plot
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_subplot(111)
    ax.axis('tight')
    ax.axis('off')

    # Adjust cell height and width for better spacing
    table = ax.table(
        cellText=table_data.values,
        colLabels=['ent', 'kurt'] + list((range(table_data.shape[1] - 2))),
        rowLabels=table_data.index,
        cellLoc='center',
        loc='center'
    )

    # Make cells more spacious
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    # Apply colors to only the first column (row labels)
    for row, cell in table.get_celld().items():
        if row[1] == -1:
            group = labels_func[labels_func['behavior'] == cell.get_text().get_text()][
                'behavior_generalization'].unique()
            color = dict(sim_func.COLOR_MAPPING_HTML).get(group[0])
            cell.set_facecolor(color)
        else:
            cell.set_facecolor('white')

    cell_width = 0.07
    for (row, col), cell in table.get_celld().items():
        cell.set_width(cell_width)

    # Apply color to the column_ks row based on the normalized values
    for j in range(len(column_en)):
        value = table_data_norm.iloc[j, 0]
        color = cmap_ks(value)
        print(value)
        table[(j + 1, 0)].set_facecolor(color)
        value = table_data_norm.iloc[j, 1]
        print(value)
        color = cmap_ks(value)
        table[(j + 1, 1)].set_facecolor(color)

    # Apply color to the rest of the cells based on the normalized values
    for i_f in range(2, len(table_data_norm.columns)):
        for j in range(len(table_data_norm.index)):
            value = table_data_norm.iloc[j, i_f]
            color = cmap(value)
            table[(j + 1, i_f)].set_facecolor(color)

    # Create the custom legend
    patches = [mpatches.Patch(color=color, label=label) for label, color in sim_func.COLOR_MAPPING_HTML.items() if
               label not in ['intermediate energy', 'unknown']]
    plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(1, 0.5))

    # Save the plot to the provided PDF
    pdf.savefig(fig)
    plt.close(fig)


def pca_eval(scaled_predictors: pd.DataFrame, pdf=None):
    """
    Function to evaluate the principal components.

    @param scaled_predictors: Array of normalized predictor data.
    """
    # pca evaluation
    scaled_predictors = scaled_predictors.dropna(axis=1)
    n = range(2, 5)
    for i_f in n:
        pca = PCA(n_components=i_f)
        pca.fit(scaled_predictors)

        components = pca.components_

        components_df = pd.DataFrame(components, columns=scaled_predictors.columns)

        fig = plt.figure(figsize=(16, 8))
        fig.suptitle('PCA analysis for ' + str(i_f) + ' Principal components', fontsize=18)
        gs = gridspec.GridSpec(1, 2, width_ratios=[0.5, 0.5])

        axs0 = plt.subplot(gs[0])

        for idx, component in enumerate(components_df.values):
            axs0.bar(range(len(component)), component, label=f'PC{idx + 1}')

        axs0.legend(loc='best')
        axs0.set_title('PCA components')
        axs0.set_xlabel('Principle components')
        axs0.set_ylabel('Feature Contributions')
        axs0.set_xticks(range(len(scaled_predictors.columns)))
        axs0.set_xticklabels(scaled_predictors.columns, rotation=90)

        axs1 = plt.subplot(gs[1])
        axs1.axis('tight')
        axs1.axis('off')
        table_data = components_df.T.round(3)

        cmap = LinearSegmentedColormap.from_list('white_green', [(0, 'white'), (1, 'green')])

        table = axs1.table(cellText=table_data.values,
                           colLabels=table_data.columns,
                           rowLabels=table_data.index,
                           cellLoc='center', loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.5, 1.2)

        for i_f in range(len(table_data.index)):
            for j in range(len(table_data.columns)):
                value = abs(table_data.iloc[i_f, j])
                color = cmap(value)
                table[(i_f + 1, j)].set_facecolor(color)

        axs1.set_title('Table representing the contribution of each feature to the principal components.')

        plt.tight_layout()

        if fig is not None:
            if pdf:
                pdf.savefig()
                plt.close()
            else:
                plt.show()


def confusion_matrices_layered(predictors: pd.DataFrame, pred_int: pd.DataFrame, labels_int: pd.DataFrame,
                               labels_3: pd.DataFrame,
                               ind_90: list, ind_10: list, int_ind_90: list, int_ind_10: list,
                               ml_algs: dict, cv, mapping_red_rev):
    """
    function to plot the confusion matrices for the layered model.
    @param predictors: dataframe of all predictors, split from labels
    @param pred_int: dataframe of all predictors used in the second layer ('climbing', 'walking', 'exploring')
    @param labels_int: dataframe of all labels used in second layer
    @param labels_3: dataframe of all labels used in first layer (interchange 'climbing', 'walking', 'exploring' with 'intermediate energy')
    @param ind_90: indeces for the 90% training dataset
    @param ind_10: indeces for the 10% external testing dataset
    @param int_ind_90: indeces for the 90% training dataset of the second layer
    @param int_ind_10: indeces for the 10% external testing dataset of the second layer
    @param ml_algs: dictionary of machine learning algorithms
    @param cv: Cross validation isntance
    @param mapping_red_rev: mapping from int to str for the first layer
    @return: pdf document with the confusion matrices
    """

    for algo_name, algo in ml_algs.items():
        with PdfPages('confusion_matrices' + algo_name + 'redfeatures_nosampling.pdf') as pdf:
            pred_int_90 = pred_int.copy()
            pred_int_90 = pred_int_90.iloc[int_ind_90]
            pred_int_10 = pred_int.copy()
            pred_int_10 = pred_int_10.iloc[int_ind_10]
            labels_int_90 = labels_int.copy()
            labels_int_90 = labels_int_90.iloc[int_ind_90]
            labels_int_10 = labels_int.copy()
            labels_int_10 = labels_int_10.iloc[int_ind_10]
            labels_3_90 = labels_3.copy()
            labels_3_90 = labels_3_90.iloc[ind_90]
            labels_3_10 = labels_3.copy()
            labels_3_10 = labels_3_10.iloc[ind_10]

            [y_pred_3_90, y_prob_3_90], [y_pred_3_10, y_prob_3_10], param_1 = mlf.ml_90_10(algo,
                                                                                           predictors.iloc[ind_90],
                                                                                           predictors.iloc[ind_10],
                                                                                           labels_3_90[
                                                                                               'behavior_generalization'],
                                                                                           algo_name, cv)

            [y_pred_int_90, y_prob_int_90], [y_pred_int_10, y_prob_int_10], param_2 = mlf.ml_90_10(algo, pred_int_90,
                                                                                                   pred_int_10,
                                                                                                   labels_int_90[
                                                                                                       'behavior_generalization'],
                                                                                                   algo_name, cv)

            y_pred_int_90 = pd.DataFrame(y_pred_int_90.ravel() + 1, columns=['prediction'])
            y_pred_int_90['prediction'] = y_pred_int_90['prediction'].map(sim_func.inverted_mapping)

            labels_int_90['behavior_generalization'] += 1
            labels_int_90['behavior_generalization'] = labels_int_90['behavior_generalization'].map(
                sim_func.inverted_mapping)

            y_pred_int_10 = pd.DataFrame(y_pred_int_10.ravel() + 1, columns=['prediction'])
            y_pred_int_10['prediction'] = y_pred_int_10['prediction'].map(sim_func.inverted_mapping)

            labels_int_10['behavior_generalization'] += 1
            labels_int_10['behavior_generalization'] = labels_int_10['behavior_generalization'].map(
                sim_func.inverted_mapping)

            y_pred_3_90 = pd.DataFrame(y_pred_3_90.ravel(), columns=['prediction'])
            y_pred_3_90['prediction'] = y_pred_3_90['prediction'].map(mapping_red_rev)

            y_pred_3_10 = pd.DataFrame(y_pred_3_10.ravel(), columns=['prediction'])
            y_pred_3_10['prediction'] = y_pred_3_10['prediction'].map(mapping_red_rev)

            labels_3_90['behavior_generalization'] = labels_3_90['behavior_generalization'].map(mapping_red_rev)
            labels_3_10['behavior_generalization'] = labels_3_10['behavior_generalization'].map(mapping_red_rev)

            classes_red = ['resting', 'intermediate energy', 'high energy']

            classes_int = [sim_func.inverted_mapping[1], sim_func.inverted_mapping[2], sim_func.inverted_mapping[3]]

            cm1 = confusion_matrix(labels_3_90['behavior_generalization'], y_pred_3_90['prediction'],
                                   labels=classes_red)
            cm1 = normalize(cm1, axis=1, norm='l1')
            labels_cm1 = classes_red
            title_cm1 = 'First layer of model - 90% 10-fold CV'

            cm2 = confusion_matrix(labels_3_10['behavior_generalization'], y_pred_3_10['prediction'],
                                   labels=classes_red)
            cm2 = normalize(cm2, axis=1, norm='l1')
            labels_cm2 = classes_red
            title_cm2 = 'First layer of model - 10% external test'

            cm3 = confusion_matrix(labels_int_90['behavior_generalization'], y_pred_int_90['prediction'],
                                   labels=classes_int)
            cm3 = normalize(cm3, axis=1, norm='l1')
            labels_cm3 = classes_int

            cm4 = confusion_matrix(labels_int_10['behavior_generalization'], y_pred_int_10['prediction'],
                                   labels=classes_int)
            cm4 = normalize(cm4, axis=1, norm='l1')
            labels_cm4 = classes_int

            title_cm3 = 'Second layer of model - 90% 10-fold CV'
            title_cm4 = 'Second layer of model - 10% external test'

            confusion_matrices = [cm1, cm2, cm3, cm4]
            labels_cms = [labels_cm1, labels_cm2, labels_cm3, labels_cm4]
            title_cms = [title_cm1, title_cm2, title_cm3, title_cm4]

            fig = plt.figure(figsize=(30, 12))
            fig.suptitle(algo_name + '\nSeparate confusion matrices\n parameters for the left: ' + ', '.join(
                f'{key}: {value}' for key, value in param_1.items()) + '\nparameters for the right: ' +
                         ', '.join(f'{key}: {value}' for key, value in param_2.items()))
            gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 0.1])

            cmap = plt.cm.Blues

            for i in range(len(confusion_matrices)):
                cm = confusion_matrices[i]
                ax = fig.add_subplot(gs[0, i])
                sns.heatmap(cm, annot=True, fmt='.2f', cmap=cmap, cbar=False, ax=ax, vmin=0, vmax=1, square=True)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title(title_cms[i])
                ax.set_xticklabels(labels_cms[i])
                ax.set_yticklabels(labels_cms[i])

            ax5 = fig.add_subplot(gs[0, 4])
            # Add a single color bar on the last subplot (right side)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=ax5)  # Use the last axis for the color bar
            cbar.set_label('Counts')

            ax11 = fig.add_subplot(gs[1, 0])
            ax11.hist(y_prob_3_90, bins=30)
            ax11.set_title('Histogram of prediction probabilities')

            ax21 = fig.add_subplot(gs[1, 1])
            ax21.hist(y_prob_3_10, bins=30)
            ax21.set_title('Histogram of prediction probabilities')

            ax31 = fig.add_subplot(gs[1, 2])
            ax31.hist(y_prob_int_90, bins=30)
            ax31.set_title('Histogram of prediction probabilities')

            ax41 = fig.add_subplot(gs[1, 3])
            ax41.hist(y_prob_int_10, bins=30)
            ax41.set_title('Histogram of prediction probabilities')

            plt.subplots_adjust(wspace=0.5, hspace=0.3)

            pdf.savefig(fig)
            plt.close(fig)


def visualise_predictions_ml(pred: pd.DataFrame, predictors_func: pd.DataFrame, labels_all: pd.DataFrame,
                             behaviors_func: pd.Series, gen: str, algorithm_name: str, algorithm,
                             opt: str, ext_test: list, pdf=None, sampling_func=None):
    """
    Function to visualize the different predictions of the machine learning algorithms.

    @type algorithm: list of algorithm estimator and parameters
    @param pred: dataframe containing all predictors
    @param predictors_func: the whole dataset of features
    @param labels_all: vector of behavior_generalization and behavior
    @param behaviors_func: all behaviors
    @param gen: generalization1 or generalization2
    @param algorithm_name: used algorithm (name)
    @param algorithm: used algorithm (function)
    @param sampling_func: list of sampling name and sampling method
    @param ext_test: list of training indeces and test indeces for external validation
    @param opt: including parameter optimization or not
    @param pdf: pdfPages instance if it should be used
    """

    class_order = list(sim_func.MAPPING.keys())
    if sampling_func is None:
        sampling_func = []
    classes = list(sim_func.MAPPING.keys())

    if 'SupportVectorMachine' in algorithm_name:
        columns = predictors_func.columns
        scaler = StandardScaler()
        predictors_func = pd.DataFrame(scaler.fit_transform(predictors_func), columns=columns)
    labels_all = labels_all['behavior_generalization'].to_numpy().ravel()

    if sampling_func:
        predictors_90 = predictors_func[predictors_func.index.isin(ext_test[0])]
        labels_90 = pd.DataFrame(labels_all[ext_test[0]], index=ext_test[0], columns=['behavior_generalization'])
        predictors_new, labels_new = sampling_func[1].fit_resample(predictors_90, labels_90)
    else:
        predictors_new = predictors_func[predictors_func.index.isin(ext_test[0])]
        labels_new = pd.DataFrame(labels_all[ext_test[0]], index=ext_test[0], columns=['behavior_generalization'])

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    if opt == 'Including parameter optimization':
        param = dict(mlf.parameter_optimization(algorithm, predictors_new,
                                                labels_new,
                                                algorithm_name))
        y_prob_all = cross_val_predict(algorithm[0].set_params(
            **param), predictors_new,
            labels_new['behavior_generalization'], method='predict_proba', cv=cv)
        alg = algorithm[0].set_params(**param)
    else:
        y_prob_all = cross_val_predict(
            algorithm, predictors_new,
            labels_new['behavior_generalization'], method='predict_proba', cv=cv)
        param = algorithm
        alg = algorithm

    alg.fit(predictors_new, labels_new)
    y_prob_9010 = alg.predict_proba(predictors_func[predictors_func.index.isin(ext_test[1])])
    labels_9010 = pd.DataFrame(labels_all[ext_test[1]], index=ext_test[1], columns=['behavior_generalization'])

    list_prob = [[y_prob_all, labels_new],
                 [y_prob_9010, labels_9010]]

    for [y_prob_all, labels_func] in list_prob:
        labels_func['sample_index'] = labels_func.index
        y_pred = np.zeros((len(y_prob_all), 1))
        y_prob = np.zeros(y_pred.shape)
        for i in range(len(y_prob)):
            y_prob[i] = np.amax(y_prob_all[i])
            y_pred[i] = np.argmax(y_prob_all[i])

        unknown_count, labels_prob, y_pred_prob = mlf.calculating_unknown_stats(y_pred, y_prob, labels_func)

        labels_func['behavior_generalization'] = labels_func['behavior_generalization'].map(sim_func.inverted_mapping)
        vectorized_mapping = np.vectorize(sim_func.inverted_mapping.get)
        y_pred = vectorized_mapping(y_pred)
        labels_func['predictions'] = y_pred

        labels_func['behavior_generalization'] = pd.Categorical(labels_func['behavior_generalization'],
                                                                categories=class_order, ordered=True)
        labels_func = labels_func.sort_values('behavior_generalization')

        labels_prob['behavior_generalization'] = labels_prob['behavior_generalization'].map(sim_func.inverted_mapping)
        y_pred_prob = vectorized_mapping(y_pred_prob)
        labels_prob['predictions'] = y_pred_prob

        labels_prob['behavior_generalization'] = pd.Categorical(labels_prob['behavior_generalization'],
                                                                categories=class_order, ordered=True)
        labels_prob = labels_prob.sort_values('behavior_generalization')

        print(y_pred)

        accuracy, recall, precision, f1, scores = mlf.calc_scores(labels_func['predictions'], labels_func)
        scores = classification_report(labels_func['behavior_generalization'], labels_func['predictions'],
                                       output_dict=True)
        accuracy_uk, recall_uk, precision_uk, f1_uk, scores_uk = mlf.calc_scores(labels_prob['predictions'],
                                                                                 labels_prob)
        scores_uk = classification_report(labels_prob['behavior_generalization'], labels_prob['predictions'],
                                          output_dict=True)
        scores = pd.DataFrame(scores).transpose()
        scores_uk = pd.DataFrame(scores_uk).transpose()
        scores_uk['proportion unknown'] = unknown_count['proportion']
        if len(labels_func) <= len(pred['behavior']):
            mislabeled, mislabeled2, mislabeled2_norm = mlf.calc_tables_eva(pred,
                                                                            pd.DataFrame(labels_func['predictions'],
                                                                                         index=labels_func[
                                                                                             'sample_index']),
                                                                            pd.DataFrame(
                                                                                labels_func['behavior_generalization'],
                                                                                index=labels_func['sample_index']),
                                                                            classes=class_order)
        else:
            mislabeled2 = pd.DataFrame()
            mislabeled2_norm = pd.DataFrame()
        print(accuracy)

        cm = confusion_matrix(labels_func['behavior_generalization'], labels_func['predictions'],
                              labels=[c for c in classes if c is not 'unknown'])
        cm_norm = normalize(cm, axis=1, norm='l1')
        cm_prob = confusion_matrix(labels_prob['behavior_generalization'], labels_prob['predictions'],
                                   labels=[c for c in classes if c is not 'unknown'])
        cm_prob_norm = normalize(cm_prob, axis=1, norm='l1')

        fig = plt.figure(figsize=(26, 20))
        if not isinstance(param, dict):
            defined_params = algorithm.get_params()

            default_rf = RandomForestClassifier()
            default_params = default_rf.get_params()

            param = {key: value for key, value in defined_params.items() if
                     key in default_params and value != default_params[key]}

        if sampling_func:
            title = (gen + ', ' + algorithm_name + ', ' + sampling_func[
                0] + ', ' + opt + ', using Parameters: ' + ', '.join(
                f'{key}: {value}' for key, value in param.items()) + ', \naccuracy: ' + str(
                round(accuracy, 3)) + '; recall: ' + str(round(recall, 3)) + '; precision: ' +
                     str(round(precision, 3)) + '; f1_score: ' + str(round(f1, 3)))
        else:
            title = (gen + ', ' + algorithm_name + ', ' + opt + ', using Parameters: ' + ', '.join(
                f'{key}: {value}' for key, value in param.items()) + ', \naccuracy: ' + str(
                round(accuracy, 3)) + '; recall: ' + str(round(recall, 3)) + '; precision: ' +
                     str(round(precision, 3)) + '; f1_score: ' + str(round(f1, 3)))

        if labels_func.shape[0] == len(ext_test[1]):
            title = 'External test with 10% of the dataset\n' + title
        else:
            title = 'Cross Validation using 90% of the dataset\n' + title

        fig.suptitle(title)
        gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 1])
        ax10 = fig.add_subplot(gs[0, 0])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_norm, display_labels=[c for c in classes if c is not 'unknown'])

        disp.plot(cmap='Blues', ax=ax10)
        ax10.set_title('Normalized Confusion Matrix')
        ax10.set_xticklabels(ax10.get_xticklabels(), rotation=90)
        ax11 = fig.add_subplot(gs[0, 1:2])
        ax11.axis('tight')
        ax11.axis('off')

        table = ax11.table(cellText=scores.values.round(3),
                           colLabels=scores.columns,
                           rowLabels=scores.index,
                           cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.5, 1.2)
        if not mislabeled2.empty:
            axs3 = fig.add_subplot(gs[1, :])
            axs3.axis('tight')
            axs3.axis('off')
            table = axs3.table(cellText=mislabeled2.values.astype(int),
                               colLabels=mislabeled2.columns,
                               rowLabels=[word[:2] for word in classes],
                               cellLoc='center', loc='center')

            colors = sim_func.COLOR_MAPPING_HTML
            first_letter_colors = {key[0]: value for key, value in colors.items()}
            # here, I am rotating the column labels and set the height so that the text is readable
            max_col_label_length = max(len(str(label))
                                       for label in mislabeled2.columns)

            header_height_scale = max(1.7, max_col_label_length / 2)
            for key, cell in table.get_celld().items():
                if key[0] == 0:
                    group = labels_func[behaviors_func == cell.get_text().get_text()][
                        'behavior_generalization'].unique()
                    print(group)
                    color = dict(colors).get(group[0])
                    cell.set_facecolor(color)
                    cell.get_text().set_rotation(90)
                    cell.get_text().set_ha('center')
                    cell.set_height(cell.get_height() * header_height_scale)
                width = cell.get_width()
                cell.set_width(width * 0.8)
            cmap_false = LinearSegmentedColormap.from_list(
                'custom_cmap', [(0.00, 'white'), (1, 'red')])
            cmap_right = LinearSegmentedColormap.from_list(
                'custom_cmap', [(0.00, 'white'), (1, 'green')])
            for i in range(1, len(mislabeled2.index) + 1):
                for j in range(-1, len(mislabeled2.columns)):
                    if j > -1:
                        group = labels_func[behaviors_func == table[(0, j)].get_text().get_text()][
                            'behavior_generalization'].unique()
                        if group[0] == mislabeled2.index[i - 1]:
                            value = mislabeled2_norm.iloc[i - 1, j]
                            color = cmap_right(value)
                        else:
                            value = mislabeled2_norm.iloc[i - 1, j]
                            color = cmap_false(value)
                    else:
                        color = dict(first_letter_colors).get(
                            table[(i, j)].get_text().get_text()[0])
                    table[(i, j)].set_facecolor(color)
                    table[(i, j)].set_height(0.05)
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.5, 1.2)

        ax41 = fig.add_subplot(gs[2, 0])
        ax41.hist(y_prob, bins=30)
        ax41.set_xlim([0,1])
        ax41.set_title('Histogram of prediction probabilities')

        ax42 = fig.add_subplot(gs[2, 1])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_prob_norm,
            display_labels=[c for c in classes if c in labels_func['behavior_generalization'].unique()])
        disp.plot(cmap='Blues', ax=ax42)
        ax42.set_title('Normalized Confusion Matrix after subtracting unknown behavior')
        ax42.set_xticklabels(ax42.get_xticklabels(), rotation=90)

        ax43 = fig.add_subplot(gs[2, 2])
        ax43.axis('tight')
        ax43.axis('off')

        table = ax43.table(cellText=scores_uk.values.round(3),
                           colLabels=scores_uk.columns,
                           rowLabels=scores_uk.index,
                           cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.5, 1.2)

        plt.subplots_adjust(wspace=0.5, hspace=0.3)
        if pdf:
            pdf.savefig(fig)
            plt.close(fig)
        else:
            plt.show()


def plotting_proportions_predictions(df: pd.DataFrame, title: str, label='unknown'):
    """
    plotting proportions of predictions probability of behavior labelled as unknown
    @param df: dataframe with prediction probabilities and labels including unknown
    @param title: title of the graph
    @param label: label for which the prediction probability proportions should be plotted
    @return: bar plot
    """
    df = df[df['pred_incl_unkn'] == label]
    df = df.sort_values(['pred'])
    df = df.reset_index()
    # Extract class labels and proportions
    classes = df['pred'].unique()

    class_boundaries = [df[df['pred'] == cls].index.min() for cls in classes] + [len(df)]

    x_positions = np.arange(len(df))

    colors = ['salmon', 'lightgreen', 'lightblue']

    fig, ax = plt.subplots(figsize=(20, 15))
    ax.stackplot(
        x_positions,
        df[classes].T,
        colors=colors,
        labels=classes
    )

    for boundary in class_boundaries[:-1]:
        ax.axvline(boundary - 0.5, color='black', linestyle='--', linewidth=1)

    x_ticks = [df[df['pred'] == cls].index.to_numpy().mean() for cls in classes]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(classes, rotation=45, ha='right')

    ax.set_xlabel('Class')
    ax.set_ylabel('Proportion')
    ax.set_title(title)
    ax.legend(title='Parts')

    plt.tight_layout()
    plt.show()


def output_hourly_contingents(y_prob_all: pd.DataFrame, path_gen: str, title: str):
    """

    @param y_prob_all: Dataframe of machine learning results (including columns: pred_incl_unkn, hour, date)
    @param path_gen: path to the directory, where pdfs should be saved
    @param title: additional title for the pdf page (usually simple or layered model)
    @return: pdf document with the plots of behavior contingents per month and hour
    """

    def time_to_float(t: datetime) -> float:
        """
        converting datetime to float
        @param t: time
        @return: # of hours (as float) since midnight
        """
        return t.hour + t.minute / 60.0

    def create_daylight_gradient(date_middle: datetime.date) -> np.array:
        """
        creating a daylight gradient using an input date and the available data for sunset, sunrise, etc. from the astral package
        for that day
        @param date_middle: day that should be used for sunlight calculations
        @return: array with gradient values for sunlight during the day, roughly ranging from 0 (dark) to 1 (noon)
        """
        location = LocationInfo("Berlin", "Germany", "GMT+1", latitude=52.52,
                                longitude=13.41)
        daylight = sun(location.observer, date=date_middle)

        sunrise = time_to_float(daylight['sunrise'])
        sunset = time_to_float(daylight['sunset'])
        dawn = time_to_float(daylight['dawn'])
        dusk = time_to_float(daylight['dusk'])
        noon = time_to_float(daylight['noon'])
        print(sunrise)
        print(sunset)

        hours = [0, dawn, sunrise, sunrise + (sunrise - dawn), noon, sunset - (dusk - sunset), sunset, dusk, 24]
        values = [0, 0.2, 0.5, 0.9, 1, 0.9, 0.5, 0.2, 0]

        new_hours = np.linspace(0, 24, num=100)

        new_values = np.interp(new_hours, hours, values)

        return new_values

    locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
    months = y_prob_all.groupby('month')
    output_pdf_path = path_gen + title + '.pdf'
    with (PdfPages(output_pdf_path) as pdf):
        for month, data in months:
            classes = data['pred_incl_unkn'].unique()
            classes = [key for key in sim_func.COLOR_MAPPING.keys() if key in classes]
            overall_counts = data['pred_incl_unkn'].value_counts(normalize=True).reindex(classes, fill_value=0)

            hourly_counts = data.groupby(['hour', 'pred_incl_unkn']).size().unstack(fill_value=0)
            hourly_proportions = hourly_counts.div(hourly_counts.sum(axis=1), axis=0)

            date_middle = data['date'].iloc[data.shape[0] // 2]
            if isinstance(date_middle, str):
                date_middle = datetime.strptime(date_middle, "%Y-%m-%d").date()

            day = create_daylight_gradient(date_middle)
            norm = plt.Normalize(day.min(), day.max())
            cmap = plt.get_cmap('plasma')

            # fig = plt.figure(figsize=(20, 10))
            fig = plt.figure(figsize=(16, 10))
            plt.rcParams.update({'font.size': 16})
            # gs = gridspec.GridSpec(2, 2, height_ratios=[5, 1], width_ratios=[1, 4])
            gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1]) #, width_ratios=[1, 4])
            # ax = fig.add_subplot(gs[0, 0])
            # bottom_value = 0
            #
            # # overall proportions as stack plot
            # for cls, color in sim_func.COLOR_MAPPING_HTML.items():
            #     if cls in overall_counts.index:
            #         ax.bar(
            #             x=[0],
            #             height=[overall_counts[cls]],
            #             color=color,
            #             bottom=bottom_value,
            #             label=cls,
            #         )
            #         bottom_value += overall_counts[cls]
            #
            #
            # ax.set_xticks([0])
            # ax.set_xticklabels(['Overall'])
            # ax.set_ylabel('Proportion')
            # ax.set_title(title + '\nOverall Class Proportions in month ' + str(month))


            # ax1 = fig.add_subplot(gs[0, 1])
            ax1 = fig.add_subplot(gs[0, 0])
            bottom_value = np.zeros(len(hourly_proportions))
            for cls, color in sim_func.COLOR_MAPPING_HTML.items():
                if cls in hourly_proportions.columns:
                    ax1.bar(
                        hourly_proportions.index,
                        hourly_proportions[cls],
                        bottom=bottom_value,
                        color=color,
                        label=cls,
                        align='edge'
                    )
                    bottom_value += hourly_proportions[cls]

            ax1.set_ylabel('Proportion')
            ax1.set_xlim(0, 24)
            ax1.set_title(calendar.month_name[int(month)])
            ax1.legend(loc='center', bbox_to_anchor=(0.5, 0.3), facecolor='white', edgecolor='white', framealpha=1)

            ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

            gradient = np.expand_dims(day, axis=0)
            ax2.imshow(gradient, aspect='auto', cmap=cmap, extent=[0, 24, 0, 1])

            ax2.set_yticks([])
            ax2.set_xlim(0, 24)
            ax2.set_xlabel('Hour')
            handles, labels_legend = ax1.get_legend_handles_labels()

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def plotting_vps(predictor_options: list, predictors: pd.DataFrame, name_dataset: str, palette: dict, option1,
                 pdf=None):
    """
    plotting violinplots per predictor

    @param predictor_options: which predictors should be plotted
    @param predictors: dataframe containing the predictor data
    @param name_dataset: which person recorded the dataset
    @param palette: color mapping for the behaviors
    @param option1: ['Normal violinplots', 'Testing generalization3 climbing and walking']
    @param pdf: pdf instance the images should be plotted to
    @return: pdf pages with violinplots
    """
    for o in predictor_options:
        print(o)
        pred_temp = predictors[[o, 'behavior']]
        pred_temp = pred_temp.dropna()

        grouped_data = pred_temp.groupby('behavior')
        grouped_data = sorted(grouped_data, key=lambda x: list(predictors['behavior'].unique()).index(x[0]))

        data_func = []
        behaviors_func = []
        missing = []
        for name1, group in grouped_data:
            values = group[o].values
            if len(values) > 0:
                data_func.append(values)
                behaviors_func.append(name1)
            else:
                missing.append(name1)

        if not data_func:
            print(f"No valid data for {o}, skipping this plot.")
            continue

        print(predictors['behavior'].unique())
        plt.figure(figsize=(20, 12))

        # plotting the violinplot: 'behavior' are the categories (x), the 'Ndyn' is the y-variable,
        # the violins are splitted (one side and color denotes one animal, other side and color the other),
        # quartiles are shown
        if option1 in ['Normal violinplots', 'Testing generalization3 climbing and walking']:
            violin_parts_func = plt.violinplot(data_func, showmedians=True)
            plt.xticks(ticks=np.arange(1, len(behaviors_func) + 1), labels=behaviors_func, rotation=90)
        else:
            data_first_func = predictors[predictors['fl'] == 'first']
            data_firstlast_func = predictors[predictors['fl'] == 'firstlast']

            data_left_func = [group[o].values for name_func, group in data_first_func.groupby('behavior')]
            data_right_func = [group[o].values for name_func, group in data_firstlast_func.groupby('behavior')]

            behaviors_func = predictors['behavior'].unique()

            for i_f in range(len(behaviors_func)):
                parts_left_func = plt.violinplot(data_left_func[i_f], positions=[i_f], points=60, widths=0.7,
                                                 showmeans=True, showextrema=True, showmedians=True,
                                                 bw_method=0.5, side='low')
                for pc_func in parts_left_func['bodies']:
                    pc_func.set_facecolor('cadetblue')
                    pc_func.set_edgecolor('black')
                    pc_func.set_alpha(0.7)

                parts_right_func = plt.violinplot(data_right_func[i_f], positions=[i_f], points=60, widths=0.7,
                                                  showmeans=True, showextrema=True, showmedians=True,
                                                  bw_method=0.5, side='high')
                for pc_func in parts_right_func['bodies']:
                    pc_func.set_facecolor('goldenrod')
                    pc_func.set_edgecolor('black')
                    pc_func.set_alpha(0.7)

            violin_parts_func = [parts_left_func, parts_right_func]
            plt.xticks(ticks=np.arange(len(behaviors_func)), labels=behaviors_func, rotation=90)
        # title is the filepath for better understanding and the coverage values
        if len(missing) == 0:
            plt.title(f'{name_dataset} Violinplot of the {o}')
        else:
            plt.title(f'{name_dataset} Violinplot of the {o}, NaN in behaviors: {missing}')

        if option1 == 'Normal violinplots':
            plt.ylim(bottom=0)
        plt.grid()

        if isinstance(violin_parts_func, dict) and 'bodies' in violin_parts_func:
            for i_f, vp_func in enumerate(violin_parts_func['bodies']):
                behavior_func = behaviors_func[i_f]
                vp_func.set_facecolor(palette[behavior_func])
                vp_func.set_edgecolor('black')
                vp_func.set_alpha(0.7)

        if option1 == 'Testing generalization3 climbing and walking':
            climbing_patch = mpatches.Patch(color='g', label='climbing')
            walking_patch = mpatches.Patch(color='y', label='walking')
            plt.legend(handles=[climbing_patch, walking_patch])

        plt.tight_layout()
        if pdf:
            pdf.savefig()
            plt.close()
        else:
            plt.show()

def sankey_diagram(all_data: pd.DataFrame, filepath: str):
    """
    Generates a Sankey diagram.

    @param all_data: DataFrame with columns "true" and "prediction"
    @param filepath: Output path for saving the figure
    """

    true_classes = list(all_data['true'].unique())
    pred_classes = list(all_data['prediction'].unique())

    ordered_classes = ["resting", "exploring", "climbing", "walking", "intermediate energy", "high energy", "unknown"]
    all_data['true'] = pd.Categorical(all_data['true'] + '_in', [c + '_in' for c in ordered_classes if c in true_classes])
    all_data['prediction'] = pd.Categorical(all_data['prediction'] + '_out', [c + '_out' for c in ordered_classes if c in pred_classes])
    all_data.sort_values(['true', 'prediction'], inplace=True)
    all_data.reset_index(drop=True)

    true_classes = [c + '_in' for c in true_classes]
    pred_classes = [c + '_out' for c in pred_classes]

    # Append "_in" and "_out" to distinguish true and predicted classes
    # all_data["true"] = all_data["true"] + "_in"
    # all_data["prediction"] = all_data["prediction"] + "_out"

    # existing_true_classes = list(all_data["true"].unique())
    # existing_pred_classes = list(all_data["prediction"].unique())
    #
    # true_classes = [cls + "_in" for cls in ordered_classes if cls + "_in" in existing_true_classes]
    # pred_classes = [cls + "_out" for cls in ordered_classes if cls + "_out" in existing_pred_classes]


    node_labels = true_classes + pred_classes
    label_to_index = {label: i for i, label in enumerate(node_labels)}
    index_to_label = {i: label for i, label in enumerate(node_labels)}

    df_counts = all_data.groupby(["true", "prediction"]).size().reset_index(name="count")

    source = df_counts["true"].map(label_to_index).tolist()
    target = df_counts["prediction"].map(label_to_index).tolist()
    values = df_counts["count"].tolist()

    node_colors = [
        sim_func.COLOR_MAPPING_HTML[label.replace("_in", "").replace("_out", "")]
        if "dummy" not in label else "rgba(255,255,255,0)"  # Transparent color for dummy nodes
        for label in node_labels
    ]
    link_colors = [
        sim_func.COLOR_MAPPING_HTML[index_to_label[x].replace("_in", "").replace("_out", "")]
        for x in target
    ]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=30,
            line=dict(color="black", width=0.5),
            color=node_colors,
            label=[""] * len(node_labels)
        ),
        link=dict(
            source=source,
            target=target,
            value=values,
            color=link_colors
        )
    ))

    annotations = [
        dict(
            x=1.1,
            y=0.65 - (i * 0.05),
            text=f"<span style='color:{color}'>■</span> {label.replace('_in', '').replace('_out', '')}",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=24)
        )
        for i, (label, color) in enumerate(zip(node_labels[5:], node_colors[5:]))
        if "dummy" not in label
    ]

    fig.update_layout(
        font_size=18,
        annotations=annotations,
        width=1200,
        height=800,
        margin=dict(l=100, r=400, t=50, b=50)
    )

    fig.write_image(filepath, scale=3)