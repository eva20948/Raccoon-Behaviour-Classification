#!/usr/bin/python3
"""
Filename: results_wild_data_summary.py
Author: Eva Reinhardt
Date: 2025-01-16
Version: 1.0
Description: summarizing the results from the wild data's classification
including analysis for distances between behaviour proportions of individuals including hfi,
behaviour proportions generally, yearly overview of proportions per dataset, plotting behaviour for death detection
of specific individuals, analysis of different probability thresholds

Functions in this file:

network_creation(): displays network of label sequence

add_diagram(): adds diagram of overall proportions as well as tables of proportions and min/max timestamp

histograms(): adds histograms of predictions probability

distances_between_individuals(): adds distance matrix between individuals (optional sorted by hfi with respective hfi
diagram)
"""

import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors

from collections import defaultdict
import pickle
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
from raccoon_acc_setup import plot_functions as plt_func
from raccoon_acc_setup import variables_simplefunctions as sim_func
import os
import re
from scipy.spatial import distance_matrix
import seaborn as sns
import json
from astral import LocationInfo
import astral
import datetime
import scipy.cluster.hierarchy as sch

def network_creation(label_sequence: list, title: str):
    """
    Function to create network visualization of subsequent behaviours
    @param label_sequence: the predicted behaviours in the correct order
    @param title: ideally logger number
    @return: creates subplots
    """
    edges_red = [(label_sequence[i], label_sequence[i + 1]) for i in range(len(label_sequence) - 1) if
                 label_sequence[i] != label_sequence[i + 1]]
    edge_counts_red = Counter(edges_red)

    total_transitions_red = sum(edge_counts_red.values())

    edges = [(label_sequence[i], label_sequence[i + 1]) for i in range(len(label_sequence) - 1)]
    edge_counts = Counter(edges)

    total_transitions = sum(edge_counts.values())

    G_red = nx.DiGraph()
    G = nx.DiGraph()
    for (src, dst), count in edge_counts_red.items():
        G_red.add_edge(src, dst, weight=count / total_transitions_red)
    for (src, dst), count in edge_counts.items():
        G.add_edge(src, dst, weight=count / total_transitions)

    weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
    min_w, max_w = min(weights), max(weights)
    norm_weights = (weights - min_w) / (max_w - min_w) if max_w > min_w else np.ones_like(weights)

    cmap = plt.cm.viridis
    edge_colors = [cmap(w) for w in norm_weights]

    pos = nx.circular_layout(G)

    weights_red = np.array([G_red[u][v]['weight'] for u, v in G_red.edges()])
    min_w_red, max_w_red = min(weights_red), max(weights_red)
    norm_weights_red = (weights_red - min_w_red) / (max_w_red - min_w_red) if max_w_red > min_w_red else np.ones_like(
        weights_red)

    cmap_red = plt.cm.viridis
    edge_colors_red = [cmap_red(w) for w in norm_weights_red]

    pos_red = nx.circular_layout(G_red)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    fig.suptitle(title)

    nx.draw(G, pos, with_labels=True, node_size=2500, node_color="lightblue",
            font_size=12, font_weight="bold", ax=ax1)

    edges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2 + 5 * norm_weights,
                                   arrowstyle='-|>', arrowsize=20, ax=ax1)

    edge_labels = {(src, dst): f"{100 * G[src][dst]['weight']:.1f}%" for src, dst in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color="black", ax=ax1)

    norm = mcolors.Normalize(vmin=min_w, vmax=max_w)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax1, orientation="vertical", fraction=0.03, pad=0.04)
    cbar.set_label("Transition Probability", fontsize=12)

    ax1.set_title("Label Transition Network", fontsize=14)

    nx.draw(G_red, pos_red, with_labels=True, node_size=2500, node_color="lightblue",
            font_size=12, font_weight="bold", ax=ax2)

    edges = nx.draw_networkx_edges(G_red, pos_red, edge_color=edge_colors_red, width=2 + 5 * norm_weights_red,
                                   arrowstyle='-|>', arrowsize=20, ax=ax2)

    edge_labels_red = {(src, dst): f"{100 * G_red[src][dst]['weight']:.1f}%" for src, dst in G_red.edges()}
    nx.draw_networkx_edge_labels(G_red, pos_red, edge_labels=edge_labels_red, font_size=10, font_color="black", ax=ax2)

    norm = mcolors.Normalize(vmin=min_w_red, vmax=max_w_red)
    sm = plt.cm.ScalarMappable(cmap=cmap_red, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax2, orientation="vertical", fraction=0.03, pad=0.04)
    cbar.set_label("Transition Probability", fontsize=12)

    ax2.set_title("Label Transition Network (Proportions, No Self-Loops)", fontsize=14)

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    else:
        plt.show()


def add_diagram(position: tuple, dictionary: dict, title: str, times_min_max: dict = None):
    """
    funtion to add diagrams of overall proportions as well as a table with the proportions and a table with earliest
    and latest timestamp/datetime
    @param position: position in the overall figure
    @param dictionary: overall proportions in dict format
    @param title: ideally logger number
    @param times_min_max: minimum and maximum timestamp, dict format
    @return: subfigures (plot, table, table)
    """
    ax = fig.add_subplot(gs[position])

    table_data = pd.DataFrame()

    positions = {'mw_layered': 0, 'mw_simple': 1, 'wo_mw_layered': 2, 'wo_mw_simple': 3}

    times_min_max = pd.DataFrame(times_min_max, index=[1])
    for name, overall_counts in dictionary.items():
        i = positions[name]
        overall_counts = pd.Series(overall_counts)
        overall_counts = pd.DataFrame(overall_counts, columns=['count'])
        overall_counts['count'] = overall_counts.div(overall_counts['count'].sum(axis=0), axis=1)
        bottom_value = 0
        for cls, color in sim_func.COLOR_MAPPING_HTML.items():
            if cls in overall_counts.index:
                ax.bar(
                    x=[i],
                    height=[overall_counts.loc[cls, 'count']],
                    color=color,
                    bottom=bottom_value,
                    label=cls,
                )
                bottom_value += overall_counts.loc[cls, 'count']
        overall_counts = overall_counts.rename(columns={'count': name})
        table_data = table_data.merge(overall_counts, 'outer', left_index=True, right_index=True, suffixes=('', name))

    table_data = table_data[['mw_layered', 'mw_simple', 'wo_mw_layered', 'wo_mw_simple']]

    print('set xticklabels')
    ax.set_xticks(range(4))
    ax.set_xticklabels(table_data.columns, rotation=45)
    ax.set_ylabel('Proportion')
    ax.set_title(title)

    ax1 = fig.add_subplot(gs[position[0], 1])
    ax1.axis('tight')
    ax1.axis('off')

    table = ax1.table(cellText=table_data.values.round(3),
                      colLabels=table_data.columns,
                      rowLabels=table_data.index,
                      cellLoc='center', loc='center')
    for key, cell in table.get_celld().items():
        cell.set_height(0.2)
        cell.set_width(0.2)

    if not times_min_max.empty:
        ax2 = fig.add_subplot(gs[position[0], 2])
        ax2.axis('tight')
        ax2.axis('off')

        table = ax2.table(cellText=times_min_max.values,
                          colLabels=times_min_max.columns,
                          rowLabels=times_min_max.index,
                          cellLoc='center', loc='center')
        for key, cell in table.get_celld().items():
            cell.set_height(0.5)
            cell.set_width(0.8)



def histograms(df: pd.DataFrame, title: str):
    """
    function to plot a histogram of prediction probabilities
    @param df: dataframe containing predictions and probabilities
    @param title: ideally logger number
    @return: histograms for prediction probabilities by behaviour
    """
    pred = [c for c in df.columns if 'pred' in c]
    prob = [c for c in df.columns if 'prob' in c]
    df = df[pred + prob]
    grouped = df.groupby(pred)

    fig, ax = plt.subplots(ncols=len(grouped), nrows=1, figsize=(16, 8))
    fig.suptitle(title)
    i = 0
    for name, group in grouped:
        ax[i].set_title(name[0])
        if name[0] != 'unknown':
            ax[i].hist(group[prob], bins=30)
            ax[i].set_xlim([0.6, 1])
        else:
            ax[i].hist(group[prob], bins=30, color='mediumorchid')
            ax[i].set_xlim([0, 0.6])
        i += 1
    fig.tight_layout()
    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    else:
        plt.show()


def distances_between_individuals(stats: dict, option: str, hfi_data: pd.DataFrame = pd.DataFrame()):
    """
    function to plot a distance matrix
    @param stats: dictionary containing the statistics of the individuals (class proportions)
    @param option: 'all_datasets' --> requires a dict of proportions from all datasets, overall distance matrix,
        anything else --> distance matrix of class proportions form individuals of the same dataset
    @return: distance matrix
    """
    if option == 'all_datasets':
        nr_row = 1
        if not hfi_data.empty:
            ordering = hfi_data['logger'].to_list()
            nr_row=2
            transformed_dict = {}
            for name, loggers in stats.items():
                for logger, models in loggers.items():
                    del models['wo_mw_layered']
                    del models['mw_simple']
                    del models['wo_mw_simple']
                    new_logger_name = f"{logger}_{name}"
                    transformed_dict[new_logger_name] = models
            stats = {}

            try:
                ordering
            except NameError:
                print("Variable does not exist!")
            else:
                for logger in ordering:
                    logger_name = [key for key, item in transformed_dict.items() if str(logger) in key]
                    if len(logger_name) != 0:
                        logger_name = logger_name[0]
                        hfi_data = hfi_data.replace(logger, logger_name)
                        stats[logger_name] = transformed_dict[logger_name]
                    else:
                        hfi_data = hfi_data.loc[hfi_data['logger'] != logger]


            size = [8, 9]

            new_order_stat = {}
            for logger, models in stats.items():
                for model, values in models.items():
                    if model not in new_order_stat:
                        new_order_stat[model] = {}
                    values_new = pd.Series(values)
                    values_new = values_new.reindex(sim_func.COLOR_MAPPING.keys(), fill_value=0)
                    new_order_stat[model][logger] = values_new.to_list()

            fig, axes = plt.subplots(nrows=nr_row, ncols=2, figsize=size, height_ratios=[3,1], width_ratios=[8,1], sharex=True)

            matrices = []
            heatmaps = []

            for i, (model, logger_dict) in enumerate(new_order_stat.items()):
                proportions = []
                labels = []
                for logger, values in logger_dict.items():
                    values_norm = [float(v) / sum(values) if sum(values) > 0 else 0 for v in values]
                    proportions.append(values_norm)
                    labels.append(logger)

                matrix = pd.DataFrame(distance_matrix(proportions, proportions), index=labels, columns=labels)

                heatmap = sns.heatmap(matrix, ax=axes[0,0], cbar=False, vmin=0, vmax=0.5, cmap="coolwarm")

                matrices.append(matrix)
                heatmaps.append(heatmap)

            cbar = fig.colorbar(heatmaps[0].collections[0], ax=axes[0,1], orientation="vertical", fraction=0.5, pad=0.04)
            cbar.set_label("Heatmap Intensity")
            axes[0,1].axis('off')


            categories = list(hfi_data['logger'].unique())  # Get unique categories
            x_positions = np.arange(len(hfi_data['logger']))+0.5  # Assign integer positions
            category_to_x = {cat: pos for cat, pos in zip(categories, x_positions)}

            hfi_data['x_mapped'] = hfi_data['logger'].map(category_to_x)

            axes[1,0].errorbar(hfi_data['x_mapped'], hfi_data['mean'],
                             yerr=[hfi_data['mean']-hfi_data['min'], hfi_data['max']-hfi_data['mean']],
                                 fmt='.', capsize=2, capthick=1, elinewidth=1, color='black', label="Data Points")
            axes[1,0].set_ylim([hfi_data['min'].min()+0.01, hfi_data['max'].max()])
            axes[1,0].set_xticks(x_positions)
            axes[1,0].set_xticklabels(categories, rotation=90, ha='center')
            axes[1,0].set_ylabel('Mean HFI')
            axes[-1, -1].axis('off')

            fig.tight_layout()
            if pdf:
                pdf.savefig(fig)
                plt.close(fig)
            else:
                plt.show()



        else:
            transformed_dict = {}
            for name, loggers in stats.items():
                for logger, models in loggers.items():
                    if 'wo_mw_layered' in models:
                        del models['wo_mw_layered']
                    if 'mw_simple' in models:
                        del models['mw_simple']
                    if 'wo_mw_simple' in models:
                        del models['wo_mw_simple']
                    new_logger_name = f"{logger}_{name}"
                    transformed_dict[new_logger_name] = models

            size = [9, 8]

            new_order_stat = {}
            for logger, models in transformed_dict.items():
                for model, values in models.items():
                    if model not in new_order_stat:
                        new_order_stat[model] = {}
                    values_new = pd.Series(values)
                    values_new = values_new.reindex(sim_func.COLOR_MAPPING.keys(), fill_value=0)
                    new_order_stat[model][logger] = values_new.to_list()

            fig, axes = plt.subplots(nrows=nr_row, ncols=2, figsize=size, width_ratios=[8, 1])

            matrices = []
            heatmaps = []

            for i, (model, logger_dict) in enumerate(new_order_stat.items()):
                proportions = []
                labels = []
                for logger, values in logger_dict.items():
                    values_norm = [float(v) / sum(values) if sum(values) > 0 else 0 for v in values]
                    proportions.append(values_norm)
                    labels.append(logger)

            matrix = pd.DataFrame(distance_matrix(proportions, proportions), index=labels, columns=labels)

            heatmap = sns.heatmap(matrix, ax=axes[0], cbar=False, cmap="coolwarm")



            cbar = fig.colorbar(heatmap.collections[0], ax=axes[1], orientation="vertical", fraction=0.5,
                                pad=0.04)
            cbar.set_label("Heatmap Intensity")
            axes[-1].axis('off')
            fig.tight_layout()
            if pdf:
                pdf.savefig(fig)
                plt.close(fig)
            else:
                plt.show()

            condensed_matrix = sch.distance.squareform(matrix)


            linkage_matrix = sch.linkage(condensed_matrix,
                                         method='ward')

            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
            sch.set_link_color_palette(['cornflowerblue', 'yellowgreen', 'black'])
            sch.dendrogram(linkage_matrix, labels=matrix.columns, ax=axes, leaf_rotation=0, orientation="left", color_threshold=0.8,
                           above_threshold_color='black')
            x_labels = axes.get_ymajorticklabels()
            for lbl in x_labels:
                if "Katti" in lbl.get_text():
                    lbl.set_color('goldenrod')
                elif "Caro W" in lbl.get_text():
                    lbl.set_color('seagreen')
                else:
                    lbl.set_color('royalblue')
            plt.xlabel('Distance')
            plt.ylabel('Individuals')
            plt.xticks(rotation=90)

            pdf.savefig(fig)
            plt.close(fig)

        if pdf:
            pdf.savefig(fig)
            plt.close(fig)
        else:
            plt.show()

    else:
        size = [16, 8]
        nr_col = 4
        new_order_stat = {}
        for logger, models in stats.items():
            for model, values in models.items():
                if model not in new_order_stat:
                    new_order_stat[model] = {}
                values_new = pd.Series(values)
                values_new = values_new.reindex(sim_func.COLOR_MAPPING.keys(), fill_value=0)
                new_order_stat[model][logger] = values_new.to_list()

        fig, axes = plt.subplots(nrows=1, ncols=nr_col, figsize=size, sharey=True)

        matrices = []
        heatmaps = []

        for i, (model, logger_dict) in enumerate(new_order_stat.items()):
            proportions = []
            labels = []
            for logger, values in logger_dict.items():
                values_norm = [float(v) / sum(values) if sum(values) > 0 else 0 for v in values]
                proportions.append(values_norm)
                labels.append(logger)

            matrix = pd.DataFrame(distance_matrix(proportions, proportions), index=labels, columns=labels)

            heatmap = sns.heatmap(matrix, ax=axes[i], cbar=False, vmin=0, vmax=0.5, cmap="coolwarm")

            axes[i].set_title(model)

            if i > 0:
                axes[i].set_yticklabels([])

            matrices.append(matrix)
            heatmaps.append(heatmap)

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(heatmaps[0].collections[0], cax=cbar_ax)

        fig.tight_layout(rect=[0, 0, 0.9, 1])

        if pdf:
            pdf.savefig(fig)
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    filepaths_class = [sim_func.IMPORT_PATH_CLASS + f for f in os.listdir(sim_func.IMPORT_PATH_CLASS) if
                       os.path.isfile(os.path.join(sim_func.IMPORT_PATH_CLASS, f)) and '.csv' in f]
    filepaths_caros = [f for f in filepaths_class if 'Caro S' in f]
    filepaths_carow = [f for f in filepaths_class if 'Caro W' in f]
    filepaths_katti = [f for f in filepaths_class if 'Katti' in f]

    filepaths_all = {'Caro W': filepaths_carow,
                     'Caro S': filepaths_caros,
                     'Katti': filepaths_katti
                     }

    option = 'distance matrix'
    if option == 'everything':
        stats_all = {}
        for name, paths in filepaths_all.items():
            # option = guif.choose_option(['create hourly output', 'do not create hourly outputs'], name + ': create hourly outputs or not')
            option_2 = 'dont create hourly outputs'
            mw_layered = []
            mw_simple = []
            wo_mw_layered = []
            wo_mw_simple = []
            stats = {}
            times = {}
            stats_all[name] = {}
            if len(paths) != 0:
                file_dict = defaultdict(list)
                for filename in paths:
                    match = re.search(r"\d{4}", filename)
                    if match:
                        key = match.group()
                        file_dict[key].append(filename)

                file_dict = dict(file_dict)

                output_pdf_path = sim_func.EXPORT_PATH + 'wild_data_stats_all_carow_' + name + '.pdf'
                with (PdfPages(output_pdf_path) as pdf):
                    print(1)

                    for logger, filepaths in file_dict.items():
                        for filepath in filepaths:
                            print(filepath)
                            data = pd.read_csv(filepath, sep=',', low_memory=False)
                            print(logger)
                            if str(logger) not in stats:
                                stats[str(logger)] = {}
                                times[str(logger)] = {}
                            df = data[['datetime', 'pred_incl_unkn']]
                            df = df.dropna(axis=0)
                            prob_cols = [c for c in data.columns if
                                         c in ['resting', 'climbing', 'exploring', 'walking', 'intermediate energy',
                                               'high energy', 'unknown']]
                            prob_cols = data[prob_cols]
                            row_max = prob_cols.max(axis=1)
                            df['probability'] = row_max
                            times[str(logger)]['min_datetime'] = df['datetime'].min()
                            times[str(logger)]['max_datetime'] = df['datetime'].max()
                            print(2)

                            if '_predictions_mw_layered.csv' in filepath:
                                df_mw_layered = df.rename(
                                    columns={'pred_incl_unkn': 'pred_mw_layered', 'probability': 'prob_mw_layered'})
                                # mw_layered.append(df)
                                stats[str(logger)]['mw_layered'] = df_mw_layered['pred_mw_layered'].value_counts()
                                title = 'Layered  mw'
                                network_creation(df_mw_layered['pred_mw_layered'].to_list(),
                                                 name + str(logger) + ', Layered Model with Moving Window')
                                histograms(df_mw_layered, title)
                                if option_2 == 'create hourly outputs':
                                    plt_func.output_hourly_contingents(data,
                                                                       '/media/eva/eva-reinhar/your folders/05 intermediate results/wild data classification/results/' + 'wild_data_' + name + str(
                                                                           logger), title)
                            elif '_predictions_mw_simple.csv' in filepath:
                                stats[str(logger)]['mw_simple'] = df['pred_incl_unkn'].value_counts()
                                df_mw_simple = df.rename(
                                    columns={'pred_incl_unkn': 'pred_mw_simple', 'probability': 'prob_mw_simple'})
                                # mw_simple.append(df)
                                title = 'Simple mw'
                                network_creation(df_mw_simple['pred_mw_simple'].to_list(),
                                                 name + str(logger) + ', Simple Model with Moving Window')
                                histograms(df_mw_simple, title)
                            elif '_predictions_wo_mw_layered.csv' in filepath:
                                df_wo_mw_layered = df.rename(columns={'pred_incl_unkn': 'pred_wo_mw_layered',
                                                                      'probability': 'prob_wo_mw_layered'})
                                # wo_mw_layered.append(df)
                                stats[str(logger)]['wo_mw_layered'] = df_wo_mw_layered[
                                    'pred_wo_mw_layered'].value_counts()
                                title = 'Layered wo mw'
                                network_creation(df_wo_mw_layered['pred_wo_mw_layered'].to_list(),
                                                 name + str(logger) + ', Layered Model without Moving Window')
                                histograms(df_wo_mw_layered, title)
                            else:
                                stats[str(logger)]['wo_mw_simple'] = df['pred_incl_unkn'].value_counts()
                                df_wo_mw_simple = df.rename(
                                    columns={'pred_incl_unkn': 'pred_wo_mw_simple', 'probability': 'prob_wo_mw_simple'})
                                # wo_mw_simple.append(df)
                                title = 'Simple wo mw'
                                network_creation(df_wo_mw_simple['pred_wo_mw_simple'].to_list(),
                                                 name + str(logger) + ', Simple Model without Moving Window')
                                histograms(df_wo_mw_simple, title)

                        fig = plt.figure(figsize=(16, 8))
                        gs = gridspec.GridSpec(1, 3, width_ratios=[2, 4, 1])
                        specified_times = times[logger]
                        add_diagram((0, 0), stats[logger], logger, specified_times)
                        plt.subplots_adjust(hspace=0.7)
                        plt.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)

                    stats_all[name] = stats


                    print(4)

                    stats_overall = defaultdict(lambda: defaultdict(int))

                    for logger, models in stats.items():
                        for model, classes in models.items():
                            for cls, count in classes.items():
                                stats_overall[model][cls] += count

                    stats_overall = {model: dict(classes) for model, classes in stats_overall.items()}

                    print('fig start')
                    logger_nr = len(stats)
                    fig = plt.figure(figsize=(16, 8))
                    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 4, 1])

                    min_time = None
                    max_time = None
                    for logger, min_max in times.items():
                        if min_time:
                            if min_time > min_max['min_datetime']:
                                min_time = min_max['min_datetime']
                            if max_time < min_max['max_datetime']:
                                max_time = min_max['max_datetime']
                        else:
                            min_time = min_max['min_datetime']
                            max_time = min_max['max_datetime']
                    times['overall'] = {}
                    times['overall']['min_datetime'] = min
                    times['overall']['max_datetime'] = max
                    add_diagram((0, 0), stats_overall, 'Overall proportions', times['overall'])

                    plt.subplots_adjust(hspace=0.7)
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

                    with open('nested_dict.pkl', 'wb') as f:
                        pickle.dump(stats_all, f)

                    distances_between_individuals(stats, 'single_dataset')


        with (PdfPages(sim_func.EXPORT_PATH + 'Zusammenfassung_all.pdf') as pdf):
            distances_between_individuals(stats_all, 'all_datasets')

    elif option == 'distance matrix':
        with open('nested_dict.pkl', 'rb') as f:
            stats_all = pickle.load(f)

        data = []
        with open('/media/eva/eva-reinhar/your folders/05 intermediate results/hfi_mean.txt', 'r') as f:
            for line in f:
                try:
                    index, dict_str = line.split(" ", 1)
                    adding = dict_str.strip()
                    adding = adding.replace("'", "\"")
                    print(adding)
                    data.append((int(index), json.loads(adding)))
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Skipping line due to error: {line.strip()} - {e}")

        hfi_data = pd.DataFrame([row[1] for row in data])
        hfi_data["logger"] = [row[0] for row in data]
        hfi_data = hfi_data.sort_values(['mean'])



        with (PdfPages(sim_func.EXPORT_PATH + 'Zusammenfassung_hfi.pdf') as pdf):
            distances_between_individuals(stats_all, 'all_datasets', hfi_data)

        with (PdfPages(sim_func.EXPORT_PATH + 'Zusammenfassung_normal_dist_mat.pdf') as pdf):
            distances_between_individuals(stats_all, 'all_datasets')

    elif option == 'stats overall':
        with open('/home/eva/Schreibtisch/Master/NeuerVersuch/Scripts/nested_dict.pkl', 'rb') as f:
            stats_all = pickle.load(f)
        output_pdf_path = sim_func.EXPORT_PATH + 'wild_data_stats_overall.pdf'
        with (PdfPages(output_pdf_path) as pdf):
            for name, d in stats_all.items():

                stats = d
                stats_overall = defaultdict(lambda: defaultdict(int))

                for logger, models in stats.items():
                    for model, classes in models.items():
                        for cls, count in classes.items():
                            stats_overall[model][cls] += count

                stats_overall = {model: dict(classes) for model, classes in stats_overall.items()}

                print('fig start')
                logger_nr = len(stats)
                fig = plt.figure(figsize=(16, 8))
                gs = gridspec.GridSpec(1, 2, width_ratios=[2, 4])
                fig.suptitle(name)
                add_diagram((0, 0), stats_overall, 'Overall proportions')

                plt.subplots_adjust(hspace=0.7)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            flattened_dict = {}

            for name, loggers in stats_all.items():
                for logger, models in loggers.items():
                    if logger not in flattened_dict:
                        flattened_dict[logger] = {}
                    flattened_dict[logger].update(models)

            stats_overall = defaultdict(lambda: defaultdict(int))

            for logger, models in flattened_dict.items():
                for model, classes in models.items():
                    for cls, count in classes.items():
                        stats_overall[model][cls] += count

            stats_overall = {model: dict(classes) for model, classes in stats_overall.items()}
            print('fig start')
            logger_nr = len(stats_all)
        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 4])
        fig.suptitle('All Datasets')
        add_diagram((0, 0), stats_overall, 'Overall proportions')

        plt.subplots_adjust(hspace=0.7)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        data_stats = []
        j = 0
        for name, logger_dict in stats_all.items():
            for logger, models_dict in logger_dict.items():
                logger_new = name + logger
                j = j + 1
                values = []
                class_names_fin = []
                for model, classes in models_dict.items():
                    value_overall = 0
                    values_temp = []
                    for class_name, value in classes.items():
                        new_class_name = model + '_' + class_name
                        value_overall = value_overall + value
                        values_temp.append(value)
                        class_names_fin.append(new_class_name)
                    values_temp = [v / value_overall for v in values_temp]
                    values = values + values_temp
                new_line = pd.DataFrame([[logger_new] + values], columns=['name'] + class_names_fin)
                data_stats.append(new_line)

        data_stats = pd.concat(data_stats)
        data_stats['wo_mw_unkn_sim_layered'] = (data_stats['wo_mw_simple_unknown'] - data_stats[
            'wo_mw_layered_unknown']) / data_stats['wo_mw_simple_unknown']
        data_stats['simple model comp wo mw and mw'] = (data_stats['wo_mw_simple_unknown'] - data_stats[
            'mw_simple_unknown']) / data_stats['wo_mw_simple_unknown']
        data_stats['layered model comp wo mw and mw'] = (data_stats['wo_mw_layered_unknown'] - data_stats[
            'mw_layered_unknown']) / data_stats['wo_mw_layered_unknown']
        data_stats['mw comp simple to layered'] = (data_stats['mw_simple_unknown'] - data_stats[
            'mw_layered_unknown']) / data_stats['mw_simple_unknown']

        means = data_stats[[col for col in data_stats if col != 'name']].mean()
        means['name'] = 'mean'
        maxes = data_stats[[col for col in data_stats if col != 'name']].max()
        maxes['name'] = 'max'
        mins = data_stats[[col for col in data_stats if col != 'name']].min()
        mins['name'] = 'min'
        data_stats = pd.concat([data_stats, means.to_frame().T, maxes.to_frame().T,
                                mins.to_frame().T], ignore_index=True)
        print(data_stats)

        data_stats.to_csv('wild_data_stats.csv')


    elif option == "death detection":
        for name, paths in filepaths_all.items():
            for filename in paths:
                match = re.search(r"\d{4}", filename)
                option2 = 'cutting dead time'
                if option2 == 'figures':
                    if ('1633' in filename):
                        data = pd.read_csv(filename, sep=',', low_memory=False)
                        data['datetime'] = pd.to_datetime(data['datetime'], format='mixed')
                        data = data.sort_values(by='datetime', ascending=False)
                        latest_date = data.iloc[0]['datetime']
                        start_date = latest_date - pd.DateOffset(months=5)
                        data = data[(data['datetime'] >= start_date) & (data['datetime'] <= latest_date)].sort_values(
                            by='datetime')

                        data['pred_incl_unkn'] = pd.Categorical(data['pred_incl_unkn'],
                                                                categories=["resting", "exploring", "climbing",
                                                                            "walking", "intermediate energy",
                                                                            "high energy", "unknown"])
                        data = data.dropna(subset=['pred_incl_unkn'], axis=0)
                        plt.figure(figsize=(12, 5))
                        plt.scatter(data['datetime'], data['pred_incl_unkn'], marker='o')
                        plt.xticks(data['datetime'].dt.normalize().unique(), rotation=45)

                        plt.xlabel('Date')
                        plt.ylabel('Behaviour')
                        plt.title(match[0])
                        plt.grid()
                        plt.show()


                elif option2 == 'create gaps files':
                    data = pd.read_csv(filename, sep=',', low_memory=False)
                    data['datetime'] = pd.to_datetime(data['datetime'], format='mixed')
                    data = data.sort_values(by='datetime')

                    data['gap'] = data['datetime'].diff()

                    threshold = pd.Timedelta(minutes=10)

                    gaps = data[data['gap'] > threshold][['datetime', 'gap']]
                    print(gaps)
                    gaps.to_csv(match[0] + '_gaps.txt', sep='\t', index=False)

                    print("Gaps saved to gaps.txt")

                elif option2 == 'cutting dead time':
                    if match[0] in ['5123', '5129', '1633']:
                        data = pd.read_csv(filename, sep=',', low_memory=False)
                        data['datetime'] = pd.to_datetime(data['datetime'], format='mixed')
                        if match[0] == '5123':
                            cut_off_date = '2017-09-09 00:00:00'
                        elif match[0] == '5129':
                            cut_off_date = '2017-09-01 00:00:00'
                        elif match[0] == '1633':
                            cut_off_date = '2011-11-01 00:00:00'

                        data = data[(data['datetime'] < cut_off_date)].sort_values(
                            by='datetime')

                        filename_out = filename.split('/')
                        filename_out = filename_out[-1]

                        data.to_csv(filename_out)

    elif option == "year overview":
        with (PdfPages('year_overview_behaviour_proportions.pdf') as pdf):
            months_overall = pd.DataFrame(np.zeros((12, 7)), columns=['resting', 'climbing', 'walking', 'exploring', 'intermediate energy',
                                                    'high energy', 'unknown'], index = range(1, 13))

            location = LocationInfo("Berlin", "Germany", "GMT+1", latitude=52.52,
                                    longitude=13.41)

            average_night_lengths = []

            year = 2019

            for month in range(1, 13):
                date = datetime.date(year, month, 15)
                s = astral.sun.sun(location.observer, date=date)
                nightstart = s["sunset"]
                nightend = astral.sun.sun(location.observer, date=date + datetime.timedelta(days=1))["sunrise"]

                night_length = (nightend-nightstart).total_seconds() / 3600 /24
                average_night_lengths.append(night_length)

            average_night_lengths = np.array(average_night_lengths)


            for name, filepaths in filepaths_all.items():

                months_name = pd.DataFrame(np.zeros((12, 7)), columns=['resting', 'climbing', 'walking', 'exploring', 'intermediate energy',
                                                    'high energy', 'unknown'], index = range(1, 13))
                for filepath in filepaths:
                    if 'predictions_mw_layered' in filepath:
                        data = pd.read_csv(filepath, sep=',', low_memory=False)
                        data['datetime'] = pd.to_datetime(data['datetime'], format='mixed')
                        data['month'] = data['datetime'].dt.month
                        months = data.groupby('month')
                        for month, data_month in months:
                            behavior_counts = data_month['pred_incl_unkn'].value_counts().to_dict()
                            for beh in months_name.columns:
                                months_name.loc[month,beh] += behavior_counts.get(beh, 0)

                months_prop = months_name.div(months_name.sum(axis=1), axis=0)

                fig = plt.figure(figsize=(5, 5))
                for column in months_prop.columns:
                    plt.plot(months_prop.index, months_prop[column], marker='o',
                             label=column, color=sim_func.COLOR_MAPPING_HTML.get(column, 'black'))  # Default to black if key is missing
                plt.plot(months_prop.index, average_night_lengths, label='night hours', color='black')
                plt.xlabel("Month")
                plt.ylim((0,0.7))
                plt.ylabel("Proportion")
                plt.title(name)

                plt.grid(True)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

                months_overall = months_overall + months_name

            months_prop = months_overall.div(months_overall.sum(axis=1), axis=0)
            fig = plt.figure(figsize=(5, 5))
            for column in months_prop.columns:
                plt.plot(months_prop.index, months_prop[column], marker='o',
                         label=column,
                         color=sim_func.COLOR_MAPPING_HTML.get(column, 'black'))  # Default to black if key is missing
            plt.plot(months_prop.index, average_night_lengths, label='night hours', color = 'black')
            plt.xlabel("Month")
            plt.ylim((0, 0.7))
            plt.ylabel("Proportion")
            plt.title('All data sets')

            plt.grid(True)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            fig_legend = plt.figure(figsize=(2, 2))

            ax = fig_legend.add_subplot(111)
            ax.axis("off")

            handles = [
                plt.Line2D([0], [0], color=sim_func.COLOR_MAPPING_HTML.get(column, 'black'), marker='o', label=column)
                for column in months_prop.columns]
            handles.append(plt.Line2D([0], [0], color='black', label='night hours'))

            legend = ax.legend(handles=handles, loc='center')
            pdf.savefig(fig_legend)
            plt.close(fig_legend)

    elif option == "prob threshold":
        with (PdfPages('probability_threshold.pdf') as pdf):
            stats_unknown = pd.DataFrame(np.zeros((6, 3)),columns=['unknown_count', 'intermediate_count', 'overall_count'], index=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            for name, filepaths in filepaths_all.items():
                stats_name = pd.DataFrame(np.zeros((6, 3)),columns=['unknown_count', 'intermediate_count', 'overall_count'],
                                             index=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                for filepath in filepaths:
                    if 'predictions_mw_layered' in filepath:
                        data = pd.read_csv(filepath, sep=',', low_memory=False)
                        data_unknown = data[['resting', 'intermediate energy', 'high energy']]
                        data_intermediate = data[['climbing', 'walking', 'exploring']]
                        data_intermediate = data_intermediate.dropna(axis=0)
                        for i in stats_name.index.to_list():
                            row_max_uk = data_unknown.max(axis=1, skipna=True)
                            row_max_ie = data_intermediate.max(axis=1, skipna=True)
                            count_below_threshold_uk = (row_max_uk < i).sum()
                            count_below_threshold_ie = (row_max_ie < i).sum()
                            stats_name.loc[i,'unknown_count'] += count_below_threshold_uk
                            stats_name.loc[i, 'intermediate_count'] += count_below_threshold_ie
                            stats_name.loc[i,'overall_count'] += data.shape[0]

                stats_unknown += stats_name
                stats_name['unknown_prop'] = stats_name['unknown_count']/stats_name['overall_count']
                stats_name['intermediate_prop'] = stats_name['intermediate_count'] / stats_name['overall_count']
                fig = plt.figure(figsize=(6, 4))
                plt.plot(stats_name.index, stats_name['unknown_prop'], marker='o', label='unknown',
                         color=sim_func.COLOR_MAPPING_HTML.get('unknown', 'black'))
                plt.plot(stats_name.index, stats_name['intermediate_prop'], marker='o',
                         label='intermediate energy',
                         color=sim_func.COLOR_MAPPING_HTML.get('intermediate energy', 'black'))
                plt.xlabel("threshold")
                plt.ylabel("Proportion")
                plt.legend()
                plt.title(name)
                plt.grid(True)
                pdf.savefig(fig)
                plt.close(fig)

            stats_unknown['unknown_prop'] = stats_unknown['unknown_count'] / stats_unknown['overall_count']
            stats_unknown['intermediate_prop'] = stats_unknown['intermediate_count'] / stats_unknown['overall_count']
            fig = plt.figure(figsize=(6, 4))
            plt.plot(stats_unknown.index, stats_unknown['unknown_prop'], marker='o',label='unknown',
                         color=sim_func.COLOR_MAPPING_HTML.get('unknown', 'black'))
            plt.plot(stats_unknown.index, stats_unknown['intermediate_prop'], marker='o',label='intermediate energy',
                         color=sim_func.COLOR_MAPPING_HTML.get('intermediate energy', 'black'))

            plt.xlabel("threshold")
            plt.ylabel("Proportion")
            plt.title('All data sets')
            plt.legend()
            plt.grid(True)
            pdf.savefig(fig)
            plt.close(fig)








