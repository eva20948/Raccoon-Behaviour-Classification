#!/usr/bin/python3

"""
Filename: data_statistics.py
Author: Eva Reinhardt
Date: 2024-12-09
Version: 1.0
Description: displaying simple statistics for the labelled data. Including class proportions,
confusion matrices, etc.
"""

from raccoon_acc_setup import predictor_calculation as pred_cal
from raccoon_acc_setup import importing_raw_data as im_raw
from raccoon_acc_setup import variables_simplefunctions as sim_func

import pandas as pd
import csv

from itertools import combinations

import re

import seaborn as sns
from matplotlib import pyplot as plt

import networkx as nx

import matplotlib.cm as cm
import matplotlib.colors as mcolors

import numpy as np

from snha4py.Snha import Snha

from matplotlib.backends.backend_pdf import PdfPages

if __name__ == "__main__":
    filepaths_peter = [
        sim_func.IMPORT_PARAMETERS['Peter']['filepath_pred']]
    filepaths_domi = [
        sim_func.IMPORT_PARAMETERS['Dominique']['filepath_pred']]


    filepaths = [filepaths_peter, filepaths_domi]

    option = 'corr matrix'

    pred = pred_cal.create_pred_complete(filepaths)

    pred = im_raw.convert_beh(pred, 'generalization', 'generalization3')
    pred = im_raw.convert_beh(pred, 'translation')
    print(pred.columns)
    #pred = pred.drop(['Ymin', 'Ymax', 'Pitch', 'Yvar', 'Ymax - Ymin', 'XZvar', 'XZmax - XZmin', 'Ydyn', 'XZdyn', 'Nvar', 'Odba', 'fft_max'], axis=1)
    columns_remain = sim_func.REDUCED_FEATURES
    columns_remain.append('behavior')
    columns_remain.append('behavior_generalization')
    pred_red = pred[columns_remain]
    predictors_red = pred_red.drop(['datetime', 'behavior', 'behavior_generalization'], axis=1)
    predictors = pred.drop(['datetime', 'behavior', 'behavior_generalization'], axis=1)

    peter_pred = pred[pred['behavior'].str.startswith('P')]
    domi_pred = pred[pred['behavior'].str.startswith('D')]
    print('Peter: ')
    print(peter_pred.shape)
    print('Dominique: ')
    print(domi_pred.shape)
    all_stats = pred['behavior'].value_counts()
    all_stats_tab = all_stats.reset_index()
    all_stats_tab.columns = ['behavior', 'count']

    total_count = all_stats_tab['count'].sum()

    all_stats_tab.loc[5] = ['Total', total_count]
    print(all_stats_tab)

    class_stats = pred['behavior_generalization'].value_counts()
    class_stats_tab = class_stats.reset_index()
    class_stats_tab.columns = ['behavior_class', 'count']

    total_count = class_stats_tab['count'].sum()

    class_stats_tab.loc[5] = ['Total', total_count]

    class_stats_tab['proportion'] = class_stats_tab['count']/total_count

    print(class_stats_tab)

    ten_percent = int(pred.shape[0] / 10)

    pred['datetime'] = pd.to_datetime(pred['datetime'])
    pred['date'] = pred['datetime'].dt.date

    dates = pred.groupby(['date'])

    print(ten_percent)
    num = 0
    lengths = []
    dates_list = []
    for date, group in dates:
        print(date)
        dates_list.append(date)
        print(len(group))
        lengths.append(len(group))
        print(group.shape)
        num = num+len(group)
        print(num)
    indices = list(range(len(lengths)))
    for combo in combinations(indices, 4):
        if sum(lengths[i] for i in combo) == ten_percent:
            print(combo)
            dates_final = [dates_list[c] for c in combo]
            num_final = [lengths[c] for c in combo]
            print(dates_final)
            print(num_final)



    if option == "corr matrix":

        pred_corr = predictors.corr(method='spearman').round(2)
        pred_corr_red = predictors_red.corr(method='spearman').round(2)



        plt.figure(figsize=(len(pred_corr.columns.values)*0.5,len(pred_corr.columns.values)*0.5))
        sns.heatmap(pred_corr,
                    xticklabels=pred_corr.columns.values,
                    yticklabels=pred_corr.columns.values,
                    annot=True,
                    cmap='coolwarm',
                    cbar=False)
        plt.tight_layout()
        plt.savefig('corr_matrix.pdf')

        plt.figure(figsize=(len(pred_corr_red.columns.values)*0.5,len(pred_corr_red.columns.values)*0.5))
        sns.heatmap(pred_corr_red,
                    xticklabels=pred_corr_red.columns.values,
                    yticklabels=pred_corr_red.columns.values,
                    annot=True,
                    cmap='coolwarm',
                    cbar=False)
        plt.tight_layout()
        plt.savefig('corr_matrix_red.pdf')

        network_data = pred_corr.stack().reset_index()
        network_data.columns = ['pred1', 'pred2', 'value']
        network_data.replace('XZmax - XZmin', 'XZmax -\nXZmin', inplace=True)
        network_data.replace('Ymax - Ymin', 'Ymax -\nYmin', inplace=True)
        #filtered = network_data.loc[(network_data['value'] > 0.8) & (network_data['pred1'] != network_data['pred2'])]
        network_data = network_data[network_data['pred1'] != network_data['pred2']]  # Remove self-loops

        G = nx.from_pandas_edgelist(network_data, 'pred1', 'pred2', edge_attr='value')

        weighted_degrees = {node: sum(G[node][neighbor]['value'] for neighbor in G[node]) for node in G.nodes()}

        sorted_nodes = sorted(weighted_degrees, key=weighted_degrees.get, reverse=True)

        strong_G = G.edge_subgraph([(u, v) for u, v, d in G.edges(data=True) if d['value'] > 0.8]).copy()

        # 3. Find connected components in the strong correlation graph
        components = list(nx.connected_components(strong_G))

        num_components = len(components)
        component_colors = plt.cm.Greens(np.linspace(0.1, 0.7, num_components))  # Shades of green

        node_color_map = {}

        for i, component in enumerate(components):
            for node in component:
                node_color_map[node] = component_colors[i]  # Assign the same color to connected nodes

        isolated_nodes = set(G.nodes()) - set([node for component in components for node in component])

        isolated_colors = plt.cm.Greens(
            np.linspace(0.2, 0.6, len(isolated_nodes)))
        for i, node in enumerate(isolated_nodes):
            node_color_map[node] = isolated_colors[i]


        filtered_edges = network_data.loc[
            (network_data['value'] > 0.8) & (network_data['pred1'] != network_data['pred2'])]

        rest_edges = network_data.loc[
            (network_data['value'] <= 0.8) & (network_data['pred1'] != network_data['pred2'])]

        G = nx.Graph()

        G.add_nodes_from(network_data['pred1'].unique())

        for _, row in filtered_edges.iterrows():
            G.add_edge(row['pred1'], row['pred2'], weight=row['value'])

        fig, ax = plt.subplots(figsize=(12, 10))

        pos = nx.arf_layout(G, scaling=2, a=5)

        for _, row in rest_edges.iterrows():
            G.add_edge(row['pred1'], row['pred2'], weight=row['value'])

        nx.draw_networkx_nodes(G, pos, node_color=[node_color_map[n] for n in G.nodes()],
                               node_size=2500,
                               node_shape='o',
                               alpha = 1)
        edge_styles = []
        edge_widths = []

        for u, v, d in G.edges(data=True):
            if abs(d['weight']) > 0.8:
                edge_styles.append('solid')
                edge_widths.append(2 + 5 * (d['weight'] - 0.8))  # Scale edge width by weight
            else:
                edge_styles.append('dashed')
                edge_widths.append(1)
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        norm = mcolors.Normalize(vmin=-1, vmax=1)
        cmap = cm.get_cmap('coolwarm')
        edge_colors = [cmap(norm(w)) for w in edge_weights]
        for (u, v), style, color, width in zip(G.edges(), edge_styles, edge_colors, edge_widths):
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color,
                                   style=style, width=width)
        # nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=2, alpha=0.6)

        nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_label("Correlation Strength")
        plt.savefig("snha_network_red.pdf", format='pdf', bbox_inches="tight")

        with (PdfPages('snha_network.pdf') as pdf):

            for a in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                s = Snha(data=predictors)
                s.new_graph()
                s.comp_corr(method='spearman')
                s.st_nich_alg(alpha = a)

                fig, ax = plt.subplots(1, 1, figsize=(10, 10))

                s.plot_graph(ax=ax, vs=0.2)

                ax.set_title('Predicted Graph, threshold: ' + str(a))
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)




    elif option == 'wild data':
            acc = {}

            for name in ['Caro W', 'Caro S', 'Katti']:
                overall_length_sum = 0
                overall_min = None
                overall_max = None
                acc_name = []

                for filename in sim_func.IMPORT_PARAMETERS[name]['filepath_acc']:
                    print(filename)
                    acc_1 = im_raw.import_acc_data(filename)

                    if 'timestamp' in acc_1.columns:
                        acc_1 = sim_func.timestamp_to_datetime(acc_1)
                        acc_1['tag_logger'] = acc_1['tag-local-identifier']
                    else:
                        logger = re.search(r"\d{4}", filename)[0]
                        acc_1['tag_logger'] = logger
                    acc_1['datetime'] = pd.to_datetime(acc_1['datetime'])

                    max_dt = acc_1['datetime'].max()
                    min_dt = acc_1['datetime'].min()
                    number = acc_1.shape[0]

                    print({'max': max_dt, 'min': min_dt, 'length': number, 'logger': acc_1['tag_logger'].unique(),
                           'filename': filename})

                    acc_name.append({
                        'max': max_dt,
                        'min': min_dt,
                        'length': number,
                        'logger': acc_1['tag_logger'].unique(),
                        'filename': filename
                    })

                    if overall_min is None or min_dt < overall_min:
                        overall_min = min_dt
                    if overall_max is None or max_dt > overall_max:
                        overall_max = max_dt
                    overall_length_sum += number

                acc_name.append({
                    'max': overall_max,
                    'min': overall_min,
                    'length': overall_length_sum,
                    'logger': [],
                    'filename': 'Overall'
                })

                acc[name] = acc_name

            print(acc)

            with open("data_statistics_wilddata.csv", "w", newline="") as f:
                writer = csv.writer(f)

                writer.writerow(["Name", "Max", "Min", "Length", "Logger", "Filename"])

                for name, records in acc.items():
                    for record in records:
                        logger_str = ", ".join(map(str, record['logger'])) if len(record['logger']) > 0 else ""
                        writer.writerow([
                            name,
                            record['max'],
                            record['min'],
                            record['length'],
                            logger_str,
                            record['filename']
                        ])
