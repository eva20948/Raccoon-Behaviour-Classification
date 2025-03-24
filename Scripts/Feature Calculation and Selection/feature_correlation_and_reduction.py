#!/usr/bin/python3

"""
Filename: feature_correlation_and_reduction.py
Author: Eva Reinhardt
Date: 2024-02-02
Version: 1.0
Description: displaying correlation matrix, correlation network and snha network of features for feature reduction.
"""

import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import networkx as nx
from snha4py.Snha import Snha

from raccoon_acc_setup import predictor_calculation as pred_cal
from raccoon_acc_setup import importing_raw_data as im_raw
from raccoon_acc_setup import variables_simplefunctions as sim_func

filepaths_peter = [
    sim_func.IMPORT_PARAMETERS['Peter']['filepath_pred']]
filepaths_domi = [
    sim_func.IMPORT_PARAMETERS['Dominique']['filepath_pred']]


filepaths = [filepaths_peter, filepaths_domi]


if __name__ == "__main__":


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
    network_data = network_data[network_data['pred1'] != network_data['pred2']]

    G = nx.from_pandas_edgelist(network_data, 'pred1', 'pred2', edge_attr='value')

    weighted_degrees = {node: sum(G[node][neighbor]['value'] for neighbor in G[node]) for node in G.nodes()}

    sorted_nodes = sorted(weighted_degrees, key=weighted_degrees.get, reverse=True)

    strong_G = G.edge_subgraph([(u, v) for u, v, d in G.edges(data=True) if d['value'] > 0.8]).copy()

    components = list(nx.connected_components(strong_G))

    num_components = len(components)
    component_colors = plt.cm.Greens(np.linspace(0.1, 0.7, num_components))
    node_color_map = {}

    for i, component in enumerate(components):
        for node in component:
            node_color_map[node] = component_colors[i]

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
            edge_widths.append(2 + 5 * (d['weight'] - 0.8))
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


