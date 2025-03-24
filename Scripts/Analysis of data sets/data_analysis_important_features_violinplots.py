"""
Filename: data_analysis_important_features_violinplots.py
Author: Eva Reinhardt
Date: 2024-12-09
Version: 1.0
Description: This file is used to create violinplots of chosen features by behaviour. Additionally, the analysis
of using only the first part of a burst or the first and last portion of desired burst length can be conducted
when choosing the option "Comparison moving window".
The thirs option 'Testing generalization3 climbing and walking' helps to compare behaviours labelled as "climbing"
to behaviours labelled as "walking".



"""

import pandas as pd
import numpy as np

from tkinter import filedialog

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches

from sklearn.cluster import KMeans

from raccoon_acc_setup import variables_simplefunctions as sim_func
from raccoon_acc_setup import importing_raw_data as im_raw
from raccoon_acc_setup import gui_functions as guif
from raccoon_acc_setup import plot_functions as plt_func

if __name__ == '__main__':

    plt.rcParams.update({'font.size': 16})

    option1 = guif.choose_option(
        options=['Normal violinplots', 'Comparison moving window', 'Testing generalization3 climbing and walking'])
    if option1 == 'Normal violinplots':
        option = guif.choose_option(options=['Choose files myself', 'Show all files'])
        if option == 'Choose files myself':
            filepaths = [guif.open_file_dialog('Select a file or files - predictors')]
        else:
            filepaths_peter = [sim_func.IMPORT_PARAMETERS['Peter']['filepath_pred']]

            filepaths_domi = [sim_func.IMPORT_PARAMETERS['Dominique']['filepath_pred']]
            filepaths = [filepaths_peter, filepaths_domi]
        options = ['XZmax - XZmin', 'XZvar', 'Yvar', 'Ymax - Ymin', 'Odba', 'fft_max', 'fft_wmean', 'fft_std']
        options_col = '|'.join(options) + '|Ndyn|behavior|fl'

    elif option1 == 'Comparison moving window':
        option2 = guif.choose_option(options=['Peter', 'Dominique'])

        if option2 == 'Peter':
            filepaths = [['/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/predictors/pred_all_first.csv',
                             sim_func.IMPORT_PARAMETERS['Peter']['filepath_pred']]]
        else:

            filepaths = [['/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/predictors_first54.csv',
                             '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/predictors_firstlast54.csv'
                             ]]
        # options = rac.choose_multiple_options(options = ['Xdyn', 'Ydyn', 'Zdyn', 'Ndyn', 'Nvar', 'Odba', 'Roll', 'Pitch', 'Yaw','fft_max', 'fft_wmean'])
        # options = ['XZmax - XZmin', 'XZvar', 'Yvar', 'Ymax - Ymin', 'Odba', 'fft_max', 'fft_wmean', 'fft_std']
        options = []
        options_col = '|'.join(options) + '|Ndyn|behavior|fl'

    else:
        filepaths_peter = [
            sim_func.IMPORT_PARAMETERS['Peter']['filepath_pred']]
        filepaths_domi = [
            sim_func.IMPORT_PARAMETERS['Dominique']['filepath_pred']]
        filepaths = [filepaths_peter, filepaths_domi]
        options = ['Ymean', 'XZmean', 'Pitch']
        options_col = 'Ymean|XZmean|Pitch|behavior'

    output_pdf_path = filedialog.asksaveasfilename(title="Save as")
    with PdfPages(output_pdf_path) as pdf:
        pred_com = pd.DataFrame()
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

                if option1 == 'Comparison moving window':
                    if 'firstlast' in filepath:
                        pred_1['fl'] = 'firstlast'
                    else:
                        pred_1['fl'] = 'first'
                    pred_11 = im_raw.convert_beh(pred_1, 'translation')

                pred = pd.concat([pred, pred_1], ignore_index=True)


            if option1 == 'Comparison moving window':
                first = pred.groupby('fl').get_group('first').drop('fl', axis=1)
                firstlast = pred.groupby('fl').get_group('firstlast').drop('fl', axis=1)
                merged = pd.merge(first, firstlast, how='inner')
            elif option1 == 'Normal violinplots':
                pred = im_raw.convert_beh(pred, 'translation')
            elif option1 == 'Testing generalization3 climbing and walking':
                pred = im_raw.convert_beh(pred, 'generalization', 'generalization3')
                pred = im_raw.convert_beh(pred, 'translation')
                pred = pred[pred['behavior_generalization'].isin(['climbing', 'walking'])]
                pred = pred.sort_values('behavior_generalization')
                # pred['behavior'] = pred['behavior_generalization']
                my_pal = pred.drop_duplicates(subset=['behavior']).set_index('behavior')['behavior_generalization'].map(
                    {'climbing': 'g', 'walking': 'y'}).to_dict()

            pred = pred.filter(regex=options_col)
            pred = pred[~(pred['behavior'] == 'schütteln')]
            pred_com = pd.concat([pred, pred_com], ignore_index=True)

            if 'Ndyn' in pred.columns:
                # Calculate and plot the mean/median
                medians = pred.groupby('behavior')['Ndyn'].median().reset_index()
                medians.columns = ['behavior', 'Ndyn_median']

                medians = medians.sort_values(by='Ndyn_median')
                pred['behavior'] = pd.Categorical(pred['behavior'], categories=medians['behavior'], ordered=True)
                pred = pred.sort_values('behavior')

                # k-means clustering for threshold determination
                median_values = medians['Ndyn_median'].values.reshape(-1, 1)
                kmeans = KMeans(n_clusters=3, random_state=0).fit(median_values)
                medians['activity_level'] = kmeans.labels_

                cluster_centers = kmeans.cluster_centers_.flatten()
                active_cluster = list(cluster_centers).index(max(cluster_centers))
                intermediate_cluster = list(cluster_centers).index(np.median(cluster_centers))
                calm_cluster = list(cluster_centers).index(min(cluster_centers))

                active_medians = medians[medians['activity_level'] == active_cluster]['Ndyn_median']
                intermediate_medians = medians[medians['activity_level'] == intermediate_cluster]['Ndyn_median']
                calm_medians = medians[medians['activity_level'] == calm_cluster]['Ndyn_median']
                threshold_1 = (active_medians.min() + intermediate_medians.max()) / 2
                threshold_2 = (calm_medians.max() + intermediate_medians.min()) / 2

                medians['activity_level'] = medians['activity_level'].replace(
                    {intermediate_cluster: 'g', active_cluster: 'y', calm_cluster: 'b'})
                my_pal = dict(zip(medians['behavior'], medians['activity_level']))

                calm_behaviors = medians.groupby('activity_level').get_group('b')['behavior']

                pred = pred[~((pred['behavior'].isin(calm_behaviors)) & (pred['Ndyn'] > threshold_1))]

                data = [group['Ndyn'].values for name, group in pred.groupby('behavior')]

                plt.figure(figsize=(12, 10))

                if option1 == 'Normal violinplots':
                    violin_parts = plt.violinplot(data, showmedians=True)
                    plt.xticks(ticks=np.arange(1, len(data) + 1), labels=pred['behavior'].unique(), rotation=90)
                elif option1 == 'Comparison moving window':
                    # splitting of dataset based on the 'fl' column
                    data_first = pred[pred['fl'] == 'first']
                    data_firstlast = pred[pred['fl'] == 'firstlast']

                    data_left = [group['Ndyn'].values for name, group in data_first.groupby('behavior')]
                    data_right = [group['Ndyn'].values for name, group in data_firstlast.groupby('behavior')]

                    behaviors = pred['behavior'].unique()

                    for i in range(len(behaviors)):
                        parts_left = plt.violinplot(data_left[i], positions=[i], points=60, widths=0.9,
                                                    showextrema=True,
                                                    showmedians=True, bw_method=0.5, side='low')
                        for pc in parts_left['bodies']:
                            pc.set_facecolor('cadetblue')
                            pc.set_edgecolor('black')
                            pc.set_alpha(0.7)
                            # pc.set_offsets([i - 0.2, 0])

                        parts_left['cmedians'].set_color('teal')
                        parts_left['cmins'].set_color('teal')
                        parts_left['cmaxes'].set_color('teal')
                        parts_left['cbars'].set_color('black')

                        parts_right = plt.violinplot(data_right[i], positions=[i], points=60, widths=0.9,
                                                     showextrema=True, showmedians=True,
                                                     bw_method=0.5, side='high')
                        for pc in parts_right['bodies']:
                            pc.set_facecolor('goldenrod')
                            pc.set_edgecolor('black')
                            pc.set_alpha(0.7)
                        parts_right['cmedians'].set_color('darkgoldenrod')
                        parts_right['cmins'].set_color('darkgoldenrod')
                        parts_right['cmaxes'].set_color('darkgoldenrod')
                        parts_right['cbars'].set_color('black')
                    violin_parts = [parts_left, parts_right]
                    plt.xticks(ticks=np.arange(len(behaviors)), labels=behaviors, rotation=90)

                    left_patch = mpatches.Patch(color='cadetblue', label='First Group')
                    right_patch = mpatches.Patch(color='goldenrod', label='FirstLast Group')

                    plt.legend(handles=[left_patch, right_patch], loc='upper right', title='Groups')

                plt.axhline(threshold_1, color='green', linestyle='--', label='Threshold')
                plt.axhline(threshold_2, color='green', linestyle='--', label='Threshold')

                # title is the filepath for better understanding and the coverage values
                plt.title(f'{name} Violinplot of the VDBA, thresholds: {threshold_1.round(3)}; {threshold_2.round(3)}')

                plt.ylim(0, 3)

                plt.grid()

                plt.tight_layout()
                pdf.savefig()
                plt.close()

            plt_func.plotting_vps(options, pred, name, my_pal, pdf)

        if option1 == 'Testing generalization3 climbing and walking':
            pred_com = pred_com.sort_values('behavior_generalization')
            my_pal = pred_com.drop_duplicates(subset=['behavior']).set_index('behavior')['behavior_generalization'].map(
                {'climbing': 'g', 'walking': 'y'}).to_dict()
        plt_func.plotting_vps(options, pred_com, 'Both', my_pal, pdf)
