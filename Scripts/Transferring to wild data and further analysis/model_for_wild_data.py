#!/usr/bin/python3
"""
Filename: model_for_wild_data.py
Author: Eva Reinhardt
Date: 2024-12-09
Version: 1.0
Description: training and evaluating the models for wild data (simple and layered model). Outputs
include hourly behavior contingents. gps csv files that map a predicted behavior to gps points is also
included.

Functions in this file:

including_moving_window(): including the moving window in the prediction process

adding_gps_column(): adding additional columns with gps data

"""


import re
import os
import joblib

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn import svm

import xgboost as xgb

from raccoon_acc_setup import predictor_calculation as pred_cal
from raccoon_acc_setup import variables_simplefunctions as sim_func
from raccoon_acc_setup import machine_learning_functions as mlf
from raccoon_acc_setup import importing_raw_data as im_raw
from raccoon_acc_setup import plot_functions as plt_func
from raccoon_acc_setup import gui_functions as guif

mapping = {'resting': 0, 'exploring': 1, 'walking': 2, 'climbing': 3, 'high energy': 4, 'unknown': 5}
inverted_mapping = {v: k for k, v in mapping.items()}
color_mapping = sim_func.COLOR_MAPPING_HTML
class_order = ['resting', 'exploring', 'walking', 'climbing', 'high energy']

time_of_day_info = np.array([0, 1, 2, 3])

model1 = xgb.XGBClassifier(colsample_bytree=1.0, gamma=0.5, learning_rate=0.2, max_depth=4,
                           min_child_weight=2, n_estimators=20, subsample=1.0)
model2 = xgb.XGBClassifier(colsample_bytree=1.0, gamma=1, learning_rate=0.2, max_depth=4,
                           min_child_weight=10, n_estimators=20, subsample=0.7)

model_comp = svm.SVC(probability=True, kernel="rbf", C=80, gamma="scale")

time_of_day_colors = {
    0: 'midnightblue',
    1: 'orange',
    2: 'goldenrod',
    3: 'orange',
}

filepaths_peter = [
    sim_func.IMPORT_PARAMETERS['Peter']['filepath_pred']]
filepaths_domi = [
    sim_func.IMPORT_PARAMETERS['Dominique']['filepath_pred']]

filepaths = [filepaths_peter, filepaths_domi]

filepaths_pred_katti = [[
    '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild - Katti Meyer/ACC_only_3/pred_tag5123red.txt',
    '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild - Katti Meyer/ACC_only_3/pred_tag5124red.txt',
    '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild - Katti Meyer/ACC_only_3/pred_tag5125red.txt',
    '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild - Katti Meyer/ACC_only_3/pred_tag5126red.txt']]
filepaths_pred_caros = [[
    '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild BB - Caro Scholz-Biomove/pred_1628_red_incl_gps.csv',
    '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild BB - Caro Scholz-Biomove/pred_1630_red_incl_gps.csv',
    '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild BB - Caro Scholz-Biomove/pred_1631_red_incl_gps.csv'
]]
filepaths_pred_carow = [[
    '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild Berlin - Caro Weh-IZW/pred_acc_all_20190402red.csv'
]]
filepaths_wild = [filepaths_pred_katti, filepaths_pred_carow, filepaths_pred_caros]
#filepaths_wild = [filepaths_pred_caros]

fs_katti = 16.67
filepaths_acc_katti = sim_func.IMPORT_PARAMETERS['Katti']['filepath_acc']
fs_caros = 18.74
filepaths_acc_caros = [
    '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild BB - Caro Scholz-Biomove/1630_ACC_movebankred.csv']
# sim_func.IMPORT_PARAMETERS['Caro S']['filepath_acc']
fs_carow = 33.3
filepaths_acc_carow = sim_func.IMPORT_PARAMETERS['Caro W']['filepath_acc']


def including_moving_window(filepath_func: str, m1: BaseEstimator, m2: BaseEstimator, m_comp: BaseEstimator) -> (
        tuple)[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    running the models with moving window
    @param filepath_func: filepath to the used acc file
    @param m1: first model for layered structure
    @param m2: second model for layered structure
    @param m_comp: simple model without layers
    @return: dataframes containing of predictions and respective probabilities for:
                    1. layered model with moving window
                    2. simple model with moving window
                    3. layered model without moving window
                    4. simple model without moving window
    """
    if 'Katti' in filepath_func:
        fs = fs_katti
    elif 'Caro S' in filepath_func:
        fs = fs_caros
    else:
        fs = fs_carow

    y_prob_3_all_func = []
    y_prob_comp_all_func = []
    y_prob_comp_wo_mw = []
    y_prob_3_wo_mw_func = []

    data = im_raw.import_acc_data(filepath_func)

    if 'datetime' not in data.columns:
        data = sim_func.timestamp_to_datetime(data)

    data_wo_mw = []
    pred_wo_mw = []

    for i in range(data.shape[0]):
        print(data.iloc[[i]])
        burst = im_raw.split_burst(data.iloc[[i]])
        bursts = mlf.moving_window(burst, fs=fs)
        if bursts:
            data_wo_mw.append(bursts[0])

            group = [
                {**dict(zip(sim_func.COLUMNS_PREDICTORS, pred_cal.calculate_features(
                    b['X'], b['Y'], b['Z'], b['XZ'], b['datetime'].iloc[0], fs)))}
                for b in bursts
            ]
            group_df = pd.DataFrame(group)
            pred_wo_mw.append(group_df.iloc[0])

            y_prob_3_bursts, y_prob_comp_bursts = mlf.model_predictions(group_df, m1, m2, m_comp, label_encoder)

            if 'climbing' in y_prob_3_bursts.columns:
                max_values_rows_3 = y_prob_3_bursts[['climbing', 'walking', 'exploring']].max(axis=1)
            else:
                max_values_rows_3 = y_prob_3_bursts[['resting', 'intermediate energy', 'high energy']].max(axis=1)
            max_values_rows_comp = y_prob_comp_bursts[
                ['resting', 'climbing', 'walking', 'exploring', 'high energy']].max(axis=1)

            max_idx_3 = max_values_rows_3.idxmax()
            max_idx_comp = max_values_rows_comp.idxmax()

            y_prob_3_all_func.append(y_prob_3_bursts.iloc[max_idx_3].to_dict())
            y_prob_comp_all_func.append(y_prob_comp_bursts.iloc[max_idx_comp].to_dict())
            y_prob_3_wo_mw_func.append(y_prob_3_bursts.iloc[0].to_dict())
            y_prob_comp_wo_mw.append(y_prob_comp_bursts.iloc[0])

    y_prob_3_all_func = pd.DataFrame(y_prob_3_all_func)
    y_prob_comp_all_func = pd.DataFrame(y_prob_comp_all_func)

    y_prob_3_wo_mw_func = pd.DataFrame(y_prob_3_wo_mw_func)
    y_prob_comp_wo_mw = pd.DataFrame(y_prob_comp_wo_mw)

    return y_prob_3_all_func, y_prob_comp_all_func, y_prob_3_wo_mw_func, y_prob_comp_wo_mw


def adding_gps_column(filepath_func: str, y_prob: pd.DataFrame, logger_func: str) -> pd.DataFrame:
    """
    function to include gps in the dataframe
    @param filepath_func: filepath to the acc file, used to check for fitting gps files in the same directory
    @param y_prob: dataframe that gps should be added to, has to contain 'datetime' column
    @param logger_func: logger number to filter the gps dataframe
    @return: dataframe including the new gps data
    """
    filepath_gps_func = filepath_func.split('/')

    matching_files = []
    matching_files_gps = []

    y_prob['datetime'] = pd.to_datetime(y_prob['datetime'])

    if 'Weh' in filepath_func:
        filepath_gps_func = '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild Berlin - Caro Weh-IZW/GPS_all_20190410.csv'

    else:
        for root, _, files in os.walk('/'.join(filepath_gps_func[0:-1])):
            for file in files:
                if 'gps' in file and logger in file and 'pred' not in file:
                    matching_files.append(os.path.join(root, file))
                if 'gps' in file:
                    matching_files_gps.append(os.path.join(root, file))

        if len(matching_files) == 1:
            filepath_gps_func = matching_files[0]
        elif len(matching_files) == 0:
            filepath_gps_func = guif.choose_option(matching_files_gps, 'Which file should be used?')
        else:
            filepath_gps_func = guif.choose_option(matching_files, 'Which file should be used?')
    gps_func = pd.read_table(filepath_gps_func, sep=',')
    if 'tag-local-identifier' in gps_func.columns:
        gps_func = gps_func[gps_func['tag-local-identifier'] == int(logger_func)]
    if 'timestamp' in gps_func.columns:
        gps_func['datetime'] = pd.to_datetime(gps_func['timestamp'])
    elif 'datetime' in gps_func.columns:
        gps_func['datetime'] = pd.to_datetime(gps_func['datetime'])
    else:
        gps_func['datetime'] = pd.to_datetime(gps_func['date'] + ' ' + gps_func['time'], dayfirst=True)
    gps_func = gps_func[
        ['datetime', 'location-long', 'location-lat', 'ground-speed', 'heading', 'height-above-ellipsoid']]
    print(gps_func.columns)
    y_final = y_prob.merge(gps_func, on='datetime', how='outer')
    return y_final


if __name__ == "__main__":

    # output_pdf_path = filedialog.asksaveasfilename(title="Save as")
    pred = pred_cal.create_pred_complete(filepaths, reduced_features=False)
    pred = im_raw.convert_beh(pred, 'generalization', 'generalization3')
    pred = im_raw.convert_beh(pred, 'translation')
    predictors, labels = mlf.splitting_pred(pred, mapping=mapping)

    labels_3, labels_int, pred_int = mlf.preparing_datasets_layered(predictors, labels)
    labels['behavior_generalization'] = labels['behavior_generalization'].map(inverted_mapping)
    labels_3['behavior_generalization'] = labels_3['behavior_generalization'].map(sim_func.MAPPING_3)
    labels_int['behavior_generalization'] = labels_int['behavior_generalization'].map(sim_func.MAPPING_INT)
    label_encoder = LabelEncoder()

    labels_encoded = label_encoder.fit_transform(labels['behavior_generalization'])

    model1.fit(predictors, labels_3['behavior_generalization'])
    model2.fit(pred_int[[f for f in sim_func.REDUCED_FEATURES if f in pred_int.columns]],
               labels_int['behavior_generalization'])
    model_comp.fit(predictors, labels_encoded)

    joblib.dump(model1, 'xgboost_model1.joblib')
    joblib.dump(model2, 'xgboost_model2.joblib')
    joblib.dump(model_comp, 'svm_model_comp.joblib')
    joblib.dump(label_encoder, 'label_encoder.joblib')

    option_mw = guif.choose_option(['Yes', 'No'], title='Do you want to use moving window?')
    if option_mw == 'No':
        for filepaths_name in filepaths_wild:
            for filepath in filepaths_name[0]:
                pred_wild = pred_cal.create_pred_complete([[filepath]], reduced_features=True)
                pattern = r"\d{4}"
                matches = re.findall(pattern, filepath)

                if 'Katti' in filepath:
                    path_general = '/home/eva/Schreibtisch/Master/NeuerVersuch/wild_data_katti_'
                    name = 'Katti'
                elif 'Scholz' in filepath:
                    path_general = '/home/eva/Schreibtisch/Master/NeuerVersuch/wild_data_caros_'
                    name = 'CaroS'
                elif 'Weh' in filepath:
                    path_general = '/home/eva/Schreibtisch/Master/NeuerVersuch/wild_data_carow_'
                    name = 'CaroW'

                else:
                    path_general = ''

                path_general = path_general + str(matches)

                y_prob_3_all, y_prob_comp_all = mlf.model_predictions(pred_wild, model1, model2, model_comp,
                                                                      label_encoder)

                # y_3_final = adding_gps_column(filepath, y_prob_3_all, str(matches[0]))
                #
                # y_3_final = y_3_final[[col for col in y_3_final.columns if '_x' not in col and '_y' not in col]]
                # y_3_final_gps = y_3_final.dropna()
                #
                # y_3_final_gps.to_csv(path_general+'only_gps.csv')
                # y_3_final.to_csv(path_general+'incl_gps.csv')

                plt_func.output_hourly_contingents(y_prob_3_all, path_general, title='Layered Model')
                plt_func.output_hourly_contingents(y_prob_comp_all, path_general, title='Simple Model')

                y_prob_final = adding_gps_column(filepath,
                                                 y_prob_3_all, matches[0])
                y_prob_final.to_csv(path_general + name + matches[0] + '_gps_behavior.csv', index=False)
            #

    else:
        # name = guif.choose_option(['Caro W', 'Caro S', 'Katti'])
        name = ''
        if name == 'Katti':
            filepaths_acc = filepaths_acc_katti[4:5]
        elif name == 'Caro S':
            filepaths_acc = filepaths_acc_caros
        else:
            filepaths_acc = filepaths_acc_carow

        for filepath in filepaths_acc:
            pattern = r"\d{4}"
            matches = re.findall(pattern, filepath)
            logger = matches[0]
            y_prob_3_all, y_prob_comp_all, y_prob_3_wo_mw, y_prob_wo_mw = including_moving_window(filepath, model1,
                                                                                                  model2, model_comp)

            path = '/home/eva/Schreibtisch/Master/NeuerVersuch/wild_data_' + name
            plt_func.output_hourly_contingents(y_prob_3_all, path, title=str(logger) + 'Layered Model')
            plt_func.output_hourly_contingents(y_prob_comp_all, path, title=str(logger) + 'Simple Model')
            plt_func.output_hourly_contingents(y_prob_3_wo_mw, path, title=str(logger) + 'Layered Model_wo_mw')
            plt_func.output_hourly_contingents(y_prob_wo_mw, path, title=str(logger) + 'Simple Model_wo_mw')
            y_prob_final = adding_gps_column(filepath,
                                             y_prob_3_all, logger)
            y_prob_final.to_csv(sim_func.IMPORT_PATH_CLASS + name + str(logger) + '_predictions_mw_layered.csv',
                                index=False)

            y_prob_final_comp = adding_gps_column(filepath,
                                                  y_prob_comp_all, logger)
            y_prob_final_comp.to_csv(sim_func.IMPORT_PATH_CLASS + name + str(logger) + '_predictions_mw_simple.csv',
                                     index=False)

            y_prob_final_wo_mw_l = adding_gps_column(filepath,
                                                     y_prob_3_wo_mw, logger)
            y_prob_final_wo_mw_l.to_csv(
                sim_func.IMPORT_PATH_CLASS + name + str(logger) + '_predictions_wo_mw_layered.csv', index=False)

            y_prob_final_wo_mw_s = adding_gps_column(filepath,
                                                     y_prob_wo_mw, logger)
            y_prob_final_wo_mw_s.to_csv(
                sim_func.IMPORT_PATH_CLASS + name + str(logger) + '_predictions_wo_mw_simple.csv', index=False)
