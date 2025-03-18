#!/usr/bin/python3
"""
Filename: predictor_calculation.py
Author: Eva Reinhardt
Date: 2024-12-09
Version: 1.0
Description: collection of functions used for predictor calculation

functions:
calculate_features(): calculating the features from the raw acc data

calculate_pred(): creating the predictor dataframe

create_pred_complete(): loading and combining several pred files
"""
import pandas as pd
import numpy as np

from . import importing_raw_data as im_raw
from . import variables_simplefunctions as sim_func
from . import machine_learning_functions as mlf




def calculate_features(X: pd.Series, Y: pd.Series, Z: pd.Series, XZ: pd.Series, datetime, fs=33.3) -> list:
    """
    function to calculate features from the raw (but converted to g) data
    @param X: x component of burst
    @param Y: y component of burst
    @param Z: z component of burst
    @param XZ: (x² + z²)^(1/2)
    @param datetime: datetime of burst
    @param fs: sampling frequency (default 33.3)
    @return: feature list: [datetime, Xmean, Xvar, Xmin, Xmax, Xmax - Xmin,
                 Ymean, Yvar, Ymin, Ymax, Ymax - Ymin,
                 Zmean, Zvar, Zmin, Zmax, Zmax - Zmin,
                 XZmean, XZvar, XZmin, XZmax, XZmax - XZmin,
                 Xdyn, Ydyn, Zdyn, XZdyn,
                 Ndyn, Nvar,
                 Odba,
                 Pitch, #Yaw,
                 fft_base, fft_max, fft_wmean, fft_std]
    """

    Xmean = np.mean(X)
    Xvar = np.var(X)
    Xmin = np.min(X)
    Xmax = np.max(X)
    Ymean = np.mean(Y)
    Yvar = np.var(Y)
    Ymin = np.min(Y)
    Ymax = np.max(Y)
    Zmean = np.mean(Z)
    Zvar = np.var(Z)
    Zmin = np.min(Z)
    Zmax = np.max(Z)
    XZmean = np.mean(XZ)
    XZvar = np.var(XZ)
    XZmin = np.min(XZ)
    XZmax = np.max(XZ)
    # dynamic data
    Xd = X - Xmean
    Yd = Y - Ymean
    Zd = Z - Zmean
    XZd = XZ - XZmean
    # mean of abs dynamic data
    Xdyn = np.mean(abs(Xd))
    Ydyn = np.mean(abs(Yd))
    Zdyn = np.mean(abs(Zd))
    XZdyn = np.mean(abs(XZd))
    # norm of dynamic data
    Nd = np.linalg.norm([Xd, Yd, Zd], axis=0)
    Ndyn = np.mean(Nd)
    Nvar = np.var(Nd)

    Odba = Xdyn + Ydyn + Zdyn

    Pitch = np.arcsin(Ymean / np.sqrt(XZmean ** 2 + Ymean ** 2))

    fft_data = np.abs(np.fft.rfft(Nd))[1:len(Nd) // 2 + 1]
    fft_max = fft_data[np.argmax(fft_data)]
    frequencies = np.fft.rfftfreq(len(Nd), d=1 / fs)[1:len(Nd) // 2 + 1]
    fft_base = frequencies[np.argmax(fft_data)]
    fft_data = fft_data / fft_max
    tmp = list(range(1, len(fft_data) + 1))
    fft_wmean = np.mean(fft_data / tmp) / np.mean(fft_data)
    fft_std = np.std(fft_data)

    feat_list = [datetime, Xmean, Xvar, Xmin, Xmax, Xmax - Xmin,
                 Ymean, Yvar, Ymin, Ymax, Ymax - Ymin,
                 Zmean, Zvar, Zmin, Zmax, Zmax - Zmin,
                 XZmean, XZvar, XZmin, XZmax, XZmax - XZmin,
                 Xdyn, Ydyn, Zdyn, XZdyn,
                 Ndyn, Nvar,
                 Odba,
                 Pitch,
                 fft_base, fft_max, fft_wmean, fft_std]

    return feat_list

def calculate_pred(data: pd.DataFrame, frequence, mw: bool = False, step: int = None) -> pd.DataFrame:
    """
    Calculate predictors from accelerometer data
    (input dataframe can contain behavior data for one or two inidividuals or none)

    @param data: DataFrame containing the accelerometer data
                    (columns: 'datetime', 'raw_x', 'raw_y', 'raw_z', 'X', 'Y', 'Z',
                    optional columns: behavior (1 or 2 columns))
    @param frequence: frequency of data
    @param mw: moving window option, True or False
    @param step: step for moving window
    @return: DataFrame containing the calculated predictors
                    (columns: 'datetime', 'Xmean', 'Xvar', 'Xmin', 'Xmax', 'Xmax - Xmin',
                        'Ymean', 'Yvar', 'Ymin', 'Ymax', 'Ymax - Ymin',
                        'Zmean', 'Zvar', 'Zmin', 'Zmax', 'Zmax - Zmin',
                        'Xdyn', 'Ydyn', 'Zdyn',
                        'Ndyn', 'Nvar',
                        'Odba',
                        'Roll', 'Pitch', 'Yaw',
                        'fft_max', 'fft_wmean', 'fft_std'
                    optional columns: behavior (1 or 2 columns))
    """
    if not step:
        step = int((frequence* 54/sim_func.GENERAL_FREQ)/2)
    burst_len = int(frequence* 54/sim_func.GENERAL_FREQ)
    columns_predictors = sim_func.COLUMNS_PREDICTORS

    # Determine if behavior columns are present
    if len(data.columns) == 10:
        behavior_columns = list(data.columns[-2:])
    elif len(data.columns) == 9:
        behavior_columns = list(data.columns[-1:])
    else:
        behavior_columns = []

    columns = columns_predictors + behavior_columns

    if 'timestamp' in data.columns:
        data['datetime'] = data['timestamp']

    data_rows = []

    if len(data['datetime'].unique()) == data.shape[0]:
        for i in range(data.shape[0]):
            burst = im_raw.split_burst(data.iloc[[i]])
            bursts = mlf.moving_window(burst, fs=frequence, step=step) if mw else [burst.iloc[0:burst_len]]

            for b in bursts:
                feat_list = calculate_features(b['X'], b['Y'], b['Z'], b['XZ'], b['datetime'].iloc[0], frequence)
                row = dict(zip(columns_predictors, feat_list))

                if len(b.columns) == 10:
                    row.update({behavior_columns[0]: b.iloc[1, -2], behavior_columns[1]: b.iloc[1, -1]})
                elif len(b.columns) == 9:
                    row.update({behavior_columns[0]: b.iloc[1, -1]})

                data_rows.append(row)
    else:
        grouped = data.groupby(['datetime'])
        for datetime, group in grouped:
            burst = group
            bursts = mlf.moving_window(burst, step) if mw else [burst]
            for b in bursts:
                X, Y, Z, XZ = b['X'][0:burst_len], b['Y'][0:burst_len], b['Z'][0:burst_len], b['XZ'][0:burst_len]
                feat_list = calculate_features(X, Y, Z, XZ, datetime, frequence)
                row = dict(zip(columns_predictors, feat_list))

                if len(data.columns) == 10:
                    row.update({behavior_columns[0]: b.iloc[1, -2], behavior_columns[1]: b.iloc[1, -1]})
                elif len(data.columns) == 9:
                    row.update({behavior_columns[0]: b.iloc[1, -1]})

                data_rows.append(row)

    pred = pd.DataFrame(data_rows, columns=columns)
    return pred



def create_pred_complete(filepaths: list, reduced_features: bool = False) -> pd.DataFrame:
    """
    Function to create a dataframe with all instances from Peter and Dominique, not including generalization and translation
    @param filepaths: list of files, to obtain pred from
    @param reduced_features: True if the reduced feature set should be used according to correlations
    @return: Dataframe including all instances from the files
    """

    pred_com = pd.DataFrame()
    for filepaths_temp in filepaths:
        pred = pd.DataFrame()
        for i, filepath in enumerate(filepaths_temp):
            pred_1 = pd.read_csv(filepath)
            if 'Peter' in filepath:
                pred_1 = im_raw.convert_beh(pred_1, 'Peter')
            elif 'Dominique' in filepath:
                if '5032' in filepath:
                    if 'behavior_Ottilie' in pred_1.columns:
                        pred_1 = pred_1.drop(['behavior_Ottilie'], axis=1)
                        pred_1 = pred_1.rename(
                            columns={'behavior_Lisa': 'behavior'})
                elif '5033' in filepath:
                    if 'behavior_Lisa' in pred_1.columns:
                        pred_1 = pred_1.drop(['behavior_Lisa'], axis=1)
                        pred_1 = pred_1.rename(
                            columns={'behavior_Ottilie': 'behavior'})
                pred_1 = im_raw.behavior_combi_domi(pred_1)
            pred = pd.concat([pred, pred_1], ignore_index=True)

        if 'behavior' in pred.columns:
            # preparing the dataframe: fitlering out schütteln, filtering out outliers from Ndyn column if wanted
            pred = pred[~(pred['behavior'] == 'schütteln')]
            pred = pred[~((pred['behavior'] == 'schlafen') & (pred['Ndyn'] > 0.5))]
        pred = pred.dropna(axis=1)
        pred = sim_func.x_z_combination(pred)
        pred_com = pd.concat([pred_com, pred], ignore_index=True)
    if 'Xdyn' in pred_com.columns:
        pred_com = pred_com.drop(["Xdyn", "Zdyn"], axis=1)
    if reduced_features:
        if 'behavior' in pred_com.columns:
            pred_com = pred_com[sim_func.REDUCED_FEATURES + ['behavior']]
        else:
            pred_com = pred_com[sim_func.REDUCED_FEATURES]
    return pred_com


