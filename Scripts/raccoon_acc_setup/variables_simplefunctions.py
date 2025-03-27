#!/usr/bin/python3
"""
Filename: variables_simplefunctions.py
Author: Eva Reinhardt
Date: 2024-12-09
Version: 1.0
Description: This file contains variables important for the whole project as well as
simple functions for converting and adding columns, datetime, etc.

functions:
combine_date_time(): date and time column into single datetime column

x_z_combination(): only using XZ columns and not X and Z columns

timestamp_to_datetime(): converting timestamp column to datetime column

remove_outliers(): removing outliers

adding_time_to_df(): adding month, hour, date columns to dataframe

check_or_input_path(): checks if path exists, if not asks for manually entered path.
"""
import pandas as pd
import numpy as np
from scipy import stats
import os

IMPORT_PATH_GENERAL = '/media/eva/eva-reinhar/your folders/01 raw data/'
IMPORT_PATH_CLASS = '/media/eva/eva-reinhar/your folders/05 intermediate results/wild data classification/'
EXPORT_PATH_HTML = '/media/eva/eva-reinhar/your folders/05 intermediate results/maps/'

IMPORT_PARAMETERS = {'Peter': {
    'filepath_pred': IMPORT_PATH_GENERAL + 'Labeldaten Waschbär Bsc Peter Geiger-IZW/predictors/pred_all_firstlast.csv',
    'filepath_acc': [
        '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/acc data/Carlo/tag5334_acc.txt',
        '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/acc data/Emma/tag5033_acc.txt',
        '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/acc data/Wilbert/1. Daten/tag5140_acc.txt',
        '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/acc data/Wilbert/2. Daten/tag5140_acc.txt',
        '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/acc data/Wilbert/3. Daten/tag7073_acc.txt'],
    'filepath_beh': [
        '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/behavior/tag 5334 C/Observations/alle.csv',
        '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/behavior/tag 5033 E/Observations/alle.csv',
        '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/behavior/tag W1/tag5140/alle.csv',
        '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/behavior/tag W1/tag5140/alle.csv',
        '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/behavior/tag W1/tag7073/alle.csv'],
    'filepath_acc_beh': [[[
                              '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/acc data/Carlo/tag5334_acc.txt'],
                          [
                              '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/behavior/tag 5334 C/Observations/alle.csv']],
                         [[
                              '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/acc data/Emma/tag5033_acc.txt'],
                          [
                              '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/behavior/tag 5033 E/Observations/alle.csv']],
                         [[
                              '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/acc data/Wilbert/1. Daten/tag5140_acc.txt'],
                          [
                              '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/behavior/tag W1/tag5140/alle.csv']],
                         [[
                              '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/acc data/Wilbert/2. Daten/tag5140_acc.txt'],
                          [
                              '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/behavior/tag W1/tag5140/alle.csv']],
                         [[
                              '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/acc data/Wilbert/3. Daten/tag7073_acc.txt'],
                          [
                              '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/behavior/tag W1/tag7073/alle.csv']]],
    'path': IMPORT_PATH_GENERAL + '/Labeldaten Waschbär Bsc Peter Geiger-IZW/',
    'fs': 33.3,
},
    'Dominique': {
        'filepath_pred': IMPORT_PATH_GENERAL + 'Labeldaten_Waschbär Msc Dominique-IZW/pred_all_first.csv',
        'filepath_acc_beh': [[[
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/1tag5032_acc.txt',
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/2tag5032_acc.txt',
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/3tag5032_acc.txt',
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/4tag5032_acc.txt',
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/5tag5032_acc.txt',
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/6tag5032_acc.txt',
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/7tag5032_acc.txt',
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/8tag5032_acc.txt',
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/9tag5032_acc.txt',
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/10tag5032_acc.txt',
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/11tag5032_acc.txt',
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/12tag5032_acc.txt',
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/13tag5032_acc.txt'],
                              [
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Beobachtungsdaten/Verhaltensweisen_concatenated.csv']],
                             [[
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/1tag5033_acc.txt',
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/2tag5033_acc.txt',
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/3tag5033_acc.txt',
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/4tag5033_acc.txt',
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/5tag5033_acc.txt',
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/6tag5033_acc.txt',
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/7tag5033_acc.txt',
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/8tag5033_acc.txt',
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/9tag5033_acc.txt',
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/10tag5033_acc.txt',
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/11tag5033_acc.txt',
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/12tag5033_acc.txt',
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Daten/13tag5033_acc.txt'],
                              [
                                  '/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Beobachtungsdaten/Verhaltensweisen_concatenated.csv']]],
        'path': IMPORT_PATH_GENERAL + 'Labeldaten_Waschbär Msc Dominique-IZW/',
        'fs': 33.3,
    },
    'Katti': {'filepath_pred': [
        IMPORT_PATH_GENERAL + 'Waschbär wild - Katti Meyer/ACC_only_3/pred_tag5126red.txt',
        IMPORT_PATH_GENERAL + 'Waschbär wild - Katti Meyer/ACC_only_3/pred_tag5125red.txt',
        IMPORT_PATH_GENERAL + 'Waschbär wild - Katti Meyer/ACC_only_3/pred_tag5124red.txt',
        IMPORT_PATH_GENERAL + 'Waschbär wild - Katti Meyer/ACC_only_3/pred_tag5123red.txt'
    ],
        'filepath_acc': [
            IMPORT_PATH_GENERAL + 'Waschbär wild - Katti Meyer/ACC_only_3/tag5126.txt',
            IMPORT_PATH_GENERAL + 'Waschbär wild - Katti Meyer/ACC_only_3/tag5124.txt',
            IMPORT_PATH_GENERAL + 'Waschbär wild - Katti Meyer/ACC_only_3/tag5130.txt',
            IMPORT_PATH_GENERAL + 'Waschbär wild - Katti Meyer/ACC_only_3/tag5132.txt',
            IMPORT_PATH_GENERAL + 'Waschbär wild - Katti Meyer/ACC_only_3/tag5125.txt',
            IMPORT_PATH_GENERAL + 'Waschbär wild - Katti Meyer/ACC_only_3/tag5128.txt',
            IMPORT_PATH_GENERAL + 'Waschbär wild - Katti Meyer/ACC_only_3/tag5123.txt',
            IMPORT_PATH_GENERAL + 'Waschbär wild - Katti Meyer/ACC_only_3/tag5129.txt',
            IMPORT_PATH_GENERAL + 'Waschbär wild - Katti Meyer/ACC_only_3/tag5336.txt',
            IMPORT_PATH_GENERAL + 'Waschbär wild - Katti Meyer/ACC_only_3/tag5131.txt'
        ],
        'fs': 16.67},
    'Caro W': {
        'filepath_pred': IMPORT_PATH_GENERAL + 'Waschbär wild Berlin - Caro Weh-IZW/pred_Raccoons of Berlin_ACC_20190402red.csv',
        'filepath_acc':  # [IMPORT_PATH_GENERAL+'Waschbär wild Berlin - Caro Weh-IZW/Raccoons of Berlin_ACC_20190402.csv'],
            [
                IMPORT_PATH_GENERAL + 'Waschbär wild Berlin - Caro Weh-IZW/acc_5607.csv',
                IMPORT_PATH_GENERAL + 'Waschbär wild Berlin - Caro Weh-IZW/acc_5606.csv',
                IMPORT_PATH_GENERAL + 'Waschbär wild Berlin - Caro Weh-IZW/acc_5605.csv',
                IMPORT_PATH_GENERAL + 'Waschbär wild Berlin - Caro Weh-IZW/acc_5604.csv',
                IMPORT_PATH_GENERAL + 'Waschbär wild Berlin - Caro Weh-IZW/acc_5603.csv',
                IMPORT_PATH_GENERAL + 'Waschbär wild Berlin - Caro Weh-IZW/acc_5602.csv',
                IMPORT_PATH_GENERAL + 'Waschbär wild Berlin - Caro Weh-IZW/acc_5599.csv',
                IMPORT_PATH_GENERAL + 'Waschbär wild Berlin - Caro Weh-IZW/acc_5136.csv',
                IMPORT_PATH_GENERAL + 'Waschbär wild Berlin - Caro Weh-IZW/acc_5135.csv',
                IMPORT_PATH_GENERAL + 'Waschbär wild Berlin - Caro Weh-IZW/acc_5134.csv',
                IMPORT_PATH_GENERAL + 'Waschbär wild Berlin - Caro Weh-IZW/acc_5133.csv',
                IMPORT_PATH_GENERAL + 'Waschbär wild Berlin - Caro Weh-IZW/acc_5116.csv',
                IMPORT_PATH_GENERAL + 'Waschbär wild Berlin - Caro Weh-IZW/acc_5115.csv',
                IMPORT_PATH_GENERAL + 'Waschbär wild Berlin - Caro Weh-IZW/acc_5038.csv',
                IMPORT_PATH_GENERAL + 'Waschbär wild Berlin - Caro Weh-IZW/acc_5036.csv',
                IMPORT_PATH_GENERAL + 'Waschbär wild Berlin - Caro Weh-IZW/acc_5035.csv',
                IMPORT_PATH_GENERAL + 'Waschbär wild Berlin - Caro Weh-IZW/acc_5034.csv',
                IMPORT_PATH_GENERAL + 'Waschbär wild Berlin - Caro Weh-IZW/acc_5031.csv'],
        'fs': 33.3},
    'Caro S': {'filepath_pred': [
        IMPORT_PATH_GENERAL + 'Waschbär wild BB - Caro Scholz-Biomove/pred_1628_red_incl_gps.csv',
        IMPORT_PATH_GENERAL + 'Waschbär wild BB - Caro Scholz-Biomove/pred_1630_red_incl_gps.csv',
        IMPORT_PATH_GENERAL + 'Waschbär wild BB - Caro Scholz-Biomove/pred_1631_red_incl_gps.csv'],
        'filepath_acc': [
            IMPORT_PATH_GENERAL + 'Waschbär wild BB - Caro Scholz-Biomove/1630_ACC_movebank.csv',
            IMPORT_PATH_GENERAL + 'Waschbär wild BB - Caro Scholz-Biomove/1628_ACC_movebank.csv',
            IMPORT_PATH_GENERAL + 'Waschbär wild BB - Caro Scholz-Biomove/1631_ACC_movebank.csv',
            IMPORT_PATH_GENERAL + 'Waschbär wild BB - Caro Scholz-Biomove/1633_ACC_movebank.csv',
            IMPORT_PATH_GENERAL + 'Waschbär wild BB - Caro Scholz-Biomove/1634_ACC_movebank.csv',
            IMPORT_PATH_GENERAL + 'Waschbär wild BB - Caro Scholz-Biomove/1636_ACC_movebank.csv',
            IMPORT_PATH_GENERAL + 'Waschbär wild BB - Caro Scholz-Biomove/1637_ACC_movebank.csv',
            IMPORT_PATH_GENERAL + 'Waschbär wild BB - Caro Scholz-Biomove/1638_ACC_movebank.csv'],
        'filepath_gps': [
            IMPORT_PATH_GENERAL + 'Waschbär wild BB - Caro Scholz-Biomove/1630_gps.csv',
            IMPORT_PATH_GENERAL + 'Waschbär wild BB - Caro Scholz-Biomove/1628_gps.csv',
            IMPORT_PATH_GENERAL + 'Waschbär wild BB - Caro Scholz-Biomove/1631_gps.csv',
            IMPORT_PATH_GENERAL + 'Waschbär wild BB - Caro Scholz-Biomove/1633_gps.csv',
            IMPORT_PATH_GENERAL + 'Waschbär wild BB - Caro Scholz-Biomove/1634_gps.csv',
            IMPORT_PATH_GENERAL + 'Waschbär wild BB - Caro Scholz-Biomove/1636_gps.csv',
            IMPORT_PATH_GENERAL + 'Waschbär wild BB - Caro Scholz-Biomove/1637_gps.csv',
            IMPORT_PATH_GENERAL + 'Waschbär wild BB - Caro Scholz-Biomove/1638_gps.csv'],
        'fs': 18.87}
}

EXPORT_PATH = '/home/eva/Schreibtisch/Master/NeuerVersuch/'

COLUMNS_PREDICTORS = ["datetime", "Xmean", "Xvar", "Xmin", "Xmax", "Xmax - Xmin",
                      "Ymean", "Yvar", "Ymin", "Ymax", "Ymax - Ymin",
                      "Zmean", "Zvar", "Zmin", "Zmax", "Zmax - Zmin",
                      "XZmean", "XZvar", "XZmin", "XZmax", "XZmax - XZmin",
                      "Xdyn", "Ydyn", "Zdyn", "XZdyn",
                      "Ndyn", "Nvar",
                      "Odba",
                      "Pitch",
                      "fft_base", "fft_max", "fft_wmean", "fft_std"]
REDUCED_FEATURES = ['datetime', 'Ymean', 'Ymin', 'Ymax', 'XZmean', 'XZmin', 'Ndyn', 'fft_base', 'fft_wmean', 'fft_std']
MAPPING = {'resting': 0, 'exploring': 1, 'walking': 2, 'climbing': 3, 'high energy': 4, 'unknown': 5}
inverted_mapping = {v: k for k, v in MAPPING.items()}
COLOR_MAPPING = {'resting': 'xkcd:goldenrod',
                 'intermediate energy': 'xkcd:cadet blue',
                 'exploring': 'xkdc:baby blue',
                 'walking': 'xkcd:darkish purple',
                 'climbing': 'xkcd:orchid',
                 'high energy': 'xkcd:bordeaux',
                 'unknown': 'r'
                 }
COLOR_MAPPING_HTML = {'resting': '#f9bc08',
                      'intermediate energy': '#4e7496',
                      'exploring': '#a2cffe',
                      'walking': '#751973',
                      'climbing': '#c875c4',
                      'high energy': '#7b002c',
                      'unknown': '#FF0000'
                      }
MAPPING_3 = {'resting': 0, 'intermediate energy': 1, 'high energy': 2}
MAPPING_3_INVERSE = {v: k for k, v in MAPPING_3.items()}
MAPPING_INT = {'exploring': 0, 'climbing': 1, 'walking': 2}
MAPPING_INT_INVERSE = {v: k for k, v in MAPPING_INT.items()}

GENERAL_FREQ = 33.3
SAMPLE_LENGTH_AT_GEN_FREQ = 54


def combine_date_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine date and time columns into single datetime column.

    @param df: DataFrame with 'date' and 'time' columns.
    @return: DataFrame with combined 'datetime' column, without 'date' & 'time'.
    """
    df = pd.DataFrame(df)

    # specifying the possible input date formats
    date_formats = ['%d.%m.%Y', '%Y-%m-%d', '%d.%m.%y']

    # check which format is present in the dataframe
    for fmt in date_formats:
        try:
            # convert date to datetime object 'date'
            df['date_parsed'] = pd.to_datetime(df['date'], format=fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"None of the date formats matched the input data in column {'date'}")

    # Combine date and time columns into a single datetime column
    df['datetime'] = pd.to_datetime(df['date_parsed'].astype(str) + ' ' + df['time'])

    # drop the now unnecessary columns
    df = df.drop(columns=['date_parsed'])
    df = df.drop(columns=['date'])
    df = df.drop(columns=['time'])

    return df


def x_z_combination(predictors: pd.DataFrame) -> pd.DataFrame:
    """
    Combining x and z variables into one xz variable.

    @param predictors: dataframe imported from the predictor files.
    @output: dataframe without seperate x and z variables, leaving only the combined columns.
    """

    predictors = predictors.drop(
        ['Xmean', 'Xvar', 'Xmin', 'Xmax', 'Xmax - Xmin', 'Zmean', 'Zvar', 'Zmin', 'Zmax', 'Zmax - Zmin'], axis=1)
    return predictors


def timestamp_to_datetime(acc: pd.DataFrame) -> pd.DataFrame:
    """
    function to convert the timestamp column to a datetime column
    @param acc: DataFrame having a 'timestamp' column
    @return: DataFrame with a 'datetime' column in datetime format
    """
    acc[['date', 'time']] = acc['timestamp'].str.split(' ', expand=True)

    # converting to datetime
    acc['time'] = acc['time'].str.slice(0, 8)
    acc = combine_date_time(acc)
    acc['datetime'] = pd.to_datetime(acc['datetime'])

    return acc


def remove_outliers(pred_func: pd.DataFrame) -> pd.DataFrame:
    """
    function to remove outliers from predcitor data set
    @param pred_func: predictor data
    @return: dataframe with removed outliers
    """
    threshold_z = 4
    pred_row_0 = pred_func.shape[0]
    pred_grouped = pred_func.groupby('behavior')
    for (name_func), group in pred_grouped:
        z = np.abs(stats.zscore(group['Ndyn']))
        outlier_indices = z.index[np.where(z > threshold_z)[0]]
        pred_func = pred_func.drop(outlier_indices)
    pred_row_1 = pred_func.shape[0]
    print(pred_row_0 - pred_row_1)

    return pred_func


def adding_time_to_df(y_pred_prob: pd.DataFrame, times: pd.Series) -> pd.DataFrame:
    """
    function to add the datetime column and split it into date, month, time, hour columns
    @param y_pred_prob: dataframe with as many rows as pd.Series times
    @param times: all times for a dataframe
    @return: dataframe with a new 'datetime', 'date', 'month', 'time', 'hour' column
    """
    y_pred_prob['datetime'] = times
    y_pred_prob['datetime'] = pd.to_datetime(y_pred_prob['datetime'], errors='coerce')
    y_pred_prob['date'] = y_pred_prob['datetime'].dt.date
    y_pred_prob['month'] = y_pred_prob['datetime'].dt.month
    y_pred_prob['time'] = y_pred_prob['datetime'].dt.time
    y_pred_prob['hour'] = y_pred_prob['datetime'].dt.hour

    return y_pred_prob


def check_or_input_path(hardcoded_path, description):
    """
    Check if the hardcoded path exists. If not, window for input appears.

    @param hardcoded_path: original hardcoded path.
    @param description: description for the window.

    @return: valid path, either the original or one manually entered.
    """
    if os.path.exists(hardcoded_path):
        print(f"Path exists: {hardcoded_path}")
        return hardcoded_path
    else:
        print(f"Path not found: {hardcoded_path}")
        while True:
            user_input = input(f"Please provide the equivalent path for '{description}': ").strip()
            if os.path.exists(user_input):
                print(f"Path selected: {user_input}")
                return user_input
            else:
                print(f"Invalid path: {user_input}. Please try again.")
