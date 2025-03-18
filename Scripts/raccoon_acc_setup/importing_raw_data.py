#!/usr/bin/python3
"""
Filename: importing_raw_data.py
Author: Eva Reinhardt
Date: 2024-12-09
Version: 1.0
Description: This file contains different import functions for raw data and behavior files.

functions:
import_eobs(): importing raw acc files from eobs files

import_inge(): importing from Inge's dataset, csv files with one line per burst

split_burst(): splitting the one line format into burst

import_acc_data(): importing acc data dependent on filepath

import_beh_peter(): importing Peter's behavior

import_beh_domi(): importing Dominique's behavior

import_beh_inge(): importing Inge's behavior

merge_inge(): merging acc data from inge with behavior

merge domi(): merging acc data from Dominique with behavior

convert_beh(): converting behavior (translation and generalization options)

behavior_combi_domi(): Function to generalize certain behaviors in Dominique's datasets
"""
import pandas as pd
import numpy as np
import csv

from . import variables_simplefunctions as sim_func
from . import gui_functions as guif


def import_eobs(filepath: str) -> pd.DataFrame:
    """
    Import accelerometer data from a specified file path.
    Function is specified for data from Peter and Dominique.

    @param filepath: Path to the file to import.
    @return: DataFrame containing the accelerometer data.
                (columns: 'datetime', 'raw_x', 'raw_y', 'raw_z', 'X', 'Y', 'Z', 'XZ')
    """

    # read the data from the chosen file
    acc = pd.read_csv(filepath, sep=',', header=None)

    if 'Wilbert' in filepath:
        acc = acc.iloc[:, [1, 3, 4, 5, 6]]
    else:
        acc = acc.iloc[:, [1, 3, 5, 4, 6]]


    acc['X'] = (acc.iloc[:, 2] - 2048) / 512
    acc['Y'] = (acc.iloc[:, 3] - 2048) / 512
    acc['Z'] = (acc.iloc[:, 4] - 2048) / 512
    acc['XZ'] = (acc['X']**2 + acc['Z']**2)**0.5

    # naming the columns and correcting the date format
    acc.columns = ['date', 'time', 'raw_x', 'raw_y', 'raw_z', 'X', 'Y', 'Z', 'XZ']
    acc = sim_func.combine_date_time(acc)

    return acc


def split_burst(burst: pd.DataFrame):
    burst = burst.filter(regex='timestamp|raw|tag-local|datetime')
    print(burst.columns)

    if 'study-local-timestamp' in burst.columns:
        burst[['date', 'time']] = burst['study-local-timestamp'].str.split(' ', expand=True)
    elif 'timestamp' in burst.columns:
        burst[['date', 'time']] = burst['timestamp'].str.split(' ', expand=True)
        print(burst)
        print('elif')
    elif 'datetime' in burst.columns:
        burst['date'] = str(burst['datetime'].dt.date)
        burst['time'] = str(burst['datetime'].dt.time)


    print(burst)
    print(burst.columns)
    if 'datetime' not in burst.columns:
    # converting to datetime
        burst['time'] = burst['time'].str.slice(0,8)
        burst = sim_func.combine_date_time(burst)

    # dropping unused columns
    burst = burst.drop(['timestamp','eobs:start-timestamp', 'study-local-timestamp'], axis=1, errors='ignore')

    raw_column = [col for col in burst.columns if 'raw' in col]
    # split the raw data column
    acc_raw_split = burst[raw_column[0]].iloc[0]
    acc_raw_split = acc_raw_split.split()
    # every 3 values are one row
    array_2d = np.array(acc_raw_split).reshape(-1, 3)
    # the new columns for raw_x, raw_y, raw_z are added, as well as datetime
    add = pd.DataFrame(array_2d, columns=['raw_y', 'raw_x', 'raw_z'])
    add['datetime'] = burst['datetime'].iloc[0]


    # raw values are converted to g
    add['X'] = (add['raw_x'].astype(float) - 2048) / 512
    add['Y'] = (add['raw_y'].astype(float) - 2048) / 512
    add['Z'] = (add['raw_z'].astype(float) - 2048) / 512
    add['XZ'] = (add['X']**2 + add['Z']**2)**0.5

    acc = add[['datetime', 'raw_x', 'raw_y', 'raw_z', 'X', 'Y', 'Z', 'XZ']]

    return acc

def import_acc_data(filepath:str) -> pd.DataFrame:
    """
    importing acc data according to the filepath
    @param filepath: filepath to the acc file
    @return: dataframe from the acc file
    """
    if filepath[-4:] == '.csv':
        acc = pd.read_csv(filepath, sep=',')
        if 'timestamp' not in acc.columns:
            acc.columns = acc.iloc[0]
            acc = acc.drop(0, axis=0)

        if '0' in acc.columns:
            acc.columns = acc.iloc[0]
            acc = acc.drop(0, axis=0)

        raw_col = [col for col in acc.columns if 'raw' in col]
        acc['raw-data'] = acc[raw_col[0]]
        acc = sim_func.timestamp_to_datetime(acc)
        acc = acc[['datetime', 'raw-data']]

    else:
        with open(filepath, 'r') as file:
            lines = file.readlines()

        if lines and lines[0][0].isdigit():
            print('eobs data: use import_eobs(), outputs different format')
            return None

        else:
            data = []
            for line in lines:
                if line.startswith('G'):
                    continue

                if line[0]=='A':
                    line_new = line.split(',')
                    line_acc = line_new[5:]
                    line_acc = ' '.join(line_acc)
                    line_new = line_new[0:5]
                    line_new.append(line_acc)
                    data.append(line_new)
            acc = pd.DataFrame(data, columns= ['acc', 'tag_logger', 'date', 'day_of_the_week', 'time', 'raw_data'])
            acc = sim_func.combine_date_time(acc)
            acc = acc.drop(['acc', 'day_of_the_week', 'tag_logger'], axis=1)

    return acc

def import_beh_peter(filepath: str) -> pd.DataFrame:
    """
    Import behavior data from Peter's data from a specified file path.

    @param filepath: Path to the file to import.
    @return: DataFrame containing the behavior data (columns: 'datetime', 'behavior')
    """
    beh = pd.read_csv(filepath, header=None)
    columns = ['date', 'time', 'behavior']
    beh.columns = columns
    beh = sim_func.combine_date_time(beh)

    # fill NaN values of behavior with '-'. '-' is also used in the behavior files if the labeling is undecided
    beh['behavior'] = beh['behavior'].fillna('-')

    # the data is filtered, only bursts with a defined behavior remain
    beh = beh[beh['behavior'].str.contains('[a-zA-Z]', regex=True)]

    return beh

def import_beh_domi(filepath: str) -> pd.DataFrame:
    """
    Import behavior from Dominique's data from specified file path.

    @param filepath: Path to the file to import.
    @return: DataFrame containing the behavior data.
                (columns: 'datetime', 'behavior_Ottilie', 'behavior_Lisa')"""

    # import file, specify and rename columns
    beh = pd.read_csv(filepath, delimiter=';')
    beh = beh[['Tag', 'Zeit', 'Verhalten', 'Tier']]
    columns = ['date', 'time', 'behavior', 'Tier']
    beh.columns = columns

    beh['date'] = beh['date'].astype(str)
    # only keep rows with actual dates and times
    beh = beh[~beh['date'].str.contains('[a-zA-Z]', regex=True)]
    beh = beh[~beh['time'].str.contains('[a-zA-Z]', regex=True)]
    beh = sim_func.combine_date_time(beh)

    # split dataset by animal
    beh_grouped = beh.groupby(['Tier'])
    lisa = beh_grouped.get_group('Lisa')
    ottilie = beh_grouped.get_group('Ottilie')

    # merge together to have a behavior column per individual
    beh = pd.merge(lisa, ottilie, left_on=['datetime'], right_on=['datetime'], how='outer')
    beh.columns = ['behavior_Lisa', 'Name_L', 'datetime', 'behavior_Ottilie', 'Name_O']

    # drop repetitive columns
    beh = beh.drop(['Name_L', 'Name_O'], axis=1)

    # sort columns, with the datetime column being the first one
    cols = ['datetime']  + [col for col in beh if col != 'datetime']
    beh = beh[cols]

    return beh


def merge_domi(acc: pd.DataFrame, beh: pd.DataFrame, max_time_diff: float = 15.0):
    """
    Merging acceleration and behavior data from Dominique's data sets.
    The datetime column is not always equivalent.

    @param acc: DataFrame containing the acceleration data
                (columns: 'datetime', 'raw_x', 'raw_y', 'raw_z', 'X', 'Y', 'Z', 'XZ)
    @param beh: DataFrame containing the behavior data
                (columns: 'datetime', 'behavior_Lisa', 'behavior_Ottilie')
    @param max_time_diff:   denotes which maximum time difference is used to still
                            merge the data
    """
    # creating date columns to pre-filter the data
    acc['date'] = acc['datetime'].dt.date
    beh['date'] = beh['datetime'].dt.date

    # only use acceleration data from dates that are in the behavior data set.
    # reduces the needed storage space
    dates = set(acc['date']).intersection(set(beh['date']))
    acc = acc[acc['date'].isin(list(dates))]

    # grouping data by date and datetime
    grouped_acc_date = acc.groupby(['date'])
    grouped_acc_datetime = acc.groupby(['datetime'])
    grouped_beh = beh.groupby(['date'])

    # initializing the merged dataframe
    merged_df = pd.DataFrame(np.zeros([acc.shape[0], 9])*np.nan )

    # specifying the dataframe's columns
    merged_df.columns = ['raw_x', 'raw_y', 'raw_z',  'X', 'Y', 'Z', 'datetime', 'behavior_Lisa', 'behavior_Ottilie']
    merged_df = merged_df.astype(dtype={"raw_x":"float64", "raw_y":"float64", "raw_z":"float64",
                                        "X":"float64", "Y":"float64", "Z":"float64", "XZ":"float64", "datetime":"datetime64[ns]",
                                        "behavior_Lisa":"object", "behavior_Ottilie":"object"})
    j=0

    # iterating through the dates
    for date in dates:

        # temporary dataframes filtered by date
        group = grouped_beh.get_group(date)
        acc_temp = grouped_acc_date.get_group(date)

        # iterating through behavior rows
        for (i), row in group.iterrows():
            # finding the closest burst to the behavior datetime
            closest_time = acc_temp.iloc[(acc_temp['datetime'] - row['datetime']).abs().argsort()[:1]]

            # calculating the time difference
            time_diff = abs((pd.Timestamp(closest_time['datetime'].values[0]) - row['datetime']).total_seconds()) #.total_seconds()
            print(time_diff)

            # checking if the time difference is smaller or equal to the max_time_diff
            if time_diff <= max_time_diff:

                # get all the values of the acceleration data set for the found datetime
                add = grouped_acc_datetime.get_group(pd.Timestamp(closest_time['datetime'].values[0]))
                add = add.drop(columns = 'date')

                # adding the behavior data to the burst data
                add['behavior_Lisa'] = row['behavior_Lisa']
                add['behavior_Ottilie'] = row['behavior_Ottilie']

                # adding the acceleration data combined with the behavior data to
                # the initiated data frame
                merged_df.iloc[j:(j+add.shape[0]), 0:add.shape[1]] = add.values
                j = j+1+add.shape[0]


    merged_df.columns = ['raw_x', 'raw_y', 'raw_z',  'X', 'Y', 'Z', 'XZ', 'datetime', 'behavior_Lisa', 'behavior_Ottilie']
    return merged_df

def convert_beh(pred: pd.DataFrame, name: str, option_gen: str = "") -> pd.DataFrame:
    """
    Converting behavior from abbreviations to words using dict files

    @param pred: Dataframe containing behavior columns
    @param name: Name of the person whose data sets are used. options are 'Peter', 'Dominique', 'Inge', 'generalization', 'translation'
    @param option_gen: option between generalization1 or generalization2
    @return: Dataframe with converted behavior columns
    """
    # according to the name, the respective dict is imported
    if name == 'Peter':
        file = open(sim_func.IMPORT_PARAMETERS['Peter']['path'] + 'dict_peter.csv', mode = 'r', encoding='utf-8')
        reader = csv.reader(file)
        dict_beh = {rows[0]: rows[1] for rows in reader}
    elif name == 'Dominique':
        file = open(sim_func.IMPORT_PARAMETERS['Dominique']['path']+ 'dict_dominique_new.csv', mode = 'r', encoding='utf-8')
        reader = csv.reader(file, delimiter=',')
        dict_beh = {rows[0]: rows[1] for rows in reader}
    elif name == 'Inge':
        file = open('/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Msc Inge Buchholz/dict_inge.csv', mode = 'r', encoding='utf-8')
        reader = csv.reader(file)
        dict_beh = {rows[0]: rows[1] for rows in reader}
    elif name == 'generalization':
        if (option_gen == "") or (option_gen not in ['generalization1', 'generalization2', 'generalization3']):
            option_gen = guif.choose_option(['generalization1', 'generalization2', 'generalization3'])

        filename = sim_func.IMPORT_PATH_GENERAL+'dict_' + option_gen + '.csv'
        file = open(filename, mode = 'r', encoding='utf-8')
        reader = csv.reader(file, delimiter=';')

        dict_beh = {rows[0]: rows[1] for rows in reader}
    elif name == 'translation':
        file = open(sim_func.IMPORT_PATH_GENERAL+'translation.csv', mode = 'r', encoding='utf-8')
        reader = csv.reader(file, delimiter=';')
        dict_beh = {rows[0]: rows[1] for rows in reader}
    else:
        raise ValueError('Options for name are: Peter, Dominique, Inge, generalization.')

    def replace_items(cell: str) -> str:
        """
        Replacing abbreviations with string from the dictionary

        @param cell: cell from the csv file
        @return: string with the replaced items
        """
        if name == 'generalization':
            if pd.isna(cell):
                return cell
            # elements are replaced and joined
            replaced_items = dict_beh.get(cell, cell)
            return replaced_items

        else:
            if pd.isna(cell):
                return cell
            if name != 'translation':
                # cell is split in case of multiple entries
                items = cell.split(',')
            else:
                items = [cell]
            # elements are replaced and joined
            replaced_items = [dict_beh.get(item, item) for item in items]
            return '; '.join(replaced_items)

    # only replacing items in behavior columns
    behavior_columns = [col for col in pred.columns if 'behavior' in col]
    behavior_columns = [col for col in behavior_columns if 'generalization' not in col]

    # replace_items function applied to each cell in the DataFrame, if it is not a datetime object
    for column in behavior_columns:
        if name != 'generalization':
            if name == 'Peter':
                pred.loc[:, column] = pred[column].str.replace(r'[^a-zA-ZäöüÄÖÜß\s]', '', regex=True)
                pred.loc[:, column] = pred[column].str.replace(r'[\s]', '', regex=True)
            pred.loc[:, column] = pred[column].apply(replace_items)
        elif name == 'generalization':
            pred[f'{column}_generalization'] = pred[column].apply(replace_items)

    return pred

def behavior_combi_domi(pred_1: pd.DataFrame) -> pd.DataFrame:
    """
    Function to generalize certain behaviors in Dominique's datasets

    @param pred_1: predictor Dataset
    @return: predictor Dataset where certain Datapoints are removed or combined
    """
    replacement_dict = {
        'klettern': ['klettern', 'rein klettern', 'raus klettern', 'rausklettern', 'klettern '],
        'stehen AH': ['stehen AH', 'stehene AH'],
        'laufgalopp': ['laufgalopp', 'laufen+galopp', 'galopp+laufen', 'lauf+galopp+stopp'],
        'stehen': ['stehen', 'stehen ']
    }
    words_to_remove = ['nicht mehr', 'ins Haus', 'raus']

    replacement_mapping = {v: k for k, values in replacement_dict.items() for v in values}

    pred_1['behavior'] = pred_1['behavior'].replace(replacement_mapping, regex=False)

    pattern = '|'.join(words_to_remove)

    pred_1 = pred_1.dropna()

    pred_1 = pred_1[~pred_1['behavior'].str.contains(pattern, regex=True)]
    return pred_1

