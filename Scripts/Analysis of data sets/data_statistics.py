#!/usr/bin/python3

"""
Filename: data_statistics.py
Author: Eva Reinhardt
Date: 2024-12-09
Version: 1.0
Description: displaying simple statistics for the different data sets. Included are class proportions fpr the labelled
data, combination of dates from the different labelled data sets that amount to 10% of the data set (external test)
as well as statistics for the wild data.
"""

from itertools import combinations
import re

import pandas as pd
import csv

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
