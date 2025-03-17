#!/usr/bin/python3
from xmlrpc.client import boolean

import pandas as pd
import os
import numpy as np
import datetime
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
import csv
from typing import List, Union
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, \
    f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV

columns_predictors = ["datetime", "Xmean", "Xvar", "Xmin", "Xmax", "Xmax - Xmin",
                      "Ymean", "Yvar", "Ymin", "Ymax", "Ymax - Ymin",
                      "Zmean", "Zvar", "Zmin", "Zmax", "Zmax - Zmin",
                      "XZmean", "XZvar", "XZmin", "XZmax", "XZmax - XZmin",
                      "Xdyn", "Ydyn", "Zdyn", "XZdyn",
                      "Ndyn", "Nvar",
                      "Odba",
                      "Pitch", # "Yaw",
                      "fft_base", "fft_max", "fft_wmean", "fft_std"]
reduced_features = ['datetime','Ymean', 'XZmean', 'XZmin', 'XZmax', 'Ndyn', 'fft_base', 'fft_wmean', 'fft_std']
mapping = {'resting':0, 'exploring':1, 'walking': 2, 'climbing':3, 'high energy':4, 'unknown':5}
inverted_mapping = {v: k for k, v in mapping.items()}
color_mapping = {'resting': 'xkcd:goldenrod',
                 'intermediate energy': 'xkcd:cadet blue',
                  'exploring': 'xkdc:baby blue',
                  'walking': 'xkcd:darkish purple',
                  'climbing': 'xkcd:orchid',
                  'high energy': 'xkcd:bordeaux',
                  'unknown': 'r'
                 }
color_mapping_html = {'resting': '#f9bc08',
                 'intermediate energy': '#4e7496',
                  'exploring': '#a2cffe',
                  'walking': '#751973',
                  'climbing': '#c875c4',
                  'high energy': '#7b002c',
                  'unknown': '#FF0000'
                 }

def __init__():
    pass

def open_file_dialog(which_file: str) -> Union[str, List[str]]:
    """
    Open a file dialog to select a file.

    @param which_file:  The title of the dialog window, specifying which data
                        is needed at this point.
    @return: the selected file path(s), a list of strings or a single string.
    """

    # Create the root window
    root = tk.Tk()
    root.withdraw()

    # Open the file dialog
    file_path = filedialog.askopenfilenames(
        title=which_file,
        filetypes=(("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*"))
    )

    return(file_path)



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
    df = df.drop(columns = ['date_parsed'])
    df = df.drop(columns = ['date'])
    df = df.drop(columns = ['time'])

    return df


def choose_option(options: List[str], title: str = "Choose an Option") -> str:
    """
    Present dialog window to choose an option.

    @param options: List of options to choose from.
    @param title: Title of the dialog window.
    @param return: Selected option.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Function to store the selected option and close the dialog
    def select_option(option):
        nonlocal selected_option
        selected_option = option
        dialog.destroy()

    # Create the dialog window
    dialog = tk.Toplevel(root)
    dialog.title(title)

    tk.Label(dialog, text="Please choose an option:").pack(pady=10)

    # Create buttons for 'Emma' and 'Susi'
    for option in options:
        tk.Button(dialog, text=option, command=lambda opt=option: select_option(opt)).pack(pady=5)


    # Initialize the selected_option variable
    selected_option = None

    # Wait for the dialog to be closed
    root.wait_window(dialog)

    return selected_option


def choose_multiple_options(options: List[str], title: str = "Choose Options") -> List[str]:
    """
    Present dialog window to choose multiple options.

    @param options: List of options to choose from.
    @param title: Title of the dialog window.
    @return: List of selected options.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    selected_options = []

    # Create the dialog window
    dialog = tk.Toplevel(root)
    dialog.title(title)

    tk.Label(dialog, text="Please choose options:").pack(pady=10)

    # Create checkboxes for each option
    vars = {}
    for option in options:
        var = tk.IntVar()
        chk = tk.Checkbutton(dialog, text=option, variable=var)
        chk.pack(anchor='w')  # Anchor left for better UI
        vars[option] = var

    # Function to store the selected options and close the dialog
    def select_options():
        print("OK button clicked")
        for option, var in vars.items():
            print(f"{option}: {var.get()}")
            if var.get():  # Only store if the checkbox is checked
                selected_options.append(option)
        print("Selected options:", selected_options)
        dialog.destroy()
        root.quit()  # Ensures the root window is properly closed

    # Create an OK button to confirm selections
    ok_button = tk.Button(dialog, text="OK", command=select_options)
    ok_button.pack(pady=10)

    # Start the main event loop
    root.mainloop()

    return selected_options





def import_pd(filepath: str) -> pd.DataFrame:
    """
    Import accelerometer data from a specified file path.
    Function is specified for data from Peter and Dominique.

    @param filepath: Path to the file to import.
    @return: DataFrame containing the accelerometer data.
                (columns: 'datetime', 'raw_x', 'raw_y', 'raw_z', 'X', 'Y', 'Z', 'XZ')
    """

    # read the data from the chosen file
    ACC = pd.read_table(filepath, sep=',', header=None)

    # some printing functions to check the data
    # print(ACC.iloc[1, :])
    # ACC.shape
    ACC = ACC.iloc[:, [1, 3, 4, 5, 6]]


    ACC['X'] = (ACC.iloc[:, 2] - 2048) / 512
    ACC['Y'] = (ACC.iloc[:, 3] - 2048) / 512
    ACC['Z'] = (ACC.iloc[:, 4] - 2048) / 512
    ACC['XZ'] = (ACC['X']**2 + ACC['Z']**2)**0.5

    # naming the columns and correcting the date format
    ACC.columns = ['date', 'time', 'raw_x', 'raw_y', 'raw_z', 'X', 'Y', 'Z', 'XZ']
    ACC = combine_date_time(ACC)
    # print(ACC.iloc[1:3, :])

    return ACC


def options_inge() -> str:
    """
    Present dialog window to make decisions necessary for Inge's dataset:
    animals: Emma and Susi

    @return: Selected option
    """
    select = choose_option(['Emma', 'Susi', 'Both'])
    #select_logger = choose_option([2486, 2487], title='Choose logger')
    return select



def import_inge(filepath: str) -> pd.DataFrame:
    """
    Import accelerometer data from Inge's Dataset from specified filepath.

    @param filepath: Path to the file to import
    @return: DataFrame containing the accelerometer data
                (columns: 'datetime', 'raw_x', 'raw_y', 'raw_z', 'X', 'Y', 'Z')
    """

    # importing the accelerometer data
    ACC1 = pd.read_table(filepath, sep=',', low_memory=False)

    if '0' in ACC1.columns:
        ACC1.columns = ACC1.iloc[0]
        ACC1 = ACC1.drop(0, axis=0)
    # filtering for needed columns
    ACC1 = ACC1.filter(regex='timestamp|-raw|tag-local')

    # drop NaN values, check again at this point
    ACC1 = ACC1.dropna()

    # decision about which time to use
    if 'study-local-timestamp' in ACC1.columns:
        ACC1[['date', 'time']] = ACC1['study-local-timestamp'].str.split(' ', expand=True)
    else:
        ACC1[['date', 'time']] = ACC1['timestamp'].str.split(' ', expand=True)
    # ACC1[['date', 'time']] = ACC1['timestamp'].str.split(' ', expand=True)
    # ACC1[['date', 'time']] = ACC1['eobs:start-timestamp'].str.split(' ', expand=True)

    # converting to datetime
    ACC1['time'] = ACC1['time'].str.slice(0,8)
    ACC1 = combine_date_time(ACC1)

    # dropping unused columns
    ACC1 = ACC1.drop(['timestamp','eobs:start-timestamp', 'study-local-timestamp'], axis=1, errors='ignore')


    # grouping by logger, unnecessary if input files are already divided by logger
    # logger = ACC1.groupby('tag-local-identifier')

    # all the raw data is concatenated in one row, ACC is created and used to split
    # the row in raw_x, raw_, raw_z
    ACC = pd.DataFrame()

    # iterating through rows
    for row in ACC1.itertuples(index=True):
        # split the raw data column
        acc_raw_split = row[1].split()
        # every 3 values are one row
        array_2d = np.array(acc_raw_split).reshape(-1, 3)
        # the new columns for raw_x, raw_y, raw_z are added, as well as datetime
        add = pd.DataFrame(array_2d, columns=['raw_x', 'raw_y', 'raw_z'])
        add['datetime'] = row.datetime
        add['tag-local-identifier'] = row[2]
        # all are concatenated to the overall dataframe ACC
        ACC = pd.concat([ACC, add], ignore_index=True)

    # raw values are converted to g
    ACC['X'] = (ACC['raw_x'].astype(float) - 2048) / 512
    ACC['Y'] = (ACC['raw_y'].astype(float) - 2048) / 512
    ACC['Z'] = (ACC['raw_z'].astype(float) - 2048) / 512
    ACC['XZ'] = (ACC['X']**2 + ACC['Z']**2)**0.5

    ACC = ACC[['datetime', 'raw_x', 'raw_y', 'raw_z', 'X', 'Y', 'Z', 'XZ', 'tag-local-identifier']]

    return ACC

def split_burst(burst: tuple):

    burst = burst.filter(regex='timestamp|-raw|tag-local')

    if 'study-local-timestamp' in burst.columns:
        burst[['date', 'time']] = burst['study-local-timestamp'].str.split(' ', expand=True)
    else:
        burst[['date', 'time']] = burst['timestamp'].str.split(' ', expand=True)
    # ACC1[['date', 'time']] = ACC1['timestamp'].str.split(' ', expand=True)
    # ACC1[['date', 'time']] = ACC1['eobs:start-timestamp'].str.split(' ', expand=True)

    # converting to datetime
    burst['time'] = burst['time'].str.slice(0,8)
    burst = combine_date_time(burst)

    # dropping unused columns
    burst = burst.drop(['timestamp','eobs:start-timestamp', 'study-local-timestamp'], axis=1, errors='ignore')

    # split the raw data column
    acc_raw_split = burst['eobs:accelerations-raw'].iloc[0].split()
    # every 3 values are one row
    array_2d = np.array(acc_raw_split).reshape(-1, 3)
    # the new columns for raw_x, raw_y, raw_z are added, as well as datetime
    add = pd.DataFrame(array_2d, columns=['raw_x', 'raw_y', 'raw_z'])
    add['datetime'] = burst['datetime'].iloc[0]
    add['tag-local-identifier'] = burst['tag-local-identifier'].iloc[0]

    # raw values are converted to g
    add['X'] = (add['raw_x'].astype(float) - 2048) / 512
    add['Y'] = (add['raw_y'].astype(float) - 2048) / 512
    add['Z'] = (add['raw_z'].astype(float) - 2048) / 512
    add['XZ'] = (add['X']**2 + add['Z']**2)**0.5

    ACC = add[['datetime', 'raw_x', 'raw_y', 'raw_z', 'X', 'Y', 'Z', 'XZ']]

    return ACC


def import_acc_data(filepath:str) -> pd.DataFrame:
    if filepath[-4:] == '.csv':
        acc = pd.read_csv(filepath, sep=',')
        if 'timestamp' not in acc.columns:
            acc.columns = acc.iloc[0]
            acc = acc.drop(0, axis=0)

        if '0' in acc.columns:
            acc.columns = acc.iloc[0]
            acc = acc.drop(0, axis=0)
    else:
        with open(filepath, 'r') as file:
            lines = file.readlines()

        if lines and lines[0][0].isdigit():
            # read the data from the chosen file
            acc = pd.read_table(filepath, sep=',', header=None)

            # some printing functions to check the data
            # print(ACC.iloc[1, :])
            # ACC.shape
            acc = acc.iloc[:, [1, 3, 4, 5, 6]]

            acc['X'] = (acc.iloc[:, 2] - 2048) / 512
            acc['Y'] = (acc.iloc[:, 3] - 2048) / 512
            acc['Z'] = (acc.iloc[:, 4] - 2048) / 512
            acc['XZ'] = (acc['X'] ** 2 + acc['Z'] ** 2) ** 0.5

            # naming the columns and correcting the date format
            acc.columns = ['date', 'time', 'raw_x', 'raw_y', 'raw_z', 'X', 'Y', 'Z', 'XZ']
            acc = combine_date_time(acc)
            # print(ACC.iloc[1:3, :])
        else:
            data = []
            skip_next = False
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
            acc = pd.DataFrame(data, columns= ['ACC', 'tag_logger', 'date', 'day_of_the_week', 'time', 'raw_data'])
            acc = combine_date_time(acc)
            acc = acc.drop(['ACC', 'day_of_the_week'], axis=1)
            # acc1 = pd.DataFrame()
            #
            # # iterating through rows
            # for row in acc.itertuples(index=True):
            #     # split the raw data column
            #     acc_raw_split = row[2].split()
            #     # every 3 values are one row
            #     array_2d = np.array(acc_raw_split).reshape(-1, 3)
            #     # the new columns for raw_x, raw_y, raw_z are added, as well as datetime
            #     add = pd.DataFrame(array_2d, columns=['raw_x', 'raw_y', 'raw_z'])
            #     add['datetime'] = row.datetime
            #     # all are concatenated to the overall dataframe ACC
            #     acc1 = pd.concat([acc1, add], ignore_index=True)
            #     print(row.datetime)
            #
            # # raw values are converted to g
            # acc1['X'] = (acc1['raw_x'].astype(float) - 2048) / 512
            # acc1['Y'] = (acc1['raw_y'].astype(float) - 2048) / 512
            # acc1['Z'] = (acc1['raw_z'].astype(float) - 2048) / 512
            # acc1['XZ'] = (acc1['X'] ** 2 + acc1['Z'] ** 2) ** 0.5
            # acc = acc1.copy()

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
    beh = combine_date_time(beh)

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
    beh = combine_date_time(beh)

    # split dataset by animal
    beh_grouped = beh.groupby(['Tier'])
    Lisa = beh_grouped.get_group('Lisa')
    Ottilie = beh_grouped.get_group('Ottilie')

    # merge together to have a behavior column per individual
    beh = pd.merge(Lisa, Ottilie, left_on=['datetime'], right_on=['datetime'], how='outer')
    beh.columns = ['behavior_Lisa', 'Name_L', 'datetime', 'behavior_Ottilie', 'Name_O']

    # drop repetitive columns
    beh = beh.drop(['Name_L', 'Name_O'], axis=1)

    # sort columns, with the datetime column being the first one
    cols = ['datetime']  + [col for col in beh if col != 'datetime']
    beh = beh[cols]

    return beh


def import_beh_inge(filepath: str, select: str) -> pd.DataFrame:
    """
    Import behavior from Inge's data from specified filepath.

    @param filepath: Path to the file to import.
    @param select: Selected option ('Emma', 'Susi', 'Both').
    @return: DataFrame containing the behavior data.
                (columns: 'datetime', 'Verhalten Emma', 'Verhalten Susi')
    """

    # import file
    beh = pd.read_csv(filepath, low_memory=False)

    # create time column
    beh[['Stunden', 'Minuten', 'Sekunden']] = beh[['Stunden', 'Minuten', 'Sekunden']].map('{:0>2}'.format)
    beh['time'] = beh[['Stunden', 'Minuten', 'Sekunden']].apply(lambda row:':'.join(map(str, row)), axis=1)
    beh = beh.drop(['Stunden', 'Minuten', 'Sekunden'], axis=1)

    # drop rows with no documented behavior for both individuals
    beh = beh.dropna(subset=['Verhalten Emma', 'Verhalten Susi'], how='all')

    # filter the data according to the selected option
    if select == "Emma":
        beh = beh.filter(regex='Datum|time|Emma')
    elif select == 'Susi':
        beh = beh.filter(regex='Datum|time|Susi')
    else:
        beh = beh.filter(regex='Datum|time|Emma|Susi')

    # renaming date column
    beh = beh.rename(columns={'Datum':'date'})

    # print(beh.head())

    # creating datetime column, sort columns drop rows with no documented behavior
    # for both individuals
    beh = combine_date_time(beh)
    cols = ['datetime']  + [col for col in beh if col != 'datetime']
    beh = beh[cols]

    # an dieser Stelle nach oben geschoben, muss noch getestet werden
    #beh = beh.dropna(subset=['Verhalten Emma', 'Verhalten Susi'], how='all')

    return beh





def calculate_features(X, Y, Z, XZ, datetime, fs=33.3) -> list:
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

    # Winkel kontrollieren
    # Roll = np.arctan2(Ymean, np.sqrt(np.square(Xmean) + np.square(Zmean))) * 180 / np.pi
    # Pitch = np.arcsin(Ymean) * 180 / np.pi
    # Yaw = np.arctan2(Zmean, np.sqrt(np.square(Xmean) + np.square(Ymean))) * 180 / np.pi
    Pitch = np.arcsin(Ymean / np.sqrt(XZmean ** 2 + Ymean ** 2))
    #Yaw = np.arcsin(Xmean / XZmean)

    fft_data = np.abs(np.fft.rfft(Nd))[1:len(Nd) // 2 + 1]
    ### Ändern, um die Frequenz zu speichern: fft_base --> Grundfrequenz
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
                 Pitch, #Yaw,
                 fft_base, fft_max, fft_wmean, fft_std]

    return feat_list


def calculate_pred(data: pd.DataFrame, frequence, mw: boolean = False) -> pd.DataFrame:
    """
    Calculate predictors from accelerometer data
    (input dataframe can contain behavior data for one or two inidividuals or none)

    @param data: DataFrame containing the accelerometer data
                    (columns: 'datetime', 'raw_x', 'raw_y', 'raw_z', 'X', 'Y', 'Z',
                    optional columns: behavior (1 or 2 columns))
    @param frequence: frequency of data
    @param mw: moving window option, True or False
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

    global columns_predictors
    # specifying if behavior data is present in 'data'
    # --> 10 columns means 2 behavior columns
    if len(data.columns) ==10:
        # initialize dataframe for two behaviors
        columns = columns_predictors + list(data.columns[-2:])

    elif len(data.columns) == 9:
        #### only one column for behavior

        columns = columns_predictors + list(data.columns[-1:])
    else:
        columns = columns_predictors

    if len(data['timestamp'].unique()) == data.shape[0]:

        if mw == False:
            pred = pd.DataFrame(np.zeros(shape=(data.shape[0], len(columns))), columns=columns)
        else:
            l = len(data['eobs:accelerations-raw'].split(' '))/3
            l = round(l/54, 0)
            pred = pd.DataFrame(np.zeros(shape=(data.shape[0]*l, len(columns))), columns=columns)
        for i in range(0, data.shape[0]):
            burst = split_burst(data.iloc[[i]])
            if mw == True:
                bursts = moving_window(burst)
            else:
                bursts = [burst.iloc[0:54]]

            for b in bursts:
                feat_list = calculate_features(b['X'], b['Y'], b['Z'], b['XZ'], b['datetime'].iloc[0], frequence)
                print(len(feat_list))
                print(pred.shape)

                if len(b.columns) == 10:
                    # adding all the predictors as a row to the end of pred (two behaviors)
                    pred.iloc[i, :] = feat_list + [b.iloc[1, -2], b.iloc[1, -1]]

                elif len(b.columns) == 9:
                    # adding all the predictors as a row to the end of pred (one behavior)
                    pred.iloc[i, :] = feat_list + [b.iloc[1, -1]]

                elif len(b.columns) == 8:
                    # adding all the predictors as a row to the end of pred (no behavior)
                    pred.iloc[i, :] = feat_list



    else:
        # group data by datetime so that each group is one burst
        grouped = data.groupby(['datetime'])


        pred = pd.DataFrame(np.zeros(shape=(len(grouped), len(columns))), columns=columns)

        i=0

        # Iterate through each group
        for (datetime), group in grouped:
            burst = group
            if mw == True:
                bursts = moving_window(burst)
            else:
                bursts = [burst]
            for b in bursts:
                X = b['X'][0:54]
                Y = b['Y'][0:54]
                Z = b['Z'][0:54]
                XZ = b['XZ'][0:54]

                feat_list = calculate_features(X, Y, Z, XZ, datetime[0], frequence)
                print(len(feat_list))
                print(pred.shape)

                if len(data.columns)==10:
                    # adding all the predictors as a row to the end of pred (two behaviors)
                    pred.iloc[i,:] = feat_list + [b.iloc[1,-2], b.iloc[1,-1]]

                elif len(data.columns) == 9:
                    # adding all the predictors as a row to the end of pred (one behavior)
                    pred.iloc[i,:] = feat_list + [b.iloc[1,-1]]

                elif len(data.columns) == 8:
                    # adding all the predictors as a row to the end of pred (no behavior)
                    pred.iloc[i,:] = feat_list
                i=i+1

    return pred




def merge_inge(ACC: pd.DataFrame, beh: pd.DataFrame, max_time_diff: float = 15.0):
    """
    Merging acceleration data from Inge with behavior data from Inge.
    Datetime can differ by some seconds, which is included by max_time_diff.

    @param ACC: DataFrame containing the acceleration data
                (columns: 'datetime', 'raw_x', 'raw_y', 'raw_z', 'X', 'Y', 'Z', 'XZ')
    @param behavior: DataFrame containing the behavior data
                (columns: 'datetime', 'Verhalten Emma', 'Verhalten Susi')
    @param max_time_diff:   denotes which maximum time difference is used to still
                            merge the data
    """

    # creating date columns to pre-filter the data
    ACC['date'] = ACC['datetime'].dt.date
    beh['date'] = beh['datetime'].dt.date

    # only use acceleration data from dates that are in the behavior data set.
    # reduces the needed storage space
    dates = set(ACC['date']).intersection(set(beh['date']))
    ACC = ACC[ACC['date'].isin(list(dates))]

    # grouping data by date and datetime
    grouped_ACC_date = ACC.groupby(['date'])
    grouped_ACC_datetime = ACC.groupby(['datetime'])
    grouped_beh = beh.groupby(['date'])

    # initializing the merged dataframe
    merged_df = pd.DataFrame(np.zeros([ACC.shape[0], 9])*np.nan )

    # specifying the dataframe's columns
    merged_df.columns = ['raw_x', 'raw_y', 'raw_z', 'datetime', 'X', 'Y', 'Z', 'XZ', 'behavior_Emma', 'behavior_Susi']
    merged_df = merged_df.astype(dtype={"raw_x":"float64", "raw_y":"float64", "raw_z":"float64",
                                        "datetime":"datetime64[ns]",
                                        "X":"float64", "Y":"float64", "Z":"float64", "XZ":"float64",
                                        "behavior_Emma":"object", "behavior_Susi":"object"})
    j=0

    # iterating through the dates
    for date in dates:

        # temporary dataframes filtered by date
        group = grouped_beh.get_group(date)
        #print(date)
        acc_temp = grouped_ACC_date.get_group(date)

        # iterating through behavior rows
        for i, row in group.iterrows():
            # finding the closest burst to the behavior datetime
            closest_time = acc_temp.iloc[(acc_temp['datetime'] - row['datetime']).abs().argsort()[:1]]
            #print(closest_time['datetime'])

            # calculating the time difference
            time_diff = abs((pd.Timestamp(closest_time['datetime'].values[0]) - row['datetime']).total_seconds()) #.total_seconds()
            #print(time_diff)
            #print(row)

            # checking if the time difference is smaller or equal to the max_time_diff
            if time_diff <= max_time_diff:

                # get all the values of the acceleration data set for the found datetime
                add = grouped_ACC_datetime.get_group(pd.Timestamp(closest_time['datetime'].values[0]))
                add = add.drop(columns = 'date')
                #print(row['behavior'])

                # adding the behavior data to the burst data
                add['behavior_Emma'] = row['Verhalten Emma']
                add['behavior_Susi'] = row['Verhalten Susi']

                # adding the acceleration data combined with the behavior data to
                # the initiated data frame
                merged_df.iloc[j:(j+add.shape[0]), 0:add.shape[1]] = add.values
                j = j+1+add.shape[0]

    # rows that where initiated but not filled because data didn't match,
    # are being dropped
    merged_df = merged_df.dropna()
    return merged_df


def merge_domi(ACC: pd.DataFrame, beh: pd.DataFrame, max_time_diff: float = 15.0):
    """
    Merging acceleration and behavior data from Domenique's data sets.
    The datetime column is not always equivalent.

    @param ACC: DataFrame containing the acceleration data
                (columns: 'datetime', 'raw_x', 'raw_y', 'raw_z', 'X', 'Y', 'Z', 'XZ)
    @param behavior: DataFrame containing the behavior data
                (columns: 'datetime', 'behavior_Lisa', 'behavior_Ottilie')
    @param max_time_diff:   denotes which maximum time difference is used to still
                            merge the data
    """
    # creating date columns to pre-filter the data
    ACC['date'] = ACC['datetime'].dt.date
    beh['date'] = beh['datetime'].dt.date

    # only use acceleration data from dates that are in the behavior data set.
    # reduces the needed storage space
    dates = set(ACC['date']).intersection(set(beh['date']))
    ACC = ACC[ACC['date'].isin(list(dates))]

    # grouping data by date and datetime
    grouped_ACC_date = ACC.groupby(['date'])
    grouped_ACC_datetime = ACC.groupby(['datetime'])
    grouped_beh = beh.groupby(['date'])

    # initializing the merged dataframe
    merged_df = pd.DataFrame(np.zeros([ACC.shape[0], 9])*np.nan )

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
        #print(date)
        acc_temp = grouped_ACC_date.get_group(date)

        # iterating through behavior rows
        for (i), row in group.iterrows():
            # finding the closest burst to the behavior datetime
            closest_time = acc_temp.iloc[(acc_temp['datetime'] - row['datetime']).abs().argsort()[:1]]
            #print(closest_time['datetime'])

            # calculating the time difference
            time_diff = abs((pd.Timestamp(closest_time['datetime'].values[0]) - row['datetime']).total_seconds()) #.total_seconds()
            print(time_diff)
            #print(row)

            # checking if the time difference is smaller or equal to the max_time_diff
            if time_diff <= max_time_diff:

                # get all the values of the acceleration data set for the found datetime
                add = grouped_ACC_datetime.get_group(pd.Timestamp(closest_time['datetime'].values[0]))
                add = add.drop(columns = 'date')
                #print(row['behavior'])

                # adding the behavior data to the burst data
                add['behavior_Lisa'] = row['behavior_Lisa']
                add['behavior_Ottilie'] = row['behavior_Ottilie']

                # adding the acceleration data combined with the behavior data to
                # the initiated data frame
                merged_df.iloc[j:(j+add.shape[0]), 0:add.shape[1]] = add.values
                j = j+1+add.shape[0]
                #print(add.head(1))

    #merged_df = merged_df.dropna()
    merged_df.columns = ['raw_x', 'raw_y', 'raw_z',  'X', 'Y', 'Z', 'XZ', 'datetime', 'behavior_Lisa', 'behavior_Ottilie']
    return merged_df


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
                pred_1 = convert_beh(pred_1, 'Peter')
                name = 'Peter'
                replacement_dict = {}
                words_to_remove = []
            elif 'Dominique' in filepath:
                name = 'Dominique'
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
                pred_1 = behavior_combi_domi(pred_1)
            pred = pd.concat([pred, pred_1], ignore_index=True)

        if 'behavior' in pred.columns:
            # preparing the dataframe: fitlering out schütteln, filtering out outliers from Ndyn column if wanted
            pred = pred[~(pred['behavior'] == 'schütteln')]
            pred = pred[~((pred['behavior'] == 'schlafen') & (pred['Ndyn'] > 0.5))]
        pred = pred.dropna(axis=1)
        pred = x_z_combination(pred)
        pred_com = pd.concat([pred_com, pred], ignore_index=True)
    if 'Xdyn' in pred_com.columns:
        pred_com = pred_com.drop(["Xdyn", "Zdyn"], axis=1)
    if reduced_features:
        if 'behavior' in pred_com.columns:
            pred_com = pred_com[['datetime','Ymean', 'XZmean', 'XZmin', 'XZmax', 'Ndyn', 'fft_base', 'fft_wmean', 'fft_std', 'behavior']]
        else:
            pred_com = pred_com[['datetime','Ymean', 'XZmean', 'XZmin', 'XZmax', 'Ndyn', 'fft_base', 'fft_wmean', 'fft_std']]
    return pred_com


def splitting_pred(predictions: pd.DataFrame, mapping: dict = {}) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered_columns = [
        col for col in predictions.columns if 'behavior' not in col and 'datetime' not in col]
    predictors = predictions[filtered_columns]
    labels_col = [
        col for col in predictions.columns if 'behavior_generalization' in col]
    labels = predictions[labels_col]
    if not mapping:
        classnames2 = list(labels['behavior_generalization'].unique())
        class_map = {classname: idx for idx, classname in enumerate(classnames2)}
    else:
        class_map = mapping
    labels['behavior_generalization'] = labels['behavior_generalization'].map(class_map).astype(int)
    return predictors, labels


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
    ax.axis([0,len(burst), -1, 1])

    # creating the title, containing the behavior per individual
    names = burst.filter(regex='behavior_')
    names = names.columns
    title = "; ".join([f"{name}: {burst[name].unique()[0]}" for name in names])

    ax.legend()
    ax.set_title(title)

    plt.show()


def save_pred(data: pd.DataFrame):
    """
    Saving the predictor data to a csv file.

    @param data: Dataframe containing the predictor data, datetime column and behavior data
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    file_path = filedialog.asksaveasfilename(title="Save as")
    data.to_csv(file_path, index=False)


def convert_beh(pred: pd.DataFrame, name: str, option_gen: str = "") -> pd.DataFrame:
    """
    Converting behavior from abbreviations to words using dict files

    @param pred: Dataframe containing behavior columns
    @param name: Name of the person who's data sets are used. options are 'Peter', 'Dominique', 'Inge', 'generalization', 'translation'
    @param option_gen: option between generalization1 or generalization2
    @return: Dataframe with converted behavior columns
    """

    # according to the name, the respective dict is imported
    if name == 'Peter':
        file = open('/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/dict_peter.csv', mode = 'r')
        reader = csv.reader(file)
        dict_beh = {rows[0]: rows[1] for rows in reader}
    elif name == 'Dominique':
        file = open('/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/dict_dominique_new.csv', mode = 'r')
        reader = csv.reader(file, delimiter=',')
        dict_beh = {rows[0]: rows[1] for rows in reader}
    elif name == 'Inge':
        file = open('/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Msc Inge Buchholz/dict_inge.csv', mode = 'r')
        reader = csv.reader(file)
        dict_beh = {rows[0]: rows[1] for rows in reader}
    elif name == 'generalization':
        if option_gen == "":
            option_gen = choose_option(['generalization1', 'generalization2', 'generalization3'])
        if option_gen == 'generalization1':
            file = open('/media/eva/eva-reinhar/your folders/01 raw data/dict_generalization1.csv', mode = 'r')
        elif option_gen == 'generalization2':
            file = open('/media/eva/eva-reinhar/your folders/01 raw data/dict_generalization2.csv', mode = 'r')
        elif option_gen == 'generalization3':
            file = open('/media/eva/eva-reinhar/your folders/01 raw data/dict_generalization3.csv', mode = 'r')
        reader = csv.reader(file, delimiter=';')
        dict_beh = {rows[0]: rows[1] for rows in reader}
    elif name == 'translation':
        file = open('/media/eva/eva-reinhar/your folders/01 raw data/translation.csv', mode = 'r')
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

            # cell is split in case of multiple entries
            # items = cell.split(',')

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
             # Add a new 'generalization' column with the generalized behaviors
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

    # Reverse the dictionary for easy replacement
    replacement_mapping = {v: k for k, values in replacement_dict.items() for v in values}

    pred_1['behavior'] = pred_1['behavior'].replace(replacement_mapping, regex=False)

    pattern = '|'.join(words_to_remove)

    pred_1 = pred_1.dropna()

    # Filter rows that do not contain any of the words
    pred_1 = pred_1[~pred_1['behavior'].str.contains(pattern, regex=True)]
    return pred_1


def moving_window(burst: pd.DataFrame) -> list:
    """
    Function to split a burst into stretches of 54 datapoints

    @param burst: Dataframe containing the complete burst
    @return: list of dataframes containing the splitted burst
    """

    bursts = []
    if len(burst) == 54:
        bursts = [burst]
    for i in range(0, len(burst)-54, 10):
        bursts.append(burst.iloc[i:i+54])
    # bursts.append(burst[0:54])
    # if len(burst)>54:
    #     bursts.append(burst[-54:])

    return bursts

def x_z_combination(predictors: pd.DataFrame) -> pd.DataFrame:
    """
    Combining x and z variables into one xz variable.

    @param predictors: dataframe imported from the predictor files.
    @output: dataframe without seperate x and z variables, leaving only the combined columns.
    """

    predictors = predictors.drop(['Xmean', 'Xvar', 'Xmin', 'Xmax', 'Xmax - Xmin', 'Zmean', 'Zvar', 'Zmin', 'Zmax', 'Zmax - Zmin'], axis=1)
    return predictors


def calc_scores(y_pred: np.ndarray, labels: pd.DataFrame) -> tuple[float, float, float, float, pd.DataFrame]:
    """
    Function to calculate the scores

    @param y_pred: predicted y
    @param labels: labels of the dataset (true labels)
    @return accuracy, recall, precision, f1 (for complete dataset), dataframe of all these parameters per class
    """

    # Calculate the metrics
    accuracy = accuracy_score(labels, y_pred)
    # 'macro' treats all classes equally
    recall = recall_score(labels, y_pred, average='macro')
    precision = precision_score(labels, y_pred, average='macro')
    f1 = f1_score(labels, y_pred, average='macro')
    # average=None calculates scores per class
    recall_all = recall_score(labels, y_pred, average=None)
    precision_all = precision_score(labels, y_pred, average=None)
    f1_all = f1_score(labels, y_pred, average=None)
    scores = pd.DataFrame(
        {'recall': recall_all, 'precision': precision_all, 'f1': f1_all})
    if 'behavior_generalization' in labels.columns:
        proportions = labels['behavior_generalization'].value_counts()/labels.shape[0]
        proportions = proportions.sort_index()
        scores['proportions'] = proportions


    return accuracy, recall, precision, f1, scores

def calculating_unknown_stats(y_pred: np.ndarray, y_prob: np.ndarray, labels: pd.DataFrame, threshold: float = 0.6, output_including_unknown: bool = False) -> tuple:
    """
    function to calculate statistics for probabilities below threshold
    for unlabeled data, y_pred can be given as labels again. The unknown_count variable is then outputting, which predicted behaviors
    were labeled as unknown due to low probability
    @param y_pred: predicted behavior classes
    @param y_prob: prediction probabilities, for each class and instance
    @param labels: true labels/ behavior classes
    @param threshold: threshold to cut off behavior as unknown
    @param output_including_unknown: if this is chosen, the returned Dataframe contains a new class for prediction probability below the threshold
    @return: tuple of three dataframes or vectors: unknown_count (number and proportion of behaviors labeled as unknown),
                labels_prob (only labels with probability above threshold),
                y_pred_prob (only predictions with probabilities above threshold)
    """
    y_pred_prob = y_pred.copy()
    y_pred_prob = y_pred_prob[y_prob > threshold]
    labels_prob = labels[y_prob.ravel() > threshold]
    y_pred_prob_unk = y_pred.copy()
    y_pred_prob_unk[y_prob <= threshold] = 5
    unknown = labels[y_prob.ravel() <= threshold]
    print(unknown.iloc[:, 0])
    print(unknown.iloc[:, 0].value_counts())
    unknown_count = unknown.iloc[:, 0].value_counts().reset_index()
    unknown_count.columns = ['behavior', 'count_unknown']
    complete_count = labels.iloc[:, 0].value_counts().reset_index()
    complete_count.columns = ['behavior', 'count_complete']
    unknown_count = pd.merge(unknown_count, complete_count, on='behavior')
    unknown_count['proportion'] = unknown_count['count_unknown'] / unknown_count['count_complete']
    if output_including_unknown:
        return unknown_count, labels_prob, y_pred_prob_unk
    else:
        return unknown_count, labels_prob, y_pred_prob

def probabilities_to_labels(y_prob_all: pd.DataFrame) -> tuple():
    """
    function to return vectors of prediction with highest probability and the highest probability per test instance
    @param y_prob_all: Dataframe containing all probabilities for the predictions (outout from algo.predict_proba()
    @return: y_pred (label with highest probability per instance), y_prob (highest probability per instance)
    """
    y_pred = np.zeros((len(y_prob_all), 1))
    y_prob = np.zeros(y_pred.shape)
    for i in range(len(y_prob)):
        y_prob[i] = np.amax(y_prob_all[i])
        y_pred[i] = np.argmax(y_prob_all[i])
    return y_pred, y_prob


def parameter_optimization(algorithm: list, predictors: pd.DataFrame, labels: pd.DataFrame,
                           algorithm_name: str) -> tuple:
    """
    Function to conduct parameter optimization
    @param algorithm: list of algorithm to use and parameter grid
    @param predictors: dataframe of all predictors
    @param labels: dataframe of all the labels
    @param algorithm_name: used algorithm (name)
    @return: optimal parameter set
    """
    estimator = algorithm[0]
    print(estimator)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    ex_search = GridSearchCV(
        estimator=estimator, param_grid=algorithm[1], scoring='precision_macro', cv=cv, verbose=2, n_jobs=-1)
    ex_search.fit(predictors, labels)
    # Base filename
    filename_base = f'param_{algorithm_name}'
    filename = f'{filename_base}.csv'
    counter = 1

    # Check if file already exists, and increment counter until a unique filename is found
    while os.path.exists(filename):
        filename = f'{filename_base}_{counter}.csv'
        counter += 1

    # Save the DataFrame to the unique filename
    pd.DataFrame(ex_search.cv_results_).to_csv(filename)
    return ex_search.best_params_

def preparing_datasets_layered(predictors: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping_red = {'resting': 0, 'intermediate energy': 1, 'high energy': 2}
    mapping_red_rev = {v: k for k, v in mapping_red.items()}

    ind_int = np.isin(labels, [1, 2, 3])

    labels_3 = labels.copy()
    labels_3['behavior_generalization'][np.isin(labels_3['behavior_generalization'], [1, 2, 3])] = 1
    labels_3['behavior_generalization'][labels_3['behavior_generalization'] == 4] = 2
    labels_3['behavior_generalization'] = labels_3['behavior_generalization'].map(mapping_red_rev)

    labels_int = labels.copy()
    labels_int = labels_int.iloc[ind_int.ravel()]
    labels_int['behavior_generalization'] = labels_int['behavior_generalization'].map(inverted_mapping)
    pred_int = predictors.copy()
    pred_int = pred_int.iloc[ind_int]

    return labels_3, labels_int, pred_int


def timestamp_to_datetime(acc):
    acc[['date', 'time']] = acc['timestamp'].str.split(' ', expand=True)
    # ACC1[['date', 'time']] = ACC1['timestamp'].str.split(' ', expand=True)
    # ACC1[['date', 'time']] = ACC1['eobs:start-timestamp'].str.split(' ', expand=True)

    # converting to datetime
    acc['time'] = acc['time'].str.slice(0, 8)
    acc = combine_date_time(acc)

    return acc