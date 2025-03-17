"""
Filename: violinplots.py
Author: Eva Reinhardt
Date: 2024-12-09
Version: 1.0
Description: This file is used to plot violinplots of different predictors of different individuals
in order to find out, which acc file belongs to which behavior file.
"""
from raccoon_acc_setup import variables_simplefunctions as sim_func
from raccoon_acc_setup import importing_raw_data as im_raw
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tkinter import filedialog
import numpy as np


if __name__ == '__main__':
    filepaths = sim_func.open_file_dialog("Select a file or files - predictors")

    if 'Inge' in filepaths[0]:
        # loading behavior data and renaming columns for consistency
        beh = im_raw.import_beh_inge('/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Msc Inge Buchholz/behavior_all.csv', 'Both')
        beh = beh.rename(columns={'Verhalten Emma':'behavior_Emma', 'Verhalten Susi':'behavior_Susi'})

        # setting needed variables for the proceedings, behaviors and animals is set
        # options can be changed: denotes the behaviors plotted in the violinplot
        # alternative for options also available further down
        behaviors=['behavior_Emma', 'behavior_Susi']
        animals = ['Emma', 'Susi']
        options=['running+foraging', 'sitting+foraging', 'running on door', 'sitting', 'sitting upright']

        # count of all examined behaviors per animal, used for coverage calculation
        all_beh_1 = len(beh[behaviors[0]].dropna())
        all_beh_2 = len(beh[behaviors[1]].dropna())

    elif 'Dominique' in filepaths[0]:
        # loading behavior data
        beh = im_raw.import_beh_domi('/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten_Waschbär Msc Dominique-IZW/Beobachtungsdaten/Verhaltensweisen_concatenated.csv')
        
        # setting needed variables for the proceedings, behaviors and animals is set
        # options can be changed: denotes the behaviors plotted in the violinplot
        # alternative for options also available further down
        behaviors = ['behavior_Lisa', 'behavior_Ottilie']
        animals = ['Lisa', 'Ottilie']
        #options = ['laufen+galopp', 'liegen', 'ruhen', 'galopp', 'gehen']


        # count of all examined behaviors per animal, used for coverage calculation
        all_beh_1 = len(beh[behaviors[0]].dropna())
        all_beh_2 = len(beh[behaviors[1]].dropna())

    elif 'Peter' in filepaths[0]:
        beh_wil = im_raw.import_beh_peter('/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/behavior/tag W1/tag5140/alle.csv')
        beh_em = im_raw.import_beh_peter('/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/behavior/tag 5033 E/Observations/alle.csv')
        beh_c = im_raw.import_beh_peter('/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/behavior/tag 5334 C/Observations/alle.csv')



    # plot's initiation, number of subplots = number of chosen files 
    fig, axes = plt.subplots(1, len(filepaths), figsize=(15, 5), sharey=True)

    # necessary if filepaths only contains one file 
    if len(filepaths) == 1:
        axes = [axes]

    # building the subplots per file 
    for i, filepath in enumerate(filepaths):
        # reading predictor files, filtering the necessary columns 
        pred = pd.read_csv(filepath)
        pred = pred.filter(regex='Ndyn|Xmean|Ymean |Verhalten|behavior')

        if 'Inge' in filepath:
            # abbreviations of behaviors are substituted 
            pred[behaviors] = im_raw.convert_beh(pred[behaviors], 'Inge')

        if 'Inge' in filepath or 'Dominique' in filepath: 
            ## the dataframe is reshaped to not contain a column for each animal's behavior, 
            # but a column which contains the animal's name and another one containing the behavior.

            pred1 = pred[['Ndyn', behaviors[0]]]
            pred1['Tier'] = animals[0]
            pred1.columns = ['Ndyn', 'behavior', 'animal']
            pred1['behavior'] = pred1['behavior'].replace('-', np.nan)

            # measurement: number of behavior observations for which a timestamp in the predictor dataset exists
            burst_1 = len(pred1['behavior'].dropna())
            print(burst_1)
            print(pred.shape)
            
            ## the dataframe is reshaped to not contain a column for each animal's behavior, 
            # but a column which contains the animal's name and another one containing the behavior.        
            pred2 = pred[['Ndyn', behaviors[1]]]
            pred2['Tier'] = animals[1]
            pred2.columns = ['Ndyn', 'behavior', 'animal']
            pred2['behavior'] = pred2['behavior'].replace('-', np.nan)

            # measurement: number of behavior observations for which a timestamp in the predictor dataset exists
            burst_2 = len(pred2['behavior'].dropna())

            # coverage calculation: assigned behavior observations / all behavior observations
            cov_1 = burst_1/ all_beh_1
            cov_2 = burst_2/ all_beh_2


            # concatenating the seperated tables
            pred = pd.concat([pred1, pred2], ignore_index=True)
            pred = pred.dropna()
            

            # alternative to defined options list: the 8 most common options in the dataset 
            if len(options)==0:
                count = pred['behavior'].value_counts()
                count = count.reset_index()
                options = list(count['behavior'].head(8))
                
            # filtering the dataset to only contain behavior from the options list
            pred = pred[pred['behavior'].isin(options)]


            # plotting the violinplot: 'behavior' are the categories (x), the 'Ndyn' is the y-variable, 
            # the violins are splitted (one side and color denotes one animal, other side and color the other), 
            # quartiles are shown 
            sns.violinplot(x=pred['behavior'], y = pred['Ndyn'], hue = pred['animal'], split = True, inner = 'quart',ax=axes[i])

            # title is the filepath for better understanding and the coverage values
            axes[i].set_title(f'{os.path.basename(filepath)}; Coverage {animals[0]}: {cov_1:.2f}, {animals[1]}: {cov_2:.2f}')
                                
            # Rotate the x-axis labels to be vertical
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=90)

        elif 'Peter' in filepath: 
            # abbreviations of behaviors are substituted 
            pred = im_raw.convert_beh(pred, 'Peter')

            if 'Carlo' in filepath:
                beh = im_raw.import_beh_peter('/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/behavior/tag 5334 C/Observations/alle.csv')
            elif 'Emma' in filepath:
                beh = im_raw.import_beh_peter('/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/behavior/tag 5033 E/Observations/alle.csv')
            elif '5140' in filepath: 
                beh = im_raw.import_beh_peter('/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/behavior/tag W1/tag5140/alle.csv')
            elif '7073' in filepath: 
                beh = im_raw.import_beh_peter('/media/eva/eva-reinhar/your folders/01 raw data/Labeldaten Waschbär Bsc Peter Geiger-IZW/behavior/tag W1/tag7073/alle.csv')
        
            all_beh_1 = len(beh['behavior'].replace('-', np.nan).dropna())


            pred = pred[['Ymean', 'behavior']]

            # measurement: number of behavior observations for which a timestamp in the predictor dataset exists
            burst_1 = len(pred['behavior'].dropna())
            print(burst_1)
            print(pred.shape)

            # coverage calculation: assigned behavior observations / all behavior observations
            cov_1 = burst_1/ all_beh_1
            

            # alternative to defined options list: the 8 most common options in the dataset 
            count = pred['behavior'].value_counts()
            count = count.reset_index()
            options = list(count['behavior'].head(8))
                
            # filtering the dataset to only contain behavior from the options list
            pred = pred[pred['behavior'].isin(options)]


            # plotting the violinplot: 'behavior' are the categories (x), the 'Ndyn' is the y-variable, 
            # the violins are splitted (one side and color denotes one animal, other side and color the other), 
            # quartiles are shown 
            sns.violinplot(x=pred['behavior'], y = pred['Ymean'], inner = 'quart',ax=axes[i])

            # title is the filepath for better understanding and the coverage values
            axes[i].set_title(f'{os.path.basename(filepath)}; Coverage: {cov_1:.2f}')
                                
            # Rotate the x-axis labels to be vertical
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=90)


    # plot the plot 
    plt.tight_layout()
    # plt.show()
    file_path = filedialog.asksaveasfilename(title="Save as")
    plt.savefig(file_path)
        
        