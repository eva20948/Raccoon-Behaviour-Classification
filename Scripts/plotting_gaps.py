"""
Filename: plotting_gaps.py
Author: Eva Reinhardt
Date: 2024-02-20
Version: 1.0
Description: This file is used to plot larger gaps in the acc files.

"""

import pandas as pd
import matplotlib.pyplot as plt
from raccoon_acc_setup import variables_simplefunctions as sim_func
import os
import re


filepaths_class = [sim_func.IMPORT_PATH_CLASS + f for f in os.listdir(sim_func.IMPORT_PATH_CLASS) if
                   os.path.isfile(os.path.join(sim_func.IMPORT_PATH_CLASS, f)) and '.csv' in f and
                   'predictions_mw_layered' in f]

for filepath in filepaths_class:
    logger = match = re.search(r"\d{4}", filepath)[0]
    output_filepath = sim_func.IMPORT_PATH_CLASS+'gaps/'+logger+'_gap_visualization.jpg'

    data = pd.read_csv(filepath, sep=',')
    data['datetime'] = pd.to_datetime(data['datetime'], format = "mixed")

    data['gap_size'] = data['datetime'].diff().dt.total_seconds() / 60

    plt.figure(figsize=(20, 4))
    plt.plot(data['datetime'], [1] * len(data), marker='o', linestyle='-', label="Data points")

    for i in range(1, len(data)):
        if data['gap_size'].iloc[i] > 30:
            plt.axvspan(data['datetime'].iloc[i-1], data['datetime'].iloc[i], color='red', alpha=0.3)

    plt.xlabel('Datetime')
    plt.ylabel('Presence of Data')
    plt.title(logger)
    plt.legend()
    plt.grid()

    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')