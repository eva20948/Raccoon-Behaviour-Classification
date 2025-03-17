"""
Filename: gui_functions.py
Author: Eva Reinhardt
Date: 2024-12-09
Version: 1.0
Description: This file contains GUI functions.

functions:
open_filedialog(): open filedialog to select file

choose_option(): choosing an option

choose_multiple_options(): choosing from multiple options

(options_inge())

save_pred(): dialog to choose how to save files.

"""

import tkinter as tk
from tkinter import filedialog
import pandas as pd

def open_file_dialog(which_file: str) -> tuple[str, ...]:
    """
    Open a file dialog to select a file.

    @param which_file:  The title of the dialog window, specifying which data
                        is needed at this point.
    @return: the selected file path(s), a list of strings or a single string.
    """
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilenames(
        title=which_file,
        filetypes=(("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*"))
    )

    return file_path





def choose_option(options: list[str], title: str = "Choose an Option") -> str:
    """
    Present dialog window to choose an option.

    @param options: List of options to choose from.
    @param title: Title of the dialog window.
    @return: Selected option.
    """
    root = tk.Tk()
    root.withdraw()

    # save selected option, destroy window after saving
    def select_option(op):
        nonlocal selected_option
        selected_option = op
        dialog.destroy()

    dialog = tk.Toplevel(root)
    dialog.title(title)

    tk.Label(dialog, text="Please choose an option:").pack(pady=10)

    for option in options:
        tk.Button(dialog, text=option, command=lambda opt=option: select_option(opt)).pack(pady=5)


    selected_option = None

    root.wait_window(dialog)

    return selected_option


def choose_multiple_options(options: list[str], title: str = "Choose Options") -> list[str]:
    """
    Present dialog window to choose multiple options.

    @param options: List of options to choose from.
    @param title: Title of the dialog window.
    @return: List of selected options.
    """
    root = tk.Tk()
    root.withdraw()

    selected_options = []

    dialog = tk.Toplevel(root)
    dialog.title(title)

    tk.Label(dialog, text="Please choose options:").pack(pady=10)

    variables = {}
    for option in options:
        var = tk.IntVar()
        chk = tk.Checkbutton(dialog, text=option, variable=var)
        chk.pack(anchor='w')
        variables[option] = var

    def select_options():
        print("OK button clicked")
        for op, v in variables.items():
            print(f"{op}: {v.get()}")
            if v.get():
                selected_options.append(option)
        print("Selected options:", selected_options)
        dialog.destroy()
        root.quit()

    ok_button = tk.Button(dialog, text="OK", command=select_options)
    ok_button.pack(pady=10)

    root.mainloop()

    return selected_options


def save_pred(data: pd.DataFrame):
    """
    Saving the predictor data to a csv file.

    @param data: Dataframe containing the predictor data, datetime column and behavior data
    """
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.asksaveasfilename(title="Save as")
    data.to_csv(file_path, index=False)


