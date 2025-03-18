import os

# Check if we are running in an HPC environment and skip GUI imports
if "SLURM_JOB_ID" not in os.environ:  # SLURM_JOB_ID is typically set in HPC environments
    from .gui_functions import open_file_dialog

    from .gui_functions import open_file_dialog
    from .gui_functions import choose_option
    from .gui_functions import choose_multiple_options
    from .gui_functions import save_pred
from .predictor_calculation import calculate_features
from .predictor_calculation import calculate_pred
from .predictor_calculation import create_pred_complete
from .variables_simplefunctions import combine_date_time
from .variables_simplefunctions import x_z_combination
from .variables_simplefunctions import timestamp_to_datetime
from .variables_simplefunctions import remove_outliers
from .variables_simplefunctions import adding_time_to_df
from .generate_init_file import generate_init_file
from .plot_functions import plot_burst
from .plot_functions import plot_clustering_results
from .plot_functions import pca_eval
from .plot_functions import confusion_matrices_layered
from .plot_functions import visualise_predictions_ml
from .plot_functions import plotting_proportions_predictions
from .plot_functions import output_hourly_contingents
from .plot_functions import plotting_vps
from .importing_raw_data import import_eobs
from .importing_raw_data import split_burst
from .importing_raw_data import import_acc_data
from .importing_raw_data import import_beh_peter
from .importing_raw_data import import_beh_domi
from .importing_raw_data import merge_domi
from .importing_raw_data import convert_beh
from .importing_raw_data import behavior_combi_domi
from .machine_learning_functions import splitting_pred
from .machine_learning_functions import calc_scores
from .machine_learning_functions import calculating_unknown_stats
from .machine_learning_functions import probabilities_to_labels
from .machine_learning_functions import parameter_optimization
from .machine_learning_functions import preparing_datasets_layered
from .machine_learning_functions import moving_window
from .machine_learning_functions import ml_90_10
from .machine_learning_functions import false_pred_eva
from .machine_learning_functions import calc_tables_eva
from .machine_learning_functions import model_predictions
from .machine_learning_functions import summarize_result