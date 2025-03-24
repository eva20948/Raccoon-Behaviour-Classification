# Raccoon-Behaviour-Classification
Scripts and Library used in the master's thesis by Eva Reinhardt

The repository is structured according to the different tasks/sections of the master's thesis: 
* the subfolder **Analysis of data sets** contains the files: 
  + *data_analysis_important_features_violinplots.py*: used for visualization of features per original behaviour, gives first impression of possible clustering
  + *data_statistics.py*: conducts statistical analysis of unlabelled as well as labelled data sets, including behaviour proportions (after generalization), number of data points, and more
  + *matching_behaviour_to_acc_data_violinplots.py*: initial analysis when not enough meta data is available to match the observed behaviour to the respective logger number
 
* the subfolder **Feature Calculation and Selection** contains the files:
  + *creating_feature_files.py*: converting the raw acceleration data to feature/predictor files
  + *feature_correlation_and_reduction.py*: analysing the correlation and connectedness of the calculated features in order to reduce the feature space
  + *sfs_feature_importance.py*: conducting Sequential Feature Selection and feature importance using a Random Forest
 
* the subfolder **Determining behaviour classes** contains the file:
  + *determining_behaviour_classes.py*: using different clustering algorithms to achieve behaviour groupings
 
* the subfolder **Optimizing behaviour classification** contains the files:
  + *building_layered_model.py*: parameter opimization for the layered model
  + *optimizing_simple_model.py*: parameter optimization for the simple model
 
* the subfolder **Transferring to wild data and further analysis** contains the files:
  + *gps_behaviour_plotting.py*: creates maps combining gps data and behaviour predictions
  + *hfi_data_use.py*: outputs mean, max, min, std hfi values for roaming area per individual
  + *hfi_mean.txt*: file containing the output from *hfi_data_use.py*
  + *model_for_wild_data.py*: uses the obtained models on unlabelled data, gives out hourly contingents and result files with prediction probabilities per behaviour per burst
  + *results_wild_data_summary.py*: file used for several analysis of the behaviour predictions
 
  * the subfolder **raccoon_acc_setup** contains the library developped in the course of this project. Subfiles are:
    + *gui_functions.py*: containing all functions related to graphical user interfaces
    + *importing_raw_data.py*: functions for importing raw data in different formats as well as behaviour observation files
    + *machine_learning_functions.py*: functions for data set preparation as well as result visualization
    + *plot_functions.py*: different plotting functions
    + *predictor_calculation.py*: calculating the different features and format standardization
    + *variables_simplefunctions.py*: definition of important variables, small general functions
