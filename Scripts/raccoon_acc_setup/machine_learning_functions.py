#!/usr/bin/python3
"""
Filename: machine_learning_functions.py
Author: Eva Reinhardt
Date: 2024-12-09
Version: 1.0
Description: This file contains functions for the machine learning applications
of the work.

functions:
splitting_pred(): splitting pred into predictor dataframe and label dataframe

calc_scores(): accuracy, recall, precision, f1 (for complete dataset), dataframe of all these parameters per class

calculating_unknown_stats(): function to calculate statistics for probabilities below threshold

probabilities_to_labels(): converting a dataframe of prediction probabilities to labels

parameter_optimization(): function to conduct parameter optimization

preparing_datasets_layered(): splitting the datasets into the needed subsets to train the two layers of the layered model

moving_window(): function to apply a moving window on the acc data

ml_90_10(): preparing datasets for the external testing

false_pred_eva(): Function to output dataframes per cluster with the mislabeled behaviors

calc_tables_eva(): Function to create two dataframes. One returns all mislabeled behaviors ordered by cluster, the
        other one ordered by behavior (this one also normalized)

model_predictions(): applying the layered model on data

summarize_result(): function to convert the output from the model into a dataframe that contains all probabilities,
        the resulting prediction, and a threshold of 0.6 probability for known behavior, with every prediction below
        that being labeled as unknown
"""
import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import variables_simplefunctions as sim_func

from sklearn.base import BaseEstimator

from sklearn.preprocessing import LabelEncoder


def splitting_pred(predictions: pd.DataFrame, mapping=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    splitting the pred dataframe into predictors and labels and converting from string classes to integers
    @param predictions: predictions dataframe with columns for each predictor, datetime and labels ('behavior_generalization')
    @param mapping: dict that maps integers to classnames, if not given, classes are mapped in appearing order
    @return: tuple of a dataframe consisting of all predictors and a dataframe of labels
    """

    if mapping is None:
        mapping = {}

    # all columns that are neither behaviour nor datetime are predictors
    filtered_columns = [
        col for col in predictions.columns if 'behavior' not in col and 'datetime' not in col]
    predictors = predictions[filtered_columns]
    labels_col = [
        col for col in predictions.columns if 'behavior_generalization' in col]
    labels = predictions[labels_col]

    # create mapping if it isn't given in the function
    if not mapping:
        classnames2 = list(labels['behavior_generalization'].unique())
        class_map = {classname: idx for idx, classname in enumerate(classnames2)}
    else:
        class_map = mapping

    labels['behavior_generalization'] = labels['behavior_generalization'].map(class_map).astype(int)
    return predictors, labels

def calc_scores(y_pred: np.ndarray, labels: pd.DataFrame) -> tuple[float, float, float, float, pd.DataFrame]:
    """
    Function to calculate the scores

    @param y_pred: predicted y
    @param labels: labels of the dataset (true labels)
    @return accuracy, recall, precision, f1 (for complete dataset), dataframe of all these parameters per class
    """

    # Calculate the metrics
    accuracy = accuracy_score(labels['behavior_generalization'], y_pred)
    # 'macro' treats all classes equally
    recall = recall_score(labels['behavior_generalization'], y_pred, average='macro')
    precision = precision_score(labels['behavior_generalization'], y_pred, average='macro')
    f1 = f1_score(labels['behavior_generalization'], y_pred, average='macro')
    # average=None calculates scores per class
    recall_all = recall_score(labels['behavior_generalization'], y_pred, average=None)
    precision_all = precision_score(labels['behavior_generalization'], y_pred, average=None)
    f1_all = f1_score(labels['behavior_generalization'], y_pred, average=None)
    scores = pd.DataFrame(
        {'recall': recall_all, 'precision': precision_all, 'f1': f1_all})


    proportions = labels['behavior_generalization'].value_counts()/labels.shape[0]
    proportions = proportions.sort_index()
    scores['proportions'] = proportions


    return accuracy, recall, precision, f1, scores

def calculating_unknown_stats(y_pred: np.ndarray, y_prob: np.ndarray, labels: pd.DataFrame, threshold: float = 0.6, output_including_unknown: bool = False) -> tuple:
    """
    function to calculate statistics for probabilities below threshold
    for unlabeled data, y_pred can be given as labels again. The unknown_count variable is then denoting, which predicted behaviors
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

def probabilities_to_labels(y_prob_all: pd.DataFrame) -> tuple:
    """
    function to return vectors of prediction with highest probability and the highest probability per test instance
    @param y_prob_all: Dataframe containing all probabilities for the predictions (outout from algo.predict_proba()
    @return: y_pred (label with highest probability per instance), y_prob (highest probability per instance)
    """
    y_pred = np.zeros((len(y_prob_all), 1))
    y_prob = np.zeros(y_pred.shape)
    y_prob_all = pd.DataFrame(y_prob_all)
    for i in range(len(y_prob)):
        y_prob[i] = np.amax(y_prob_all.iloc[i])
        y_pred[i] = np.argmax(y_prob_all.iloc[i])
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
    filename_base = f'param_{algorithm_name}'
    filename = f'{filename_base}.csv'
    counter = 1

    while os.path.exists(filename):
        filename = f'{filename_base}_{counter}.csv'
        counter += 1

    pd.DataFrame(ex_search.cv_results_).to_csv(filename)
    return ex_search.best_params_

def preparing_datasets_layered(predictors: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    for the layered model, the predictor files are split and classes are converted (to 'resting', 'intermediate energy', 'high energy')
    for the first model. the predictor file for the second model is filtered to only contain the classes 'climbing', 'walking',
    'exploring'
    @param predictors: the predictor dataframe containing all the predictors, aligned with the label dataframe
    @param labels: dataframe containing all the labels, aligned with predictors (column 'behavior_generalization')
    @return: three dataframes: 1. converted labels for the first model ('resting', 'intermediate energy', 'high energy' as int)
                                2. filtered labels for the second model ('climbing', 'walking', 'exploring' as int)
                                3. filtered predictors for second model
    """

    mapping_red = {'resting': 0, 'intermediate energy': 1, 'high energy': 2}
    mapping_red_rev = {v: k for k, v in mapping_red.items()}

    ind_int = np.isin(labels, [1, 2, 3])

    labels_3 = labels.copy()
    labels_3['behavior_generalization'][np.isin(labels_3['behavior_generalization'], [1, 2, 3])] = 1
    labels_3['behavior_generalization'][labels_3['behavior_generalization'] == 4] = 2
    labels_3['behavior_generalization'] = labels_3['behavior_generalization'].map(mapping_red_rev)

    labels_int = labels.copy()
    labels_int = labels_int.iloc[ind_int.ravel()]
    labels_int['behavior_generalization'] = labels_int['behavior_generalization'].map(sim_func.inverted_mapping)
    pred_int = predictors.copy()
    pred_int = pred_int.iloc[ind_int]

    return labels_3, labels_int, pred_int

def moving_window(burst: pd.DataFrame, burst_len_gen: int =54, fs: float=33.3, step: int = None) -> list:
    """
    Function to split a burst into stretches of 54 datapoints

    @param burst: Dataframe containing the complete burst
    @param burst_len_gen: general burst length of the training data set, corresponding to sim_func.GENERAL_FREQ
    @param fs: frequency of the data set
    @param step: either half of the calculated burst length or custom
    @return: list of dataframes containing the splitted burst
    """
    # calculating step length according to given frequency
    if not step:
        step = int((fs* burst_len_gen/sim_func.GENERAL_FREQ)/2)
    burst_len = int(fs* burst_len_gen/sim_func.GENERAL_FREQ)
    bursts = []
    if len(burst) == burst_len:
        bursts = [burst]
    for i in range(0, len(burst)-burst_len, step):
        bursts.append(burst.iloc[i:i+burst_len])

    return bursts

def ml_90_10(algo_f: list, predictors_90: pd.DataFrame, predictors_10: pd.DataFrame, labels_90: pd.DataFrame,
             algo_name_f: str, cv_f: StratifiedKFold) -> tuple[list, list, dict]:
    """
    conducting machine learning with a dataset containing 90% of original dataset (including parameter optimization and cross-validation)
    and testing the model on 10%
    @param algo_f: list of algorithm that should be used and parameter space for parameter optimization
    @param predictors_90: the dataset containing 90% of instances from original data set
    @param predictors_10: dataset containing 10% of instances form original data set (external testing)
    @param labels_90: labels for the 90%
    @param algo_name_f: name of the algorithm
    @param cv_f: cross-validation that should be used
    @return: 1. list of dataframe containing predicted labels and dataframe containing probabilities for the 90% dataset
                2. list of dataframe containing predicted labels and dataframe containing probabilities for the 10% dataset
                3. dict containing the parameters resulting from parameter optimization
    """
    param = dict(parameter_optimization(algo_f, predictors_90, labels_90,
                                            algo_name_f))
    y_prob_all_3_90 = cross_val_predict(algo_f[0].set_params(
        **param), predictors_90,
        labels_90, method='predict_proba', cv=cv_f)

    alg = algo_f[0].set_params(**param)

    y_pred_90, y_prob_90 = probabilities_to_labels(y_prob_all_3_90)

    alg.fit(predictors_90, labels_90)

    y_prob_all_3_10 = alg.predict_proba(predictors_10)

    y_pred_10, y_prob_10 = probabilities_to_labels(y_prob_all_3_10)

    return [y_pred_90, y_prob_90], [y_pred_10, y_prob_10], param

def false_pred_eva(y_pred: pd.DataFrame, y_test: pd.DataFrame, label_behavior: pd.DataFrame,
                   option: str = 'cluster') -> list:
    """
    Function to output dataframes per cluster with the mislabeled behaviors

    @param y_pred: predicted clusters
    @param y_test: true clusters
    @param label_behavior: ungeneralized behaviors
    @param option: what should be returned, mislabeling sorted by cluster (default) or by behavior ('behavior')
    @return list of dataframes containing mislabeled behavior per cluster sorted by count
    """

    # preparing dataframe
    results = pd.DataFrame({'true_label': np.ravel(
        y_test.values), 'pred_label': y_pred.iloc[:, 0], 'index': y_test.index})
    results['is_false'] = results['true_label'] != results['pred_label']
    false_predictions = results[results['is_false']]

    # Get the corresponding test data
    labels_func = pd.DataFrame(label_behavior.iloc[y_test.index], index=y_test.index)
    false_predictions = false_predictions.join(labels_func, on='index')
    results = results.join(labels_func, on='index')
    # Group false predictions by cluster or behavior
    false_predictions_by_cluster = false_predictions.groupby('true_label')[
        'behavior'].apply(list)
    false_predictions_by_behavior = results.groupby(
        'behavior')['pred_label'].apply(list)
    behavior_to_true_label = results.groupby('behavior')['true_label'].first()
    false_predictions_by_behavior_sorted = false_predictions_by_behavior.loc[behavior_to_true_label.sort_values(
    ).index]
    if option == 'cluster':
        return false_predictions_by_cluster
    else:
        return false_predictions_by_behavior_sorted

def calc_tables_eva(pred: pd.DataFrame, all_pred_test: pd.DataFrame, all_lab_test: pd.DataFrame, classes: list) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Function to create two dataframes. One returns all mislabeled behaviors ordered by cluster,
    the other one ordered by behavior (this one also normalized)

    @param all_pred_test: predicted labels
    @param all_lab_test: true labels
    @param classes:
    @return list of three dataframes: mislabeled behaviors ordered by cluster, behavior and behavior but normalized
    """
    class_order = sim_func.MAPPING.keys()
    # Now generate the mislabeled table with combined predictions from all folds
    false_predictions = false_pred_eva(all_pred_test,
        all_lab_test, pred['behavior'])
    mislabeled = pd.DataFrame(
        np.zeros((1, len(classes))), columns=classes)
    lab_test_indeces = list(all_lab_test.index)
    test = pred.iloc[lab_test_indeces]

    test_list = test['behavior'].tolist()
    counts= {}
    for item in test_list:
        counts[item] = counts.get(item,0)+1
    label_counts_df = pd.DataFrame.from_dict(counts, orient='index', columns=['TotalCount'])
    label_counts_df.index.name = 'Instance'
    label_counts_df.reset_index(inplace=True)
    print(label_counts_df['TotalCount'].sum())

    counts_list = []
    for f in range(len(false_predictions)):
        counts = pd.DataFrame(false_predictions[f]).value_counts()
        counts_df = counts.reset_index()
        counts_df.columns = ['Instance', 'Count']
        counts_df = pd.merge(counts_df, label_counts_df, on='Instance')
        string = '\n'.join(
            [f'{row["Instance"]}: {row["Count"]} / {row["TotalCount"]}' for _, row in counts_df.iterrows()])
        string = string.replace("'", "").replace(
            ",", "").replace("(", "").replace(")", "")
        mislabeled[classes[f]] = string
        counts_list.append(counts_df)
    # second false predictions table
    false2 = pd.DataFrame(false_pred_eva(all_pred_test, pd.DataFrame(
        all_lab_test), pred['behavior'], option='behavior'))
    columns_table = false2.index
    mislabeled2 = pd.DataFrame(
        np.zeros((len(classes), len(columns_table))), columns=columns_table, index=class_order)
    for ins in columns_table:
        f = pd.Series(false2.loc[ins].iloc[0])
        counts = pd.DataFrame(f).value_counts()
        for ind in counts.index:
            print(ind)
            mislabeled2.loc[ind[0]][ins] = counts[ind[0]]
            mislabeled2.index = classes
    mislabeled2_norm = mislabeled2.copy()
    mislabeled2_norm = mislabeled2_norm.div(mislabeled2_norm.sum(axis=0), axis=1)
    return mislabeled, mislabeled2, mislabeled2_norm


def model_predictions(pred_w: pd.DataFrame, m1: BaseEstimator, m2: BaseEstimator, m_comp: BaseEstimator,
                      label_encoder: LabelEncoder = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    running the layered model

    @param pred_w: predictors in dataframe
    @param m1: model for first layer
    @param m2: model for second layer
    @param m_comp: simple model without layers
    @param label_encoder: LabelEncoder instance for the one layer model (SVM)
    @return: probability dataframes for the combined model and the simple model
    """
    times = pred_w['datetime']
    predictors_test = pred_w.drop(['datetime'], axis=1)
    pred1_labels = list(m1.get_booster().feature_names)
    y_prob_3_all_func = m1.predict_proba(predictors_test[list(m1.get_booster().feature_names)])
    y_prob_3_all_func = pd.DataFrame(y_prob_3_all_func)

    y_prob_3_all_func = summarize_result(y_prob_3_all_func, sim_func.MAPPING_3_INVERSE)
    predictors_test.index = y_prob_3_all_func.index

    if (y_prob_3_all_func['pred_incl_unkn'] == 'intermediate energy').any():
        predictors_test_2 = predictors_test.loc[y_prob_3_all_func['pred_incl_unkn'] == 'intermediate energy']
        pred_labels = list(m2.get_booster().feature_names)
        y_prob_int_all = m2.predict_proba(predictors_test_2[list(m2.get_booster().feature_names)])
        y_prob_int_all = pd.DataFrame(y_prob_int_all)
        #y_prob_int_all = y_prob_int_all.rename(columns=sim_func.MAPPING_INT_INVERSE)

        y_prob_int_all = summarize_result(y_prob_int_all, sim_func.MAPPING_INT_INVERSE)

        y_prob_int_all.index = predictors_test_2.index

        # y_prob_int_all = adding_time_to_df(y_prob_int_all, times.values[predictors_test_2.index])
        # y_prob_3_all = adding_time_to_df(y_prob_3_all, times.values)

        not_unknown_int = y_prob_int_all['pred_incl_unkn'] != 'unknown'
        indeces = y_prob_int_all[not_unknown_int].index

        y_prob_3_all_func['pred_incl_unkn'].iloc[indeces] = y_prob_int_all['pred'][indeces]
        y_prob_int_all = y_prob_int_all.drop(['pred_incl_unkn'], axis=1)
        y_prob_3_all_func = y_prob_3_all_func.merge(y_prob_int_all, left_index=True, right_index=True, how='outer')
        y_prob_3_all_func = sim_func.adding_time_to_df(y_prob_3_all_func, times.values)
    else:
        y_prob_3_all_func = sim_func.adding_time_to_df(y_prob_3_all_func, times.values)

    y_prob_comp_all_func = m_comp.predict_proba(predictors_test[list(m1.get_booster().feature_names)])
    y_prob_comp_all_func = summarize_result(y_prob_comp_all_func, dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_)))
    y_prob_comp_all_func = sim_func.adding_time_to_df(y_prob_comp_all_func, times.values)
    return y_prob_3_all_func, y_prob_comp_all_func


def summarize_result(y_prob_all: pd.DataFrame, class_mapping: dict) -> pd.DataFrame:
    """
    function to convert the output from the model into a dataframe that contains all probabilities, the resulting prediction,
    and a threshold of 0.6 probability for known behavior, with every prediction below that being labeled as unknown
    @param y_prob_all: output from the model, probabilities for all classes per instance
    @param class_mapping: mapping used to convert classes to integers
    @return: dataframe containing probabilities of all classes and columns 'pred', 'pred-incl_unkn_'
    """

    y_pred_3, y_prob_3 = probabilities_to_labels(y_prob_all)
    y_prob_3_all_func = pd.DataFrame(y_prob_all)
    y_prob_3_all_func = y_prob_3_all_func.rename(columns = class_mapping)
    y_pred_3 = pd.DataFrame(y_pred_3, columns=['prediction'])
    y_pred_3['prediction'] = y_pred_3['prediction'].map(class_mapping)

    unknown_count, labels_prob, y_pred_prob = calculating_unknown_stats(np.array(y_pred_3), y_prob_3, y_pred_3,
                                                                            output_including_unknown=True)

    y_prob_3_all_func['pred'] = y_pred_3['prediction']
    y_prob_3_all_func['pred_incl_unkn'] = labels_prob
    y_prob_3_all_func.fillna('unknown', inplace=True)

    return y_prob_3_all_func