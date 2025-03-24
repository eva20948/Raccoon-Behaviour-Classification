#!/usr/bin/python3
"""
Filename: building_layered_model.py
Author: Eva Reinhardt
Date: 2024-12-09
Version: 1.0
Description: evaluating the layered model's performance using parameter optimization and
cross-validation. Output are confusion matrices for both layers of the model for 4 different ml
algorithms.

"""

import pandas as pd
import numpy as np

from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize, StandardScaler

import xgboost as xgb

from raccoon_acc_setup import importing_raw_data as im_raw
from raccoon_acc_setup import machine_learning_functions as mlf
from raccoon_acc_setup import predictor_calculation as pred_cal
from raccoon_acc_setup import plot_functions as plt_func
from raccoon_acc_setup import variables_simplefunctions as sim_func

mapping = {'resting': 0, 'exploring': 1, 'walking': 2, 'climbing': 3, 'high energy': 4}
inverted_mapping = {v: k for k, v in mapping.items()}
class_order = ['resting', 'exploring', 'walking', 'climbing', 'high energy']

xgb_all = xgb.XGBClassifier(colsample_bytree=0.6, gamma=0.5,
                            learning_rate=0.2, max_depth=6,
                            n_estimators=30, subsample=0.8)
xgb_part = xgb.XGBClassifier(colsample_bytree=0.6, gamma=0.5,
                             learning_rate=0.2, max_depth=6,
                             n_estimators=30, subsample=0.8)

# ml_algs = {
#     # 'RandomForest': RandomForestClassifier(class_weight='balanced_subsample', random_state=42, bootstrap= True, max_depth= 20, max_features= 'sqrt', min_samples_leaf= 1, min_samples_split= 2, n_estimators= 200),
#     'RandomForest_low': RandomForestClassifier(bootstrap=True, class_weight='balanced_subsample', max_depth=4,
#                                                max_features='sqrt', min_samples_leaf=3, min_samples_split=5,
#                                                n_estimators=30),
#     #
#     # 'SupportVectorMachine_linearkernel': svm.SVC(probability=True, kernel="linear", C=80, gamma='scale'),
#     # 'SupportVectorMachine_rbfkernel': svm.SVC(probability=True, kernel="rbf", gamma='scale', C=30),
#     # 'XGBoost_low': xgb.XGBClassifier(colsample_bytree=0.8, gamma=1,
#     #                                  learning_rate=0.2, max_depth=4,
#     #                                  n_estimators=20, subsample=0.8, min_child_weight = 10 )
# }
# ml_algs = {
#     'RandomForest': [RandomForestClassifier(), {
#         'n_estimators': [20, 25, 30],  # Number of trees
#         'max_depth': [2, 4],  # Maximum depth of a tree
#         'min_samples_split': [5, 7, 10],  # Minimum samples required to split
#         'min_samples_leaf': [2, 3, 4],  # Minimum samples required at a leaf node
#         'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider at each split
#         'bootstrap': [True],  # Use bootstrap sampling
#         'class_weight': ['balanced_subsample']  # Adjust class weights
#         #
#         # 'n_estimators': [20],  # Number of trees
#         # 'max_depth': [2],  # Maximum depth of a tree
#         # 'min_samples_split': [5],  # Minimum samples required to split
#         # 'min_samples_leaf': [2],  # Minimum samples required at a leaf node
#         # 'max_features': [3],  # Number of features to consider at each split
#         # 'bootstrap': [True],  # Use bootstrap sampling
#         # 'class_weight': ['balanced_subsample']  # Adjust class weights
#         #     # #
#     }]
#     ,
#     'SupportVectorMachine_linearkernel': [svm.SVC(probability=True, kernel="linear"), {
#         'C': [0.1, 1, 10, 80],
#         'gamma': ['scale', 'auto']  # Kernel coefficient
#     }],
#     'SupportVectorMachine_rbfkernel': [svm.SVC(probability=True, kernel="rbf"), {
#         'C': [0.1, 10, 30, 80],
#         'gamma': ['scale', 'auto']  # Kernel coefficient
#     }],
#     # 'SupportVectorMachine_polykernel': [svm.SVC(probability=True, kernel="poly"), {
#     #     'C': [0.01, 1, 10, 80],  # uniform(0.1, 10),  # Regularization parameter
#     #     'gamma': ['scale', 'auto'],  # Kernel coefficient
#     #     'degree': [2, 3, 4, 5],  # Degree for 'poly' kernel
#     #     # Independent term in kernel function for poly and sigmoid
#     #     'coef0': [0.1, 0.5, 0.8]
#     # }],
#     # 'SupportVectorMachine_sigmoid': [svm.SVC(probability=True, kernel="sigmoid"), {
#     #     'C': [0.01, 1, 10],  # uniform(0.1, 10),  # Regularization parameter
#     #     'gamma': ['scale', 'auto'],  # Kernel coefficient
#     #     # Independent term in kernel function for poly and sigmoid
#     #     # 'coef0': uniform(0, 1)
#     #     'coef0': [0.1, 0.5, 0.8]
#     # }],
#     'XGBoost': [xgb.XGBClassifier(), {
#         'n_estimators': [10, 15, 20],  # Number of trees
#         'max_depth': [2, 4],  # Maximum depth of a tree
#         'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage
#         'subsample': [0.6, 0.8, 1.0],  # Fraction of samples used per tree
#         'colsample_bytree': [0.6, 0.8, 1.0],  # Fraction of features used per tree
#         'gamma': [0, 0.5, 1, 3],  # Minimum loss reduction
#         'min_child_weight': [2, 5, 10],  # Minimum sum of instance weight
#
#         # 'learning_rate': [0.2],  # Step size shrinkage
#         # 'subsample': [0.6],  # Fraction of samples used per tree
#         # 'colsample_bytree': [0.6],  # Fraction of features used per tree
#         # 'gamma': [0.5],  # Minimum loss reduction
#         # 'min_child_weight': [2],  # Minimum sum of instance weight
#         #
#         # 'n_estimators': [10],  # Number of trees
#         # 'max_depth': [2]  # Maximum depth of a tree
#
#     }]
# }



ml_algs = {
    # 'RandomForest': [ImbPipeline([('resampler', SMOTE()), ('classifier', RandomForestClassifier())]), {
    #     'resampler': [SMOTE(sampling_strategy='minority', k_neighbors=5),
    #         BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5, kind='borderline-1'),
    #         ADASYN(sampling_strategy='minority', n_neighbors=5),
    #         RandomUnderSampler(random_state=42)],
    #     'classifier__n_estimators': [20, 25, 30],
    #     'classifier__max_depth': [2, 4],
    #     'classifier__min_samples_split': [5, 7, 10],
    #     'classifier__min_samples_leaf': [2, 3, 4],
    #     'classifier__max_features': ['auto', 'sqrt', 'log2'],
    #     'classifier__bootstrap': [True],
    #     'classifier__class_weight': ['balanced_subsample']
    # }],
    'SupportVectorMachine_linearkernel': [ImbPipeline([('resampler', SMOTE()), ('scaling', StandardScaler()), ('classifier', svm.SVC(probability=True, kernel="linear"))
                                                       ]),
    {
        'resampler': [SMOTE(sampling_strategy='minority', k_neighbors=5),
            BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5, kind='borderline-1'),
            ADASYN(sampling_strategy='minority', n_neighbors=5),
            RandomUnderSampler(random_state=42)],
        'scaling': [StandardScaler()],
        'classifier__C': [0.1, 1, 10, 80],
        'classifier__gamma': ['scale', 'auto']
    }],
    'SupportVectorMachine_rbfkernel': [ImbPipeline([('resampler', SMOTE()), ('scaling', StandardScaler()), ('classifier', svm.SVC(probability=True, kernel="rbf"))]),  {
        'resampler': [SMOTE(sampling_strategy='minority', k_neighbors=5),
        BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5, kind='borderline-1'),
        ADASYN(sampling_strategy='minority', n_neighbors=5),
        RandomUnderSampler(random_state=42)],
        'scaling': [StandardScaler()],
        'classifier__C': [0.1, 10, 30, 80],
        'classifier__gamma': ['scale', 'auto']
    }],
    # 'SupportVectorMachine_polykernel': [ImbPipeline([('resampler', SMOTE()), ('scaling', StandardScaler()),
    #                                                 ('classifier', svm.SVC(probability=True, kernel="poly"))]), {
    #        'resampler': [SMOTE(sampling_strategy='minority', k_neighbors=5),
    #                      BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5,
    #                                      kind='borderline-1'),
    #                      ADASYN(sampling_strategy='minority', n_neighbors=5),
    #                      RandomUnderSampler(random_state=42)],
    #        'scaling': [StandardScaler()],
    #         'classifier__C': [0.01, 1, 10, 80],  # uniform(0.1, 10),  # Regularization parameter
    #         'classifier__gamma': ['scale', 'auto'],  # Kernel coefficient
    #         'classifier__degree': [2, 3, 4, 5],  # Degree for 'poly' kernel
    #         # Independent term in kernel function for poly and sigmoid
    #         'classifier__coef0': [0.1, 0.5, 0.8]
    #                                    }]
    # #,
    'XGBoost': [ImbPipeline([('resampler', SMOTE()), ('classifier', xgb.XGBClassifier())]), {
        'resampler': [SMOTE(sampling_strategy='minority', k_neighbors=5),
        BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5, kind='borderline-1'),
        ADASYN(sampling_strategy='minority', n_neighbors=5),
        RandomUnderSampler(random_state=42)],
        'classifier__n_estimators': [10, 15, 20],
        'classifier__max_depth': [2, 4],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__subsample': [0.6, 0.8, 1.0],
        'classifier__colsample_bytree': [0.6, 0.8, 1.0],
        'classifier__gamma': [0, 0.5, 1, 3],
        'classifier__min_child_weight': [2, 5, 10]
    }]
}

filepaths_peter = [
    sim_func.IMPORT_PARAMETERS['Peter']['filepath_pred']]
filepaths_domi = [
    sim_func.IMPORT_PARAMETERS['Dominique']['filepath_pred']]

filepaths = [filepaths_peter, filepaths_domi]

if __name__ == "__main__":


    pred = pred_cal.create_pred_complete(filepaths, reduced_features=False)

    pred = im_raw.convert_beh(pred, 'generalization', 'generalization3')
    pred = im_raw.convert_beh(pred, 'translation')
    predictors, labels = mlf.splitting_pred(pred, mapping=mapping)

    mapping_red = {'resting': 0, 'intermediate energy': 1, 'high energy': 2}
    mapping_red_rev = {v: k for k, v in mapping_red.items()}

    ind_int = np.isin(labels, [1, 2, 3])

    print(pred)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    train_index, test_index = next(cv.split(predictors, labels))

    ind_90 = sorted(train_index)
    ind_10 = sorted(test_index)

    labels_3 = labels.copy()
    labels_3['behavior_generalization'][np.isin(labels_3['behavior_generalization'], [1, 2, 3])] = 1
    labels_3['behavior_generalization'][labels_3['behavior_generalization'] == 4] = 2

    labels_int = labels.copy()
    labels_int = labels_int.iloc[ind_int.ravel()] - 1
    pred_int = predictors.copy()
    pred_int = pred_int.iloc[ind_int]

    int_ind_90, int_ind_10 = next(cv.split(pred_int, labels_int))
    int_ind_90 = sorted(int_ind_90)
    int_ind_10 = sorted(int_ind_10)

    option = 'testing performance'
    if option == 'parameter optimization':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        plt_func.confusion_matrices_layered(predictors, pred_int, labels_int, labels_3,
                                            ind_90, ind_10, int_ind_90, int_ind_10,
                                            ml_algs, cv, mapping_red_rev)

    elif option == 'testing performance':

        classes = class_order
        classes.append('intermediate energy')

        models1 = [
            xgb.XGBClassifier(colsample_bytree= 0.8, gamma= 0.5, learning_rate= 0.2, max_depth= 4,
                               min_child_weight= 2, n_estimators= 15, subsample= 0.8),
              xgb.XGBClassifier(colsample_bytree= 1.0, gamma= 0.5, learning_rate= 0.2, max_depth= 4,
                                min_child_weight= 2, n_estimators= 20, subsample= 1.0), # reduced features
              svm.SVC(kernel = 'rbf', probability=True, C= 80, gamma= 'auto'), # reduced features
            xgb.XGBClassifier(colsample_bytree=1.0, gamma=0.5, learning_rate=0.2, max_depth=4,
                              min_child_weight=2, n_estimators=20, subsample=1.0)
        ]

        models2 = [
            xgb.XGBClassifier(colsample_bytree=1.0, gamma=1, learning_rate=0.2, max_depth=4,
                                             min_child_weight= 10, n_estimators= 20, subsample= 0.6),
            xgb.XGBClassifier(colsample_bytree=1.0, gamma=1, learning_rate=0.2, max_depth=4,
                              min_child_weight=10, n_estimators=20, subsample=0.7),
            svm.SVC(kernel = 'rbf', C= 80, gamma= 'auto', probability=True),
            xgb.XGBClassifier(colsample_bytree= 0.6, gamma= 0, learning_rate= 0.2, max_depth= 4,
                              min_child_weight= 5, n_estimators= 20, subsample= 0.8),  # reduced features
             RandomForestClassifier(bootstrap= True, class_weight= 'balanced_subsample', max_depth= 4,
                                    max_features= "log2", min_samples_leaf= 3, min_samples_split= 7,
                                    n_estimators= 30)
        ]
        with PdfPages(sim_func.EXPORT_PATH + '/layered_model/test.pdf') as pdf:
            for i_model1 in range(len(models1)):
                for i_model2 in range(len(models2)):
                    model1 = models1[i_model1]
                    model2 = models2[i_model2]

                    all_labels_pred_true = []
                    accuracy_scores = []
                    precision_scores = []
                    recall_scores = []
                    predictors_train_original = predictors.loc[ind_90]
                    predictors_train_original.reset_index()
                    labels_train = labels.loc[ind_90]
                    labels_train.reset_index()
                    labels_3_train = labels_3.loc[ind_90]
                    labels_3_train.reset_index()
                    labels_test = labels.loc[ind_10]
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                    for train_index, test_index in cv.split(predictors_train_original, labels_train):
                        X_train_original, X_test_original = predictors_train_original.iloc[list(train_index)], \
                        predictors_train_original.iloc[list(test_index)]
                        y_3_train, y_3_test = labels_3_train.iloc[train_index], labels_3_train.iloc[test_index]
                        y_train, y_test = labels_train.iloc[train_index], labels_train.iloc[test_index]

                        if i_model1 != 0:
                            X_train = X_train_original[
                                [feat for feat in sim_func.REDUCED_FEATURES if feat in X_train_original.columns]]
                            X_test = X_test_original[
                                [feat for feat in sim_func.REDUCED_FEATURES if feat in X_train_original.columns]]
                        else:
                            X_train = X_train_original
                            X_test = X_test_original

                        model1.fit(X_train, y_3_train)

                        y_predprob1 = model1.predict_proba(X_test)
                        y_pred1, y_prob1 = mlf.probabilities_to_labels(y_predprob1)

                        cm = confusion_matrix(y_3_test, y_pred1)
                        print('First model')
                        print(cm)

                        ind_red = np.isin(y_train['behavior_generalization'], [1, 2, 3])
                        X_train_red_all = X_train_original.iloc[ind_red]
                        y_train_red = y_train.iloc[ind_red] - 1

                        if i_model2 in [2, 3]:
                            X_train_red = X_train_red_all[
                                [feat for feat in sim_func.REDUCED_FEATURES if feat in X_train_red_all.columns]]
                            X_test_ready = X_test[
                                [feat for feat in sim_func.REDUCED_FEATURES if feat in X_test.columns]]
                        else:
                            X_test_ready = X_test
                            X_train_red = X_train_red_all

                        model2.fit(X_train_red, y_train_red)

                        ind_test_red = [index for index, value in enumerate(y_pred1) if value == 1]

                        y_predprob2 = model2.predict_proba(X_test_ready.iloc[ind_test_red])
                        y_pred2, y_prob2 = mlf.probabilities_to_labels(y_predprob2)

                        y_test_red = y_test.iloc[ind_test_red] - 1

                        cm = confusion_matrix(y_test_red, y_pred2)
                        print('Second model')
                        print(cm)

                        all_predictions = pd.DataFrame()
                        all_predictions['prediction'] = y_pred1.ravel()
                        all_predictions['probability'] = y_prob1.ravel()
                        all_predictions['prediction'] = all_predictions['prediction'].astype(int).map(mapping_red_rev)

                        predictions_red = pd.DataFrame()
                        predictions_red['prediction'] = y_pred2.ravel() + 1
                        predictions_red['prediction'] = predictions_red['prediction'].astype(int).map(inverted_mapping)
                        predictions_red['probability'] = y_prob2.ravel()
                        predictions_red.index = ind_test_red

                        for i in ind_test_red:
                            if predictions_red['probability'].loc[i] >= 0.6:
                                all_predictions.iloc[i] = predictions_red.loc[i]

                        y_test = y_test.reset_index()
                        all_predictions['true'] = y_test['behavior_generalization'].map(inverted_mapping)
                        all_labels_pred_true.append(all_predictions)

                    all_data = pd.concat(all_labels_pred_true, ignore_index=True)

                    cm = confusion_matrix(all_data['true'], all_data['prediction'],
                                          labels=classes)
                    cm_norm = normalize(cm, axis=1, norm='l1')

                    precision_int_is_wrong = precision_score(all_data['true'], all_data['prediction'], average='macro')
                    accuracy_int_is_wrong = accuracy_score(all_data['true'], all_data['prediction'])
                    recall_int_is_wrong = recall_score(all_data['true'], all_data['prediction'], average='macro')

                    ind_intermediate_pred = all_data.index[all_data['prediction'] == 'intermediate energy'].tolist()
                    all_data_original = all_data.copy()
                    for i in ind_intermediate_pred:
                        if all_data['true'].iloc[i] in ['climbing', 'walking', 'exploring']:
                            all_data.loc[i, 'true'] = 'intermediate energy'

                    precision_int_is_correct = precision_score(all_data['true'], all_data['prediction'],
                                                               average='macro')
                    accuracy_int_is_correct = accuracy_score(all_data['true'], all_data['prediction'])
                    recall_int_is_correct = recall_score(all_data['true'], all_data['prediction'], average='macro')

                    scores = pd.DataFrame([[round(precision_int_is_wrong, 3), round(accuracy_int_is_wrong, 3),
                                            round(recall_int_is_wrong, 3)],
                                           [round(precision_int_is_correct, 3), round(accuracy_int_is_correct, 3),
                                            round(recall_int_is_correct, 3)]],
                                          columns=['precision', 'accuracy', 'recall'],
                                          index=['intermediate as false', 'intermediate as correct'])

                    ### External Test

                    pred_test = predictors.iloc[ind_10]

                    ind_red = np.isin(labels_train['behavior_generalization'], [1, 2, 3])
                    pred_int_90 = predictors_train_original.iloc[ind_red]
                    labels_int_90 = labels_train.iloc[ind_red] - 1

                    ind_red_10 = np.isin(labels_test['behavior_generalization'], [1, 2, 3])
                    pred_int_10 = pred_test.iloc[ind_red_10]
                    labels_int_10 = labels_test.iloc[ind_red_10] - 1

                    if i_model1 != 0:
                        predictors_train = predictors_train_original[
                            [feat for feat in sim_func.REDUCED_FEATURES if feat in predictors_train_original.columns]]
                        predictors_ready = predictors[
                            [feat for feat in sim_func.REDUCED_FEATURES if feat in predictors.columns]]
                    else:
                        predictors_train = predictors_train_original
                        predictors_ready = predictors

                    model1.fit(predictors_train, labels_3_train)

                    y_predprob1 = model1.predict_proba(predictors.iloc[ind_10])
                    y_pred1, y_prob1 = mlf.probabilities_to_labels(y_predprob1)
                    cm = confusion_matrix(labels_3.iloc[ind_10], y_pred1)
                    print('First model - external')
                    print(cm)

                    if i_model2 in [2, 3]:
                        pred_int_90 = pred_int_90[
                            [feat for feat in sim_func.REDUCED_FEATURES if feat in pred_int_90.columns]]
                        predictors_ready = predictors_ready.iloc[ind_10]
                        predictors_ready = predictors_ready[
                            [feat for feat in sim_func.REDUCED_FEATURES if feat in predictors.columns]]
                    else:
                        predictors_ready = predictors.iloc[ind_10]

                    model2.fit(pred_int_90, labels_int_90)

                    ind_test_red = [index for index, value in enumerate(y_pred1) if value == 1]

                    y_predprob2 = model2.predict_proba(predictors_ready.iloc[ind_test_red])
                    y_pred2, y_prob2 = mlf.probabilities_to_labels(y_predprob2)

                    labels_3_test = labels_test.copy()
                    labels_3_test['behavior_generalization'][
                        np.isin(labels_3_test['behavior_generalization'], [1, 2, 3])] = 1
                    labels_3_test['behavior_generalization'][labels_3_test['behavior_generalization'] == 4] = 2

                    cm = confusion_matrix(labels_test.iloc[ind_test_red] - 1, y_pred2)
                    print('Second model - external')
                    print(cm)

                    pred_ext = pd.DataFrame()
                    pred_ext['prediction'] = y_pred1.ravel()
                    pred_ext['probability'] = y_prob1.ravel()
                    pred_ext['prediction'] = pred_ext['prediction'].astype(int).map(mapping_red_rev)

                    predictions_red = pd.DataFrame()
                    predictions_red['prediction'] = y_pred2.ravel() + 1
                    predictions_red['prediction'] = predictions_red['prediction'].astype(int).map(inverted_mapping)
                    predictions_red['probability'] = y_prob2.ravel()
                    predictions_red.index = ind_test_red
                    print(predictions_red['prediction'].value_counts())

                    for i in ind_test_red:
                        if predictions_red['probability'].loc[i] >= 0.6:
                            pred_ext.loc[i, 'prediction'] = predictions_red.loc[i, 'prediction']
                            pred_ext.loc[i, 'probability'] = predictions_red.loc[i, 'probability']
                            # print(i)
                            # print(predictions_red.loc[i])

                    labels_test = labels_test.reset_index()
                    pred_ext['true'] = labels_test['behavior_generalization'].map(inverted_mapping)
                    print('Predictions')
                    print(pred_ext['prediction'].value_counts())
                    print('True')
                    print(pred_ext['true'].value_counts())

                    cm_ext = confusion_matrix(pred_ext['true'], pred_ext['prediction'],
                                              labels=classes)
                    print('external model')
                    print(cm_ext)
                    cm_ext_norm = normalize(cm_ext, axis=1, norm='l1')

                    precision_int_is_wrong = precision_score(all_data['true'], all_data['prediction'], average='macro')
                    accuracy_int_is_wrong = accuracy_score(all_data['true'], all_data['prediction'])
                    recall_int_is_wrong = recall_score(all_data['true'], all_data['prediction'], average='macro')

                    ind_intermediate_pred = pred_ext.index[pred_ext['prediction'] == 'intermediate energy'].tolist()

                    for i in ind_intermediate_pred:
                        if all_data['true'].iloc[i] in ['climbing', 'walking', 'exploring']:
                            all_data.loc[i, 'true'] = 'intermediate energy'

                    precision_int_is_correct = precision_score(all_data['true'], all_data['prediction'],
                                                               average='macro')
                    accuracy_int_is_correct = accuracy_score(all_data['true'], all_data['prediction'])
                    recall_int_is_correct = recall_score(all_data['true'], all_data['prediction'], average='macro')

                    scores_ext = pd.DataFrame([[round(precision_int_is_wrong, 3), round(accuracy_int_is_wrong, 3),
                                                round(recall_int_is_wrong, 3)],
                                               [round(precision_int_is_correct, 3), round(accuracy_int_is_correct, 3),
                                                round(recall_int_is_correct, 3)]],
                                              columns=['precision', 'accuracy', 'recall'],
                                              index=['intermediate as false', 'intermediate as correct'])

                    matrices_and_tables = [['90% cross-validation', cm_norm, scores],
                                           ['10% external test', cm_ext_norm, scores_ext]]

                    for title, cm_norm, scores in matrices_and_tables:
                        fig = plt.figure(figsize=(15, 12))
                        fig.suptitle('Layered Model, combined confusion matrix, ' + title + '\n' +
                                     type(model1).__name__ + ' ' + str(i_model1) + '\n' +
                                     type(model2).__name__ + ' ' + str(i_model2))
                        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1])

                        # Define a colormap
                        cmap = plt.cm.Blues

                        ax = fig.add_subplot(gs[0, 0])
                        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap=cmap, cbar=False, ax=ax, vmin=0, vmax=1,
                                    square=True)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('True')
                        ax.set_xticklabels(classes, rotation=45)
                        ax.set_yticklabels(classes, rotation=45)

                        ax1 = fig.add_subplot(gs[0, 1])
                        ax1.hist(all_data['probability'], bins=30)
                        ax1.set_title('Histogram of prediction probabilities')

                        axs2 = plt.subplot(gs[1, 0:2])
                        axs2.axis('tight')
                        axs2.axis('off')

                        table = axs2.table(cellText=scores.values,
                                           colLabels=scores.columns,
                                           rowLabels=scores.index,
                                           cellLoc='center', loc='center')

                        table.auto_set_font_size(False)
                        table.set_fontsize(12)

                        plt.subplots_adjust(wspace=0.5, hspace=0.3)

                        pdf.savefig(fig)
                        plt.close(fig)

                    all_data_original_1 = all_data_original.copy()
                    plt_func.sankey_diagram(all_data_original_1, filepath="layered_model_sankey.svg")

                    all_data_original_2 = all_data_original.copy()
                    all_data_original_2["prediction"][all_data_original_2["probability"] < 0.6] = "unknown"
                    plt_func.sankey_diagram(all_data_original_2, filepath="layered_model_sankey_unknown.svg")
