#!/usr/bin/python3
"""
Filename: machine_learning.py
Author: Eva Reinhardt
Date: 2024-12-09
Version: 1.0
Description: conducting machine learning with different algorithms. Evaluate the performance.
Conducting parameter optimization.
"""

from raccoon_acc_setup import predictor_calculation as pred_cal
from raccoon_acc_setup import gui_functions as guif
from raccoon_acc_setup import importing_raw_data as im_raw
from raccoon_acc_setup import machine_learning_functions as mlf
from raccoon_acc_setup import plot_functions as plt_func
from raccoon_acc_setup import variables_simplefunctions as sim_func

from tkinter import filedialog

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

import xgboost as xgb

from matplotlib.backends.backend_pdf import PdfPages

import pandas as pd

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, CondensedNearestNeighbour

mapping = sim_func.MAPPING
inverted_mapping = {v: k for k, v in mapping.items()}
class_order = mapping.keys()




if __name__ == "__main__":
    filepaths_peter = [
        sim_func.IMPORT_PARAMETERS['Peter']['filepath_pred']]
    filepaths_domi = [
        sim_func.IMPORT_PARAMETERS['Dominique']['filepath_pred']]
    ml_algs_set_param = {
        # # 'RandomForest': RandomForestClassifier(class_weight='balanced_subsample', random_state=42, bootstrap= True, max_depth= 20, max_features= 'sqrt', min_samples_leaf= 1, min_samples_split= 2, n_estimators= 200),
        # 'RandomForest_low': RandomForestClassifier(bootstrap=True, class_weight='balanced_subsample', max_depth=4,
        #                                            max_features='sqrt', min_samples_leaf=3, min_samples_split=5,
        #                                            n_estimators=30),
        #
        # # 'SupportVectorMachine_linearkernel': svm.SVC(probability=True, kernel="linear", C=80, gamma='scale'),
        # # # 'LinearSVM': svm.LinearSVC(C=1, max_iter=10000),
        'SupportVectorMachine_rbfkernel': svm.SVC(probability = True, kernel="rbf", C= 30, gamma= "scale")  # Model training step

        # 'SupportVectorMachine_polykernel': svm.SVC(probability=True, kernel="poly", degree=4, gamma="auto", C=10,
        #                                            coef0=0.5),
        # 'SupportVectorMachine_sigmoidkernel': svm.SVC(probability=True, kernel="sigmoid", gamma="auto", C=0.01,
        #                                            coef0=0.8),
        # # # 'XGBoost': xgb.XGBClassifier(colsample_bytree= 0.8493192507310232, gamma= 3.308980248526492,
        # # #                              learning_rate= 0.02906750508580709, max_depth= 9, min_child_weight= 8,
        # # #                              n_estimators= 904, subsample= 0.8918424713352255)
        # 'XGBoost_low': xgb.XGBClassifier(colsample_bytree=0.8, gamma=1,
        #                                  learning_rate=0.2, max_depth=4,
        #                                  n_estimators=20, subsample=0.8, min_child_weight = 10 )
    }

    ml_algs_optim_param = {
        'RandomForest': [RandomForestClassifier(), {
            #     'n_estimators': [100, 200, 500],                  # Number of trees
            #     'max_depth': [10, 20, 30, None],                  # Maximum depth of a tree
            #     'min_samples_split': [2, 5, 10, 15],                  # Minimum samples required to split
            #     'min_samples_leaf': [1, 2, 4],                    # Minimum samples required at a leaf node
            #     'max_features': ['auto', 'sqrt', 'log2'],         # Number of features to consider at each split
            #     'bootstrap': [True, False],                       # Use bootstrap sampling
            #     'class_weight': ['balanced', 'balanced_subsample', None]  # Adjust class weights
            #
            'n_estimators': [20, 25, 30],  # Number of trees
            'max_depth': [2, 4],  # Maximum depth of a tree
            'min_samples_split': [5, 7,10],  # Minimum samples required to split
            'min_samples_leaf': [2, 3, 4],  # Minimum samples required at a leaf node
            'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider at each split
            'bootstrap': [True],  # Use bootstrap sampling
            'class_weight': ['balanced_subsample']  # Adjust class weights
        #     # #
         }],
        'SupportVectorMachine_linearkernel': [svm.SVC(probability=True, kernel="linear"), {
            'C': [0.1, 1, 10, 80],
            'gamma': ['scale', 'auto']  # Kernel coefficient
        }],
        'SupportVectorMachine_rbfkernel': [svm.SVC(probability=True, kernel="rbf"), {
            'C': [0.1, 10, 30, 80],
            'gamma': ['scale', 'auto']  # Kernel coefficient
        }],
        'SupportVectorMachine_polykernel': [svm.SVC(probability=True, kernel="poly"), {
            'C': [0.01, 1, 10, 80],  # uniform(0.1, 10),  # Regularization parameter
            'gamma': ['scale', 'auto'],  # Kernel coefficient
            'degree': [2, 3, 4, 5],  # Degree for 'poly' kernel
            # Independent term in kernel function for poly and sigmoid
            'coef0': [0.1, 0.5, 0.8]
        }],
        'SupportVectorMachine_sigmoid': [svm.SVC(probability=True, kernel="sigmoid"), {
            'C': [0.01, 1, 10],  # uniform(0.1, 10),  # Regularization parameter
            'gamma': ['scale', 'auto'],  # Kernel coefficient
            # Independent term in kernel function for poly and sigmoid
            # 'coef0': uniform(0, 1)
            'coef0': [0.1, 0.5, 0.8]
        }],
        'XGBoost': [xgb.XGBClassifier(), {
            #
            # 'n_estimators': [100, 300, 800, 1000],  # Number of boosting rounds
            # 'max_depth': [3, 6, 9],  # Maximum depth of a tree
            'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage
            'subsample': [0.6, 0.8, 1.0],  # Fraction of samples used per tree
            'colsample_bytree': [0.6, 0.8, 1.0],  # Fraction of features used per tree
            'gamma': [0, 0.5, 1, 3],  # Minimum loss reduction
            # #
            'min_child_weight': [2, 5, 10],  # Minimum sum of instance weight
            # 'reg_alpha': [0, 0.01, 0.1],  # L1 regularization term
            # 'reg_lambda': [1, 1.5, 2],  # L2 regularization term
            # 'scale_pos_weight': [1, 3, 5]  # Class balancing (useful for imbalanced datasets)

            'n_estimators': [10, 15, 20],  # Number of trees
            'max_depth': [2, 4]  # Maximum depth of a tree

        }]
    }

    sampling_methods = {
        'SMOTE': SMOTE(sampling_strategy='minority', k_neighbors=5),
        'BorderlineSMOTE': BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5, kind='borderline-1'),
        'ADASYN': ADASYN(sampling_strategy='minority', n_neighbors=5),
        'RandomUndersampler': RandomUnderSampler(random_state=42),
        'NearMiss': NearMiss(version=3, n_neighbors=5),
        'CondensedNearestNeighbour': CondensedNearestNeighbour(n_neighbors=5)
    }

    seq_feat_sel = {
        # 'RandomForest': ['Ndyn', 'Ymean', 'XZmax', 'XZmin', 'fft_wmean', 'XZmean', 'fft_std', 'behavior',
        #                  'behavior_generalization'],
        # 'XGBoost': ['Ndyn', 'Ymean', 'XZmin', 'fft_wmean', 'XZmean', 'fft_base', 'fft_std', 'behavior', 'behavior_generalization']
    }

    filepaths = [filepaths_peter, filepaths_domi]

    # output_pdf_path = filedialog.asksaveasfilename(title="Save as")
    pred_com = pred_cal.create_pred_complete(filepaths)

    # option_gen = guif.choose_option(
    #     ['generalization1', 'generalization2', 'generalization3', 'All generalizations'])
    # option_opt = guif.choose_option(
    #     ['Including parameter optimization', 'Using set parameters'])
    # option_feat = guif.choose_option(['Including Feature reduction results', 'Don\'t include'])

    output_pdf_path = sim_func.EXPORT_PATH + 'new_external_test_red_feat.pdf'
    option_gen = 'generalization3'
    option_opt = 'Including parameter optimization'
    option_feat = 'Including Feature reduction results'

    if option_gen == 'All generalizations':
        gens = ['generalization1', 'generalization2', 'generalization3']
    else:
        gens = [option_gen]
    for g in gens:
        pred = im_raw.convert_beh(pred_com, 'generalization', g)
        pred = im_raw.convert_beh(pred, 'translation')


        pred['datetime'] = pd.to_datetime(pred['datetime'])
        pred['date'] = pred['datetime'].dt.date

        ### external test with set dates as external test data set

        date_external = pd.to_datetime(['2016-10-31','2016-11-3', '2022-09-26','2022-09-28']).date
        ind_10 = pred.index[pred['date'].isin(date_external)]
        ind_90 = pred.index[~pred['date'].isin(date_external)]

        pred = pred.drop('date', axis=1)



        # # ind_rand = pred.sample(frac=0.1, random_state=42).index.tolist()
        predictors, labels = mlf.splitting_pred(pred, mapping = mapping)
        behaviors = pred['behavior']


        ### external test with StratifiedShuffleSplit

        # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        # ind_90, ind_10 = next(sss.split(predictors, labels))
        # ind_90 = sorted(ind_90)
        # ind_10 = sorted(ind_10)

        if option_opt == 'Using set parameters':
            ml_dict = ml_algs_set_param
        else:
            ml_dict = ml_algs_optim_param



        for algo_name, algo in ml_dict.items():
            print(algo_name)
            if 'Support' in algo_name:
                scaler = StandardScaler()
                scaler.fit_transform(predictors)

            output_pdf_path_alg = output_pdf_path.split('.')
            output_pdf_path_alg[-2] = output_pdf_path_alg[-2] + \
                                      '_' + algo_name + '_' + g
            output_pdf_path_alg = '.'.join(output_pdf_path_alg)
            if option_feat == 'Including Feature reduction results':
                predictors_1 = predictors.copy()
                predictors_1 = predictors_1[[c for c in sim_func.REDUCED_FEATURES if not 'datetime']]
                if algo_name in seq_feat_sel:
                    selected_features = seq_feat_sel[algo_name]
                    predictors_1 = predictors.copy()
                    predictors_1 = predictors_1[selected_features]
                else:
                    predictors_1 = predictors.copy()
            else:
                predictors_1 = predictors.copy()

            with PdfPages(output_pdf_path_alg) as pdf:
                plt_func.visualise_predictions_ml(
                    pred, predictors_1, labels, behaviors, g, algo_name, algo, opt=option_opt, ext_test = [ind_90, ind_10], pdf=pdf)
                for sampling_name, sampling in sampling_methods.items():
                    plt_func.visualise_predictions_ml(pred, predictors_1, labels, behaviors, g, algo_name, algo, opt=option_opt,
                                              sampling_func= [sampling_name, sampling], ext_test = [ind_90, ind_10], pdf=pdf)
        pred = pred_com.copy()
