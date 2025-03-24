#!/usr/bin/python3

"""
Filename: sfs_feature_importance.py
Author: Eva Reinhardt
Date: 2024-12-09
Version: 1.0
Description: Evaluating feature_selection using feature importance for RandomForest applications
and Sequential Feature Selection for the other approaches. Output is a diagram depicting precision
for forward feature selection.

"""

from time import time

import pandas as pd
import numpy as np

from tkinter import filedialog

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

import xgboost as xgb

from raccoon_acc_setup import predictor_calculation as pred_cal
from raccoon_acc_setup import machine_learning_functions as mlf
from raccoon_acc_setup import importing_raw_data as im_raw
from raccoon_acc_setup import gui_functions as guif
from raccoon_acc_setup import variables_simplefunctions as sim_func


filepaths_peter = [
    sim_func.IMPORT_PARAMETERS['Peter']['filepath_pred']]
filepaths_domi = [
    sim_func.IMPORT_PARAMETERS['Dominique']['filepath_pred']]
ml_algs_set_param = {
    'RandomForest_low': RandomForestClassifier(bootstrap= True, class_weight= 'balanced_subsample', max_depth= 4,
                                               max_features= 'sqrt', min_samples_leaf= 3, min_samples_split= 5,
                                               n_estimators= 30),
    # 'RandomForest': RandomForestClassifier(class_weight='balanced_subsample', random_state=42, bootstrap=True,
    #                                         max_depth=20, max_features='sqrt', min_samples_leaf=1,
    #                                         min_samples_split=2, n_estimators=200),
    # 'SupportVectorMachine_linearkernel': svm.SVC(probability=True, kernel="linear", C=80, gamma='scale'), # rausnehmen, weil es ewig dauert?
    # 'SupportVectorMachine_rbfkernel': svm.SVC(probability=True, kernel='rbf', C=30, gamma = 'scale'),
    'XGBoost_low': xgb.XGBClassifier(colsample_bytree= 0.8, gamma= 1,
                                  learning_rate= 0.2, max_depth= 4,
                                  n_estimators= 20, subsample= 0.8, min_child_weight=10)
}

filepaths = [filepaths_peter, filepaths_domi]


if __name__ == "__main__":

    pred = pred_cal.create_pred_complete(filepaths)

    pred = im_raw.convert_beh(pred, 'generalization', 'generalization3')
    pred = im_raw.convert_beh(pred, 'translation')
    variations = [['resting', 'walking', 'climbing', 'exploring', 'high energy'], ['exploring', 'resting', 'climbing'], ['exploring', 'walking', 'climbing']]

    option_method = guif.choose_multiple_options(['feature importance', 'SFS'], 'Which feature selection methods should be used?')
    output_pdf_path = filedialog.asksaveasfilename(title="Save as")
    for algo_name, algo in ml_algs_set_param.items():
        output_pdf_path_alg = output_pdf_path.split('.')
        output_pdf_path_alg[-2] = output_pdf_path_alg[-2] + \
                                  '_' + algo_name
        output_pdf_path_alg = '.'.join(output_pdf_path_alg)
        with (PdfPages(output_pdf_path_alg) as pdf):
            for classes in variations:
                pred_temp = pred.copy()
                pred_temp = pred_temp[pred_temp['behavior_generalization'].isin(classes)]

                predictors, labels = mlf.splitting_pred(pred_temp)
                predictors = predictors[['Ymean', 'Ndyn', 'XZmean', 'XZmin', 'XZmax', 'fft_base', 'fft_wmean', 'fft_std']]
                labels = labels['behavior_generalization'].to_numpy().ravel()

                if 'feature importance' in option_method:
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
                    ind_80, ind_20 = next(sss.split(predictors, labels))
                    pred_train = predictors.iloc[ind_80]
                    pred_test = predictors.iloc[ind_20]
                    lab_train = labels[ind_80]
                    lab_test = labels[ind_20]
                    algo.fit(pred_train, lab_train)
                    lab_pred = algo.predict(pred_test)

                    accuracy, recall, precision, f1, scores = mlf.calc_scores(lab_pred, pd.DataFrame(lab_test))
                    if hasattr(algo, 'feature_importances_'):
                        importances = pd.DataFrame({'Importance': algo.feature_importances_, 'Feature':predictors.columns})
                        print(importances)
                        fig = plt.figure(figsize=(10, 6))
                        fig.suptitle('Feature importances when distinguishing: ' + ', '.join(classes) + '\naccuracy: '+str(round(accuracy,3))+
                                     ', precision: ' + str(round(precision,3)) + ', recall: ' + str(round(recall,3)) + ', f1 score: ' + str(round(f1,3)))

                        plt.bar(importances['Feature'], importances['Importance'])

                        plt.xlabel('Features')
                        plt.ylabel('Importance')
                        plt.ylim(0, 0.5)
                        plt.title('Feature Importances')
                        plt.xticks(rotation=90)

                        plt.tight_layout()
                        if pdf:
                            pdf.savefig(fig)
                            plt.close(fig)
                        else:
                            plt.show()

                if 'SFS' in option_method:

                    start = time()
                    sfs = SFS(algo, k_features=int(predictors.shape[1]), cv=5, scoring='precision_macro')
                    sfs.fit(predictors, labels)
                    end = time()
                    print(end - start)
                    subsets_df = pd.DataFrame([
                        {
                            'subset_size': subset_size,
                            'selected_features': list(details['feature_idx']),
                            'cv_scores': details['cv_scores'],
                            'avg_score': details['avg_score']
                        }
                        for subset_size, details in sfs.subsets_.items()
                    ])

                    if len(classes) ==5:
                        subset_path = 'sfs_subsets_' + algo_name + '.csv'
                        subsets_df.to_csv(subset_path, index=False)

                    print(sfs.subsets_)
                    subsets = sfs.subsets_
                    avg_scores = [entry['avg_score'] for entry in subsets.values()]
                    maximum = max(avg_scores)
                    max_ind = np.argmax(avg_scores)
                    added_features = []
                    previous_features = set()
                    print('Best accuracy score: %.2f' % maximum)

                    for step in sorted(subsets.keys()):
                        current_features = set(subsets[step]['feature_names'])
                        new_feature = list(current_features - previous_features)
                        if new_feature:
                            added_features.append(new_feature[0])
                        previous_features = current_features

                    best_features = added_features[0:(max_ind+1)]
                    fig = plt.figure(figsize=(22, 12))
                    fig.suptitle('Sequential Feature Selection for subset: ' + ', '.join(classes) + '\nBest average precision score: ' + str(maximum) +
                    '\nBest subset (corresponding names):' + ', '.join(best_features))
                    x_pos = np.arange(len(avg_scores))
                    plt.axhline(y=maximum, color='r', linestyle='-')
                    plt.plot(x_pos, avg_scores, marker='o')
                    plt.xticks(x_pos, added_features, rotation=90, ha='right')
                    plt.xlabel('Added Feature (meaning all features to the left of the x-tick are also used)')
                    plt.ylabel('Average Precision')
                    plt.ylim(0.3, 0.9)
                    plt.grid()
                    plt.title('Model Performance during Sequential Feature Selection, mlxtend')
                    if pdf:
                        pdf.savefig(fig)
                        plt.close(fig)
                    else:
                        plt.show()




