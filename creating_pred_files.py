"""
Filename: creating_pred_files.py
Author: Eva Reinhardt
Date: 2024-12-09
Version: 1.0
Description: creating the predictor files from raw acc files
"""
from raccoon_acc_setup import predictor_calculation as pred_cal
from raccoon_acc_setup import importing_raw_data as im_raw
from raccoon_acc_setup import gui_functions as guif
from raccoon_acc_setup import variables_simplefunctions as sim_func
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")



if __name__ == "__main__":

    option_files = guif.choose_option(options=['Choose file myself', 'Choose set datasets'])

    option_labeled = guif.choose_option(['Labelled datasets', 'Unlabelled data'])
    if option_labeled == 'Labelled datasets':
        option_names = guif.choose_option(['Peter', 'Dominique'], title='Choose a dataset: ')
    else:
        option_names = guif.choose_option(['Katti', 'Caro S', 'Caro W'], title= 'Choose a dataset: ')

    if option_files == 'Choose file myself':
        filepaths = list(guif.open_file_dialog("Select a file - raw acc data"))
        if option_labeled == 'Labelled datasets':
            filepath_beh = list(guif.open_file_dialog("Select a file - behaviour data"))
            filepaths_both = [[filepaths, filepath_beh]]

    else:
        if option_names == 'Peter':
            fs = 33.3
            filepaths_both = sim_func.IMPORT_PARAMETERS['Peter']['filepath_acc_beh']

        elif option_names == 'Dominique':
            fs = 33.3
            filepaths_both = sim_func.IMPORT_PARAMETERS['Dominique']['filepath_acc_beh']
        elif option_names == 'Katti':
            fs = 16.67
            # filepaths = ['/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild - Katti Meyer/ACC_only_3/tag5123.txt',
            #              '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild - Katti Meyer/ACC_only_3/tag5124.txt',
            #              '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild - Katti Meyer/ACC_only_3/tag5125.txt',
            #              '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild - Katti Meyer/ACC_only_3/tag5126.txt',
            #              '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild - Katti Meyer/ACC_only_3/tag5128.txt',
            #              '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild - Katti Meyer/ACC_only_3/tag5129.txt',
            #              '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild - Katti Meyer/ACC_only_3/tag5130.txt',
            #              '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild - Katti Meyer/ACC_only_3/tag5131.txt',
            #              '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild - Katti Meyer/ACC_only_3/tag5132.txt',
            #              '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild - Katti Meyer/ACC_only_3/tag5336.txt']
            filepaths = ['/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild - Katti Meyer/ACC_only_3/tag5123red.txt',
                         '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild - Katti Meyer/ACC_only_3/tag5124red.txt',
                         '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild - Katti Meyer/ACC_only_3/tag5125red.txt',
                         '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild - Katti Meyer/ACC_only_3/tag5126red.txt']
        elif option_names == 'Caro S':
            fs = 18.74
            filepaths = [
                #'/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild BB - Caro Scholz-Biomove/1630_ACC_movebankred.csv',
                         # '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild BB - Caro Scholz-Biomove/1628_ACC_movebankred.csv',
                         # '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild BB - Caro Scholz-Biomove/1631_ACC_movebankred.csv',
                         # '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild BB - Caro Scholz-Biomove/1633_ACC_movebankred.csv',
                         # '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild BB - Caro Scholz-Biomove/1634_ACC_movebankred.csv',
                         # '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild BB - Caro Scholz-Biomove/1636_ACC_movebankred.csv',
                         # '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild BB - Caro Scholz-Biomove/1637_ACC_movebankred.csv',
                         '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild BB - Caro Scholz-Biomove/1638_ACC_movebank.csv']
            # filepaths = ['/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild BB - Caro Scholz-Biomove/1630_ACC_movebank.csv',
            #              '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild BB - Caro Scholz-Biomove/1628_ACC_movebank.csv',
            #              '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild BB - Caro Scholz-Biomove/1631_ACC_movebank.csv',
            #              '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild BB - Caro Scholz-Biomove/1633_ACC_movebank.csv',
            #              '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild BB - Caro Scholz-Biomove/1634_ACC_movebank.csv',
            #              '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild BB - Caro Scholz-Biomove/1636_ACC_movebank.csv',
            #              '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild BB - Caro Scholz-Biomove/1637_ACC_movebank.csv',
            #              '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild BB - Caro Scholz-Biomove/1638_ACC_movebank.csv']
        else:
            fs = 33.3
            filepaths = ['/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild Berlin - Caro Weh-IZW/acc_tag5031.csv',
                         '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild Berlin - Caro Weh-IZW/acc_tag5034.csv',
                         '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild Berlin - Caro Weh-IZW/acc_tag5035.csv',
                         '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild Berlin - Caro Weh-IZW/acc_tag5036.csv',
                         '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild Berlin - Caro Weh-IZW/acc_tag5038.csv',
                         '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild Berlin - Caro Weh-IZW/acc_tag5115.csv',
                         '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild Berlin - Caro Weh-IZW/acc_tag5116.csv',
                         '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild Berlin - Caro Weh-IZW/acc_tag5133.csv',
                         '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild Berlin - Caro Weh-IZW/acc_tag5134.csv',
                         '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild Berlin - Caro Weh-IZW/acc_tag5135.csv',
                         '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild Berlin - Caro Weh-IZW/acc_tag5136.csv',
                         '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild Berlin - Caro Weh-IZW/acc_tag5599.csv',
                         '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild Berlin - Caro Weh-IZW/acc_tag5602.csv',
                         '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild Berlin - Caro Weh-IZW/acc_tag5603.csv',
                         '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild Berlin - Caro Weh-IZW/acc_tag5604.csv',
                         '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild Berlin - Caro Weh-IZW/acc_tag5605.csv',
                         '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild Berlin - Caro Weh-IZW/acc_tag5606.csv',
                         '/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild Berlin - Caro Weh-IZW/acc_tag5607.csv']
            # filepaths = ['/media/eva/eva-reinhar/your folders/01 raw data/Waschbär wild Berlin - Caro Weh-IZW/Raccoons of Berlin_ACC_20190402.csv']

    option_mw = guif.choose_option(options=['Take only the first 54 samples of one burst', 'Take first and last 54 samples of burst'])
    pred = []
    if option_labeled == 'Labelled datasets':
        fs=33.3
        for f in filepaths_both:
            filepaths = f[0]
            filepath_beh = f[1]
            if type(filepath_beh)!=list:
                filepath_beh = [filepath_beh]
            # depending on the dataset's person, the behavior is differently imported

            if 'Peter' in filepath_beh[0]:
                beh = im_raw.import_beh_peter(filepath_beh[0])
            else:
                beh = im_raw.import_beh_domi(filepath_beh[0])

            # this part can be used later to get an overview, which behaviors are categorized in which work
            if 'Peter' in filepath_beh:
                behavior_peter = beh['behavior'].unique()
            elif 'Dominique' in filepath_beh:
                behavior_domi = beh['behavior'].unique()
            elif 'Inge' in filepath_beh:
                behavior_inge = beh['behavior'].unique()

            for filepath in filepaths:
                ACC = im_raw.import_eobs(filepath)

                if 'Dominique' in (filepath and filepath_beh[0]):
                    # datetime_unique_acc.append(ACC['datetime'].unique())
                    print('merging')
                    print(filepath)
                    ACC = ACC.sort_values('datetime')
                    beh = beh.sort_values('datetime')
                    data = pd.merge_asof(ACC, beh, on='datetime', tolerance=pd.Timedelta('10s'),
                                         direction='nearest')
                    if '5032' in filepath:
                        data = data.drop(['behavior_Ottilie'], axis=1)
                        data = data.rename(columns={'behavior_Lisa': 'behavior'})
                    elif '5033' in filepath:
                        data = data.drop(['behavior_Lisa'], axis=1)
                        data = data.rename(columns={'behavior_Ottilie': 'behavior'})
                    data = data.dropna(axis=0)
                    data = im_raw.convert_beh(data, 'Dominique')

                else:
                    ACC = ACC.sort_values('datetime')
                    beh = beh.sort_values('datetime')
                    data = pd.merge_asof(ACC, beh, on='datetime', tolerance=pd.Timedelta('10s'),
                                         direction='nearest')
                    data = data.dropna()

                print(data.head())
                print(filepath)
                print(fs)
                # predictor data from each file is appended to the list pred
                if option_mw == 'Take only the first 54 samples of one burst':
                    pred.append(pred_cal.calculate_pred(data, fs))
                else:
                    pred.append(pred_cal.calculate_pred(data, fs, mw=True, step=54))

        column_names = list(pred[0].columns)
        column_names[-1] = 'behavior'

        pred = [df.rename(columns={col: i for i, col in enumerate(df.columns)}) for df in pred]

        result = pd.concat(pred, ignore_index=True)

        result.columns = column_names

        guif.save_pred(result)



    else:
        for filepath in filepaths:
            acc = im_raw.import_acc_data(filepath)

            print(acc)
            output_filepath = filepath.split('/')
            filename = output_filepath[-1]
            filename = 'pred_' + filename
            output_filepath[-1] = filename
            output_filepath = '/'.join(output_filepath)
            print(output_filepath)

            pred = pred_cal.calculate_pred(acc, fs)
            pred.to_csv(output_filepath, index=False)
