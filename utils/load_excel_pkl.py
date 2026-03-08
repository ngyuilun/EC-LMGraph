import os
import sys
import pickle

os.umask(0)
import pandas as pd


def load_excel_pkl(path_excel):


    if 1:
        if os.path.exists(path_excel.replace('.xlsx','.pkl')) == True:
            print('Found pkl... Loading...')
            with open(path_excel.replace('.xlsx','.pkl'),'rb') as f1:
                df_target_list = pickle.load(f1)
        else:
            print('Cannot find pkl... Loading from xlsx...')
            df_target_list = pd.read_excel(path_excel)
            with open(path_excel.replace('.xlsx','.pkl'),'wb') as f1:
                pickle.dump(df_target_list,f1, pickle.HIGHEST_PROTOCOL)

    return df_target_list


def write_excel_pkl(df_target_list,path_train_test_list):


    with open(path_train_test_list.replace('.xlsx','.pkl'), 'wb') as f:
        pickle.dump(df_target_list, f, pickle.HIGHEST_PROTOCOL)

    df_target_list.to_excel(path_train_test_list,index=False)




if __name__ == '__main__':
    ""