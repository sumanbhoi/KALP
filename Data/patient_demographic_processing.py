# Patient demographic info creation

import pandas as pd
import numpy as np
import dill

def patient_demo_process():
    path_demo = '../Data/visit_demo'
    df_demo = pd.read_pickle(path_demo)
    df_demo.dropna(inplace=True)
    df_demo.drop(columns=['racedesc', 'DM_diagnosisYr'], axis=1, inplace=True)
    print('size', np.size(df_demo))
    print('shape', np.shape(df_demo))
    path_all = '../Data/visit_all'
    df_all = pd.read_pickle(path_all)
    df_all.dropna(inplace=True)
    # print(df_all.head())
    df_all.drop(columns=['date_blended', 'prescribeddateyyyymmdd1', 'entereddateyyyymmdd1', 'monitoringdateyyyymmdd1'], axis=1, inplace=True)
    df_all['YOV'] = pd.DatetimeIndex(df_all['visitdateyyyymmdd1']).year
    df_all['YOV'] = df_all['YOV'].apply(np.int64)
    df_all = df_all.drop_duplicates(subset=['patientcode'],keep='first')


    df_merge = pd.merge(df_demo, df_all, on='patientcode', how='inner')
    df_merge['Age'] = df_merge['YOV'] - df_merge['YOB']
    df_merge = df_merge.drop_duplicates(subset=['patientcode'], keep='first')

    path_moni = '../Data/visit_Monitoring'
    df_moni = pd.read_pickle(path_moni)
    df_moni.drop(columns=['monitoringdateyyyymmdd1'], axis=1, inplace=True)
    df_info = pd.merge(df_merge, df_moni, on=['patientcode', 'vID'], how='inner')
    df_info = df_info[df_info['resultitemname'] == 'Weight']
    df_info = df_info.reset_index(drop=True)
    df_info.dropna(inplace=True)
    print(df_info.columns)
    df_info.drop(columns=['YOB', 'vID', 'visitdateyyyymmdd1', 'YOV', 'resultitemname'],
                axis=1, inplace=True)
    print(df_info.columns)
    print(df_info.head())
    # df_info['genderdesc'] = df_info['genderdesc'].apply({1: 'Male', 0: 'Female'}.get)
    df_info['genderdesc'].replace(['Female', 'Male'], [0, 1], inplace=True)
    df_info["Age"] = (df_info["Age"] - df_info["Age"].min()) / (df_info["Age"].max() - df_info["Age"].min())
    df_info["resultvalue_nbr"] = (df_info["resultvalue_nbr"] - df_info["resultvalue_nbr"].min()) / (df_info["resultvalue_nbr"].max() - df_info["resultvalue_nbr"].min())
    print(df_info.head())
    print(df_info['genderdesc'])
    return df_info

df_demo_final = patient_demo_process()