# Final data processing code for KALP IJCAI 2022
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy import float64

data_path = './MM/'


def process_med():
    med_pd = pd.read_pickle(data_path + "visit_Med")
    # filter
    med_pd.drop(columns=['prescribeddateyyyymmdd1',
                         'prescribedmedicationdesc', 'frequencycode',
                         'freqCode_dailyNbr', 'durationquantity', 'durationquantityunit'], axis=1, inplace=True)
    # g to mg conversion
    med_pd.loc[med_pd['unitofmeasure1'] == 'g', 'dosage_nbr'] = med_pd['dosage_nbr'] * 1000
    med_pd.drop(columns=['unitofmeasure1'], axis=1, inplace=True)
    med_pd['dosage_norm'] = med_pd.groupby('medDesc1')['dosage_nbr'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()))
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd = med_pd.reset_index(drop=True)

    # visit > 2
    def process_visit_lg2(med_pd):
        a = med_pd[['patientcode', 'vID']].groupby(by='patientcode')['vID'].unique().reset_index()
        # print(a.head())
        a['vID_Len'] = a['vID'].map(lambda x: len(x))
        a = a[a['vID_Len'] > 1]
        return a

    med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)
    med_pd = med_pd.merge(med_pd_lg2[['patientcode']], on='patientcode', how='inner')

    return med_pd.reset_index(drop=True)


def process_diag():
    diag_pd = pd.read_pickle(data_path + "visit_Diag")
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=['visitdateyyyymmdd1', 'diagnosiscodedesc',
                          'diagnosiscode', 'icd10cm'], inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['patientcode', 'vID'], inplace=True)
    return diag_pd.reset_index(drop=True)


def process_lab():
    lab_pd = pd.read_pickle(data_path + "visit_Lab")
    lab_pd.drop(columns=['entereddateyyyymmdd1', 'resultvalue1'], inplace=True)
    lab_pd.dropna(inplace=True)
    lab_pd.drop_duplicates(inplace=True)
    lab_pd.sort_values(by=['patientcode', 'vID'], inplace=True)
    # Take patients with HBA1c, blood test only
    lab_pd = lab_pd[lab_pd.resultitemdesc == 'HBA1c, blood']
    lab_pd['value_norm'] = lab_pd.groupby('resultitemdesc')['resultvalue_nbr'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()))
    # print('max and min lab test value')
    # print(lab_pd['resultvalue_nbr'].max())
    # print(lab_pd['resultvalue_nbr'].min())
    lab_pd.drop(columns=['resultitemdesc', 'resultvalue_nbr'], inplace=True)
    return lab_pd.reset_index(drop=True)


def process_all():
    # get med and diag (visit>=2)
    med_pd = process_med()
    diag_pd = process_diag()
    lab_pd = process_lab()

    med_pd_key = med_pd[['patientcode', 'vID']].drop_duplicates()
    diag_pd_key = diag_pd[['patientcode', 'vID']].drop_duplicates()
    lab_pd_key = lab_pd[['patientcode', 'vID']].drop_duplicates()

    combined_key = med_pd_key.merge(diag_pd_key, on=['patientcode', 'vID'], how='inner')
    combined_key = combined_key.merge(lab_pd_key, on=['patientcode', 'vID'], how='inner')

    diag_pd = diag_pd.merge(combined_key, on=['patientcode', 'vID'], how='inner')
    med_pd = med_pd.merge(combined_key, on=['patientcode', 'vID'], how='inner')
    #     pro_pd = pro_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    lab_pd = lab_pd.merge(combined_key, on=['patientcode', 'vID'], how='inner')
    med_pd_duplicate = med_pd

    # flatten and merge
    diag_pd = diag_pd.groupby(by=['patientcode', 'vID'])['icd9cm'].unique().reset_index()
    med_pd = med_pd_duplicate.groupby(['patientcode', 'vID']).agg(lambda x: list(x))
    # print('Medicine_processed', med_pd.head())

    def uniqueIndexes(l):
        seen = set()
        res = []
        for i, n in enumerate(l):
            if n not in seen:
                res.append(i)
                seen.add(n)
        return res

    for index, row in med_pd.iterrows():
        indexes = uniqueIndexes(row['medDesc1'])
        #     item = row['ITEMID']
        item = [row['medDesc1'][i] for i in indexes]
        val = [row['dosage_norm'][i] for i in indexes]
        med_pd['medDesc1'][index] = item
        med_pd['dosage_norm'][index] = val

    lab_pd = lab_pd.groupby(by=['patientcode', 'vID'])['value_norm'].unique().reset_index()
    data = diag_pd.merge(med_pd, on=['patientcode', 'vID'], how='inner')
    data = data.merge(lab_pd, on=['patientcode', 'vID'], how='inner')

    print('Done')
    return data


data = process_all()
print(data.head())
print(data.columns)
# data.to_pickle('data_with_lab_test_polyclinic_icde.pkl')
data.to_pickle('data_with_lab_test_polyclinic_icde_final.pkl')