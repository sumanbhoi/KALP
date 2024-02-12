import pandas as pd
import numpy as np
import dill

#  Collect the set of drugs with negative and positive impact
sider_drugnames = pd.read_table('drug_names.tsv', header=None)

# Negative impact

df_sider_se = pd.read_table('meddra_all_se.tsv', header=None)
df_diabetes = df_sider_se[df_sider_se[5].str.contains('Diabetes mellitus')]
neg_impact = df_diabetes.drop(columns=[1, 2, 3, 4])
neg_impact = neg_impact.reset_index(drop=True)
# print(neg_impact.head(20))
neg_impact['drug_name'] = neg_impact[0].map(sider_drugnames.drop_duplicates().set_index(0)[1])
# print(neg_impact.head(20))
neg_impact['side_effect'] = neg_impact[5]
neg_impact.drop(columns=[0, 5],inplace=True)
# print(neg_impact.head(20))
neg_impact.to_csv('neg_impact.csv', index=False)

# Positive impact

df_sider_in = pd.read_table('meddra_all_indications.tsv', header=None)
df_in_diabetes = df_sider_in[df_sider_in[3].str.contains('Diabetes')]
# print(df_in_diabetes[3].unique())
df_in_diabetes = df_in_diabetes[~df_in_diabetes[3].str.contains('Diabetes Insipidus')]
# print(df_in_diabetes[3].unique())
# print(df_in_diabetes.head())
df_in_diabetes.drop(columns=[1, 2, 4, 5, 6], inplace=True)
pos_impact = df_in_diabetes.reset_index(drop=True)
# print(pos_impact.head())

pos_impact['drug_name'] = pos_impact[0].map(sider_drugnames.drop_duplicates().set_index(0)[1])
# print(neg_impact.head(20))
pos_impact['indicator'] = pos_impact[3]
pos_impact.drop(columns=[0, 3],inplace=True)


# Positive impact from MEDI dataset

df_medi = pd.read_csv('MEDI_01212013.csv')
# print('medi indicator', df_medi.head())
# print(type(df_medi['INDICATION_DESCRIPTION'][0]))
df_medi.dropna(inplace=True)
df_in_diabetes_medi = df_medi[df_medi['INDICATION_DESCRIPTION'].str.contains('Diabetes')]
df_in_diabetes_medi = df_in_diabetes_medi[~df_in_diabetes_medi['INDICATION_DESCRIPTION'].str.contains('Diabetes insipidus')]
# print(df_in_diabetes_medi['INDICATION_DESCRIPTION'].unique())
# print(df_in_diabetes_medi[df_in_diabetes_medi['DRUG_DESC'] == 'Metformin'].head())
df_in_diabetes_medi.drop(columns=['RXCUI_IN', 'ICD9', 'MENTIONEDBYRESOURCES', 'HIGHPRECISIONSUBSET', 'POSSIBLE_LABEL_USE'], inplace=True)
df_in_diabetes_medi = df_in_diabetes_medi.reset_index(drop=True)
df_in_diabetes_medi = df_in_diabetes_medi.rename(columns={"DRUG_DESC": "drug_name", "INDICATION_DESCRIPTION": "indicator"})
# print('indicator', df_in_diabetes_medi.head())

# print(len(df_in_diabetes_medi))
# print(len(pos_impact))
pos_impact_final = pos_impact.merge(df_in_diabetes_medi.drop_duplicates(), on=['drug_name','indicator'],
                   how='left', indicator=True)
# print(len(pos_impact_final))
# print(len(neg_impact))
pos_impact_final.to_csv('pos_impact.csv', index=False)