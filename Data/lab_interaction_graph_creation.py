import pandas as pd
import numpy as np
import dill

path1 = './pos_impact.csv'
path2 = './neg_impact.csv'
path3 = './pos_impact_disease.csv'
path4 = './neg_impact_disease.csv'

pos_impact = pd.read_csv(path1)
neg_impact = pd.read_csv(path2)
pos_impact_disease = pd.read_csv(path3, encoding='unicode_escape')
neg_impact_disease = pd.read_csv(path4, encoding='unicode_escape')

# For Polyclinic
record_m = dill.load(open("../Data/records_polyclinic_with_lab_test_polyclinic_ijcai_final.pkl", 'rb'))
voc_m = dill.load(open("../Data/voc_polyclinic_with_lab_test_polyclinic_ijcai_final.pkl", 'rb'))

med_voc_m = voc_m['med_voc']
med_voc_size_m = len(med_voc_m.idx2word)
# print(med_voc_m.idx2word)
# print(med_voc_m.idx2word[3].lower())
#
# print('type', type(med_voc_m.word2idx['Metformin']))
diag_voc_m = voc_m['diag_voc']
diag_voc_size_m = len(diag_voc_m.idx2word)
# print(diag_voc_m.idx2word)

# Positive and Negative lab interaction adjacency matrix for polyclinic
final_size = med_voc_size_m + diag_voc_size_m + 1

pi_adj = np.zeros((final_size, final_size)) + 0.001
ni_adj = np.zeros((final_size, final_size)) + 0.001

# print(list(pos_impact['drug_name']))
vocab_med_set_m = list(med_voc_m.word2idx.keys())

for i in range(med_voc_size_m):
    if med_voc_m.idx2word[i].lower() in list(pos_impact['drug_name']) or med_voc_m.idx2word[i] in list(pos_impact['drug_name']):

        pi_adj[0, i + 1] = 1
        pi_adj[i + 1, 0] = 1
        # if med_voc_m.idx2word[i].lower() == 'metformin':
        #     print('index', i)

for i in range(diag_voc_size_m):
    if diag_voc_m.idx2word[i].lower() in list(pos_impact_disease['ICD9_code']) or diag_voc_m.idx2word[i] in list(pos_impact_disease['ICD9_code']):

        pi_adj[0, med_voc_size_m + i + 1] = 1
        pi_adj[med_voc_size_m + i + 1 + 1, 0] = 1
        # print('in disease')
        # if diag_voc_m.idx2word[i].lower() == '25000':
        #     print('index disease', i)




print(pi_adj)
print(np.nonzero(pi_adj))
print(len(pi_adj))
# dill.dump(pi_adj, open('pi_adj.pkl', 'wb'))

for i in range(med_voc_size_m):
    if med_voc_m.idx2word[i].lower() in list(neg_impact['drug_name']) or med_voc_m.idx2word[i] in list(neg_impact['drug_name']):

        ni_adj[0, i + 1] = 1
        ni_adj[i + 1, 0] = 1
        if med_voc_m.idx2word[i].lower() == 'metformin':
            print('index', i)

for i in range(diag_voc_size_m):
    if diag_voc_m.idx2word[i].lower() in list(neg_impact_disease['ICD9_code']) or diag_voc_m.idx2word[i] in list(neg_impact_disease['ICD9_code']):

        ni_adj[0, med_voc_size_m + i + 1] = 1
        ni_adj[med_voc_size_m + i + 1, 0] = 1
        print('in disease negative')
        if diag_voc_m.idx2word[i].lower() == '25000':
            print('index disease', i)

print(ni_adj)
print(np.nonzero(ni_adj))
print(len(ni_adj))
# dill.dump(ni_adj, open('ni_adj.pkl', 'wb'))

