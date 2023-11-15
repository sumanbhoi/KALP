import pandas as pd
import numpy as np
import dill
from matplotlib import pyplot as plt
from numpy import float64
from patient_demographic_processing import patient_demo_process

# Patient similarity graph creation [diagnosis, med, lab]
data_path = '../Data/records_polyclinic_with_lab_test_polyclinic_ijcai_final.pkl'
voc_path = '../Data/voc_polyclinic_with_lab_test_polyclinic_ijcai_final.pkl'
data = dill.load(open(data_path, 'rb'))
voc = dill.load(open(voc_path, 'rb'))
diag_voc, med_voc = voc['diag_voc'], voc['med_voc']
print('med_voc', len(med_voc.idx2word))
print(data[0])
print(len(data))

#  Patient demographic info at 1st visit

df_info = patient_demo_process()
print(np.shape(df_info))

# Patient vector creation

# patient_voc_path = './patient_voc_polyclinic_with_lab_test_polyclinic_ijcai_final.pkl'
# patient_voc = dill.load(open(patient_voc_path, 'rb'))
diag_voc, med_voc = voc['diag_voc'], voc['med_voc']
vector_size = len(diag_voc.idx2word) + len(med_voc.idx2word) + 1
# patient_vector = np.zeros((len(data),vector_size))
patient_vector = []

# adm[4] is the subjectID or patientcode
count = 0
for step, input_data in enumerate(data):
    diag_vec = np.zeros(len(diag_voc.idx2word))
    med_vec = np.zeros(len(med_voc.idx2word))
    demo = np.zeros(3)                         # [gender, age, weight]
    p_vec = []
    count = count + 1
    for idx, adm in enumerate(input_data):
        #         print(type(adm[3][0]))
        if idx == 0:
            # Retrieve patient demographic info

            if adm[4] in df_info['patientcode'].values:
                demo[0] = df_info.loc[df_info['patientcode'] == adm[4][0], 'genderdesc']
                # print(demo[0])
                demo[1] = df_info.loc[df_info['patientcode'] == adm[4][0], 'Age']
                # print(demo[1])
                demo[2] = df_info.loc[df_info['patientcode'] == adm[4][0], 'resultvalue_nbr']
                # print(demo[2])

            diag_vec[adm[0]] = 1
            # print(demo)
            for i in range(len(adm[1])):
                med_vec[adm[1][i]] = adm[2][i]
            print(np.shape(adm[3][0]))
            
            ax = adm[3][0]
            print(type(adm[3][0]))
            ax1 = []
            if len(adm[3][0]) > 1:
                ax1.append(adm[3][0][0])
                ax = ax1
                print(type(ax))
            print(ax)
            p_vec = np.concatenate((diag_vec, med_vec, ax, demo))
            patient_vector.append(p_vec)
            # print(np.shape(adm[3][0]))
            # print(type(adm[3][0]))
            # print(adm[3][0][0])
            # ax = adm[3][0]
            # print(ax[0])
            # print('p_vec', len(p_vec))
            # print(patient_vector)
            print(np.shape(patient_vector))
            break

    # if count == 10:
    #     break

# print(type(med_vec))
# print(p_vec)
# print(patient_vector)
# print(np.shape(patient_vector))

#  Patient similarity calculation
from numpy import dot
from numpy.linalg import norm


def sim_func(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim


patient_sim = np.zeros((len(patient_vector), len(patient_vector)))
for i, p_i in enumerate(patient_vector):
    for j, p_j in enumerate(patient_vector):
        # print(np.shape(p_i))
        # print(np.shape(p_j))
        if j <= i:
            continue
        patient_sim[i, j] = sim_func(p_i, p_j)
        patient_sim[j, i] = patient_sim[i, j]
print(np.shape(patient_sim))
print(len(patient_sim))
dill.dump(patient_sim, open('patient_similarity.pkl', 'wb'))
print('Done')
