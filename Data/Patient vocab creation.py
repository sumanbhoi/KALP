# Create patient dictionary

import dill
import pandas as pd

class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, word):
#         for word in sentence:
        if word not in self.word2idx:
            self.idx2word[len(self.word2idx)] = word
            self.word2idx[word] = len(self.word2idx)


def create_str_token_mapping(df):
    patient_voc = Voc()

    for index, row in df.iterrows():
        patient_voc.add_sentence(row['patientcode'])
    dill.dump(obj={'patient_voc': patient_voc},
              file=open('patient_voc_polyclinic_with_lab_test_polyclinic_ijcai_final.pkl', 'wb'))
    return patient_voc

path = 'data_with_lab_test_polyclinic_ijcai_final.pkl'
df = pd.read_pickle(path)
patient_voc = create_str_token_mapping(df)
