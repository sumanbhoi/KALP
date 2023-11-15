# Vocabulary Creation Polyclinic
import dill
import pandas as pd


class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


def create_str_token_mapping(df):
    diag_voc = Voc()
    med_voc = Voc()
    ## only for DMNC
    #     diag_voc.add_sentence(['seperator', 'decoder_point'])
    #     med_voc.add_sentence(['seperator', 'decoder_point'])
    #     pro_voc.add_sentence(['seperator', 'decoder_point'])

    for index, row in df.iterrows():
        diag_voc.add_sentence(row['icd9cm'])
        med_voc.add_sentence(row['medDesc1'])

    dill.dump(obj={'diag_voc': diag_voc, 'med_voc': med_voc},
              file=open('voc_polyclinic_with_lab_test_polyclinic_ijcai_final.pkl', 'wb'))
    # print(med_voc)
    return diag_voc, med_voc


def create_patient_record(df, diag_voc, med_voc):
    records = []  # (patient, code_kind:2, codes)  code_kind:diag,med
    for subject_id in df['patientcode'].unique():
        item_df = df[df['patientcode'] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([diag_voc.word2idx[i] for i in row['icd9cm']])
            admission.append([med_voc.word2idx[i] for i in row['medDesc1']])
            admission.append(row['dosage_norm'])
            admission.append([float(row['value_norm'][-1])])
            admission.append([subject_id])
            patient.append(admission)
        records.append(patient)
    dill.dump(obj=records, file=open('records_polyclinic_with_lab_test_polyclinic_ijcai_final.pkl', 'wb'))
    #     print(records)
    return records


path = 'data_with_lab_test_polyclinic_ijcai_final.pkl'
df = pd.read_pickle(path)
diag_voc, med_voc = create_str_token_mapping(df)
records = create_patient_record(df, diag_voc, med_voc)