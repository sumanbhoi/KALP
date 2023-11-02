import torch
import argparse
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F
from collections import defaultdict
from scipy.stats import linregress

from models import KALP
from util import llprint, regression_metrics, get_n_params

torch.manual_seed(1203)
np.random.seed(1203)

model_name = 'KALP'
resume_name = ''


def patient_data_process(records, med_voc):
    record_new = []
    for step, input_p in enumerate(records):
        patient = []
        # print('patient', step)
        for idx, adm in enumerate(input_p):
            visit = []
            med_vec = np.zeros(len(med_voc.idx2word))
            med_vec[adm[1]] = adm[2]
            visit.append(adm[0])
            visit.append(adm[1])
            visit.append(list(med_vec))
            visit.append(adm[3])
            visit.append(adm[4])
            patient.append(visit)
        record_new.append(patient)
    return record_new

def eval(model, data_eval, voc_size, epoch):

    # evaluate
    print('')
    model.eval()
    rmse, mae, mape = [[] for _ in range(3)]

    visit_cnt = 0
    A_count = 0
    for step, input in enumerate(data_eval):
        A_count = A_count + 1
        y_gt = []
        y_pred = []

        for adm_idx, adm in enumerate(input):

            target_output1 = model(input[:adm_idx+1])
            y_gt.append(adm[3])

            target_output1 = target_output1.detach().cpu().numpy()[0]
            y_pred.append(target_output1)
            visit_cnt += 1


        y_gt = np.reshape(y_gt, (len(y_gt), 1))
        print(np.shape(y_gt), np.shape(y_pred))
        adm_rmse, adm_mae, adm_mape = regression_metrics(np.array(y_gt), np.array(y_pred))


        rmse.append(adm_rmse)
        mae.append(adm_mae)
        mape.append(adm_mape)
        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))


    llprint('\tRMSE: %.4f, \tMAE: %.4f, \tMAPE: %.4f\n' % (np.mean(rmse), np.mean(mae), np.mean(mape)))

    return np.mean(rmse), np.mean(mae), np.mean(mape)


def main():
    if not os.path.exists(os.path.join("saved_original", model_name)):
        os.makedirs(os.path.join("saved_original", model_name))

    data_path = '../Data/records_polyclinic_with_lab_test_polyclinic_ijcai_final.pkl'
    voc_path = '../Data/voc_polyclinic_with_lab_test_polyclinic_ijcai_final.pkl'
    voc_patient_path = '../Data/patient_voc_polyclinic_with_lab_test_polyclinic_ijcai_final.pkl'

    pi_adj_path = '../Data/pi_adj.pkl'
    ni_adj_path = '../Data/ni_adj.pkl'
    patient_sim_path = '../Data/patient_similarity.pkl'
    device = torch.device('cpu:0')

    pi_adj = dill.load(open(pi_adj_path, 'rb'))    # positive lab impact
    # print(np.shape(ehr_adj))
    ni_adj = dill.load(open(ni_adj_path, 'rb'))    # negative lab impact
    patient_sim = dill.load(open(patient_sim_path, 'rb'))    # patient similarity

    data_initial = dill.load(open(data_path, 'rb'))  # patient data
    voc = dill.load(open(voc_path, 'rb'))      # vocabulary of diagnosis and medications
    diag_voc, med_voc = voc['diag_voc'], voc['med_voc']
    # print('med_voc', len(med_voc.idx2word))
    voc_patient = dill.load(open(voc_patient_path, 'rb'))
    patient_voc = voc_patient['patient_voc']
    data = patient_data_process(data_initial, med_voc) # added long list of medication dosages at med locations

    # locations and the data : [0] is diag location; [1] is med locations; [2] is med vector with dosages; [3] is lab test val; [4] is patientcode

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]

    EPOCH = 20
    LR = 0.0001
    TEST = False
    # num_heads = 8
    # num_layers = 6
    # d_ff = 256
    # max_seq_length = 1000
    # dropout = 0.1

    voc_size = (len(diag_voc.idx2word), len(med_voc.idx2word), len(diag_voc.idx2word) + len(med_voc.idx2word) + 1, len(patient_sim))
# Here pass the parameters for the dimension etc.

    model = KALP(voc_size, pi_adj, ni_adj, patient_sim, patient_voc, emb_dim=64, num_heads=2, num_layers=2, d_ff=256, max_seq_length=1000, dropout=0.1, device=device)

    if TEST:
        model.load_state_dict(torch.load(open(os.path.join("saved_original", model_name, resume_name), 'rb')))
    model.to(device=device)

    print('parameters', get_n_params(model))
    optimizer = Adam(list(model.parameters()), lr=LR)

    if TEST:

        eval(model, data_test, voc_size, 0)

        print('in test')
    else:
        history = defaultdict(list)
        best_epoch = 0
        best_rmse = 0
        for epoch in range(EPOCH):
            loss_record1 = []
            start_time = time.time()
            model.train()
            prediction_loss_cnt = 0
            neg_loss_cnt = 0
            for step, input in enumerate(data_train):
                for idx, adm in enumerate(input):
                    seq_input = input[:idx+1]
                    loss_target = adm[3][0]

                    target_output = model(seq_input)

                    loss = F.mse_loss(target_output, torch.FloatTensor(loss_target).to(device))
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    loss_record1.append(loss.item())

                llprint('\rTrain--Epoch: %d, Step: %d/%d, L_p cnt: %d, L_neg cnt: %d' % (epoch, step, len(data_train), prediction_loss_cnt, neg_loss_cnt))


            rmse, mae, mape = eval(model, data_eval, voc_size, epoch)

            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            llprint('\tEpoch: %d, Loss: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                                np.mean(loss_record1),
                                                                                                elapsed_time,
                                                                                                elapsed_time * (
                                                                                                            EPOCH - epoch - 1)/60))

            torch.save(model.state_dict(), open( os.path.join('saved_original', model_name, 'Epoch_%d_rmse_%.4f.model' % (epoch, rmse)), 'wb'))
            print('')
            if epoch != 0 and best_rmse < rmse:
                best_epoch = epoch
                best_rmse = rmse



        # test
        torch.save(model.state_dict(), open(
            os.path.join('saved_original', model_name, 'final.model'), 'wb'))

        print('best_epoch:', best_epoch)


if __name__ == '__main__':
    main()
