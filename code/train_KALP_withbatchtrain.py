import torch
import gc
gc.collect()
torch.cuda.empty_cache()
import argparse
import numpy as np
import dill
import time
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"

import torch.nn.functional as F
from collections import defaultdict
from scipy.stats import linregress
from torch.utils.data import Dataset

from models_withbatchtrain import KALP
from util import llprint, regression_metrics, get_n_params
torch.set_float32_matmul_precision("medium")

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
# data_train = data[:split_point]
data_train = data[:10]
# train_loader = DataLoader(data_train, batch_size=2)
eval_len = int(len(data[split_point:]) / 2)
# data_test = data[split_point:split_point + eval_len]
# data_eval = data[split_point+eval_len:]
data_test = data[10:15]
data_eval = data[15:20]






def my_collate(batch):
    data = [item for item in batch]
    # target = [item[1] for item in batch]
    # target = torch.LongTensor(target)
    return data


# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(data_train, batch_size=2, shuffle=True, collate_fn=my_collate)
print(len(training_loader))
print(type(training_loader))
# print(training_loader[1])
# print(training_loader[2])
validation_loader = torch.utils.data.DataLoader(data_eval, batch_size=2, shuffle=False, collate_fn=my_collate)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=2, shuffle=False)

EPOCH = 3
LR = 0.0001
x_max = 14.0
x_min = 3.8
TEST = False
# num_heads = 8
# num_layers = 6
# d_ff = 256
# max_seq_length = 1000
# dropout = 0.1

voc_size = (len(diag_voc.idx2word), len(med_voc.idx2word), len(diag_voc.idx2word) + len(med_voc.idx2word) + 1, len(patient_sim))



def eval(model, data_eval, voc_size, epoch, x_max, x_min):

    # evaluate
    print('')
    model.eval()
    rmse, mae, mape = [[] for _ in range(3)]

    visit_cnt = 0
    # A_count = 0
    for step, input in enumerate(data_eval):
        # A_count = A_count + 1
        y_gt = []
        y_pred = []

        for adm_idx, adm in enumerate(input):

            target_output1 = model(input[:adm_idx+1])

            target_unnorm = (target_output1 * (x_max - x_min)) + x_min

            if len(adm[3][0]) > 1:
                gt = []
                gt.append(adm[3][0][0])
                GT_unnorm = (gt * (x_max - x_min)) + x_min
            else:
                GT_unnorm = (adm[3][0] * (x_max - x_min)) + x_min

            y_gt.append(GT_unnorm)
            # y_gt.append(adm[3])
            target_unnorm = target_unnorm.detach().cpu().numpy()[0]
            y_pred.append(target_unnorm)
            visit_cnt += 1


        y_gt = np.reshape(y_gt, (len(y_gt), 1))
        y_pred = np.reshape(y_pred, (len(y_pred), 1))
        print(np.shape(y_gt), np.shape(y_pred))
        adm_rmse, adm_mae, adm_mape = regression_metrics(np.array(y_gt), np.array(y_pred))


        rmse.append(adm_rmse)
        mae.append(adm_mae)
        mape.append(adm_mape)
        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))


    llprint('\tRMSE: %.4f, \tMAE: %.4f, \tMAPE: %.4f\n' % (np.mean(rmse), np.mean(mae), np.mean(mape)))

    return np.mean(rmse), np.mean(mae), np.mean(mape)


def train_one_epoc():
    loss_record1 = []
    last_loss = 0.
    print(np.shape(training_loader))
    for i, data_train in enumerate(training_loader):
        optimizer.zero_grad()
        for step, input in enumerate(data_train):
            for idx, adm in enumerate(input):
                if idx > 0:
                    seq_input = input[:idx]
                    if len(adm[3][0]) > 1:
                        lt = []
                        lt.append(adm[3][0][0])
                        loss_target = lt
                    else:
                        loss_target = adm[3][0]

                    target_output = model(seq_input)
                    print('target output training', target_output)
                    # target_unnorm = (target_output * (x_max - x_min)) + x_min
                    target_unnorm = target_output
                    print(target_unnorm)
                    GT_unnorm = (loss_target * (x_max - x_min)) + x_min
                    print(GT_unnorm)
                    # loss = F.mse_loss(target_output, torch.FloatTensor(loss_target).to(device))
                    loss = F.mse_loss(target_unnorm, torch.FloatTensor(GT_unnorm).to(device))
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    loss_record1.append(loss.item())

            # llprint('\rTrain--Epoch: %d, Step: %d/%d, L_p cnt: %d, L_neg cnt: %d' % (
            # epoch, step, len(data_train), prediction_loss_cnt, neg_loss_cnt))

        # rmse, mae, mape = eval(model, data_eval, voc_size, epoch, x_max, x_min)
    return np.mean(loss_record1)


def eval_one_epoc(test_var):
    model.eval()
    rmse, mae, mape = [[] for _ in range(3)]
    visit_cnt = 0
    loss_record1 = []
    y_gt = []
    y_pred = []
    if test_var:
        val_test_loader = test_loader
    else:
        val_test_loader = validation_loader

    with torch.no_grad():
        for i, data_val in enumerate(val_test_loader):
            for step, input in enumerate(data_val):
                for idx, adm in enumerate(input):
                    if idx > 0:
                        visit_cnt += 1
                        seq_input = input[:idx]
                        if len(adm[3][0]) > 1:
                            lt = []
                            lt.append(adm[3][0][0])
                            loss_target = lt
                        else:
                            loss_target = adm[3][0]

                        target_output = model(seq_input)

                        target_unnorm = (target_output * (x_max - x_min)) + x_min
                        target_unnorm = target_unnorm.detach().cpu().numpy()[0]
                        GT_unnorm = (loss_target * (x_max - x_min)) + x_min
                        # loss = F.mse_loss(target_output, torch.FloatTensor(loss_target).to(device))
                        # loss = F.mse_loss(target_unnorm, torch.FloatTensor(GT_unnorm).to(device))
                        # optimizer.zero_grad()
                        # loss.backward(retain_graph=True)
                        # optimizer.step()
                        print(type(GT_unnorm))
                        print(GT_unnorm)
                        print(type(target_unnorm))
                        # y_gt.append(GT_unnorm)
                        y_gt = np.append(y_gt, GT_unnorm)
                        # y_pred.append(target_unnorm)
                        y_pred = np.append(y_pred, target_unnorm)

            y_gt = np.reshape(y_gt, (len(y_gt), 1))
            y_pred = np.reshape(y_pred, (len(y_pred), 1))
            print(np.shape(y_gt), np.shape(y_pred))
            print('eval y_gt and y_pred', y_gt)
            print('eval y_gt and y_pred', y_pred)
            adm_rmse, adm_mae, adm_mape = regression_metrics(np.array(y_gt), np.array(y_pred))

            rmse.append(adm_rmse)
            mae.append(adm_mae)
            mape.append(adm_mape)
    return np.mean(rmse), np.mean(mae), np.mean(mape)

# def main():

# Here pass the parameters for the dimension etc.

model = KALP(voc_size, pi_adj, ni_adj, patient_sim, patient_voc, emb_dim=64, num_heads=2, num_layers=2, d_ff=256, max_seq_length=1000, dropout=0.1, device=device)

if TEST:
    model.load_state_dict(torch.load(open(os.path.join("saved_original", model_name, resume_name), 'rb')))
model.to(device=device)

print('parameters', get_n_params(model))
optimizer = Adam(list(model.parameters()), lr=LR)

if TEST:
    # eval(model, data_test, voc_size, 0, x_max, x_min)
    print('in test')
    rmse_test, mae_test, mape_test = eval_one_epoc(TEST)
    print('RMSE test', rmse_test)
    print('MAE test', mae_test)
    print('MAPE test', mape_test)

else:
    history = defaultdict(list)
    best_epoch = 0
    best_rmse = 0
    for epoch in range(EPOCH):
        loss_record1 = []
        start_time = time.time()
        model.train()
        # prediction_loss_cnt = 0
        # neg_loss_cnt = 0
        avg_loss = train_one_epoc()  # If want to analyze loss
        rmse, mae, mape = eval_one_epoc(TEST)
        # for step, input in enumerate(data_train):
        #     for idx, adm in enumerate(input):
        #         seq_input = input[:idx+1]
        #         if len(adm[3][0]) > 1:
        #             lt = []
        #             lt.append(adm[3][0][0])
        #             loss_target = lt
        #         else:
        #             loss_target = adm[3][0]
        #
        #         target_output = model(seq_input)
        #
        #         target_unnorm = (target_output * (x_max-x_min)) + x_min
        #         GT_unnorm = (loss_target * (x_max-x_min)) + x_min
        #         # loss = F.mse_loss(target_output, torch.FloatTensor(loss_target).to(device))
        #         loss = F.mse_loss(target_unnorm, torch.FloatTensor(GT_unnorm).to(device))
        #         optimizer.zero_grad()
        #         loss.backward(retain_graph=True)
        #         optimizer.step()
        #
        #         loss_record1.append(loss.item())
        #
        #     # llprint('\rTrain--Epoch: %d, Step: %d/%d, L_p cnt: %d, L_neg cnt: %d' % (epoch, step, len(data_train), prediction_loss_cnt, neg_loss_cnt))
        #
        #
        # rmse, mae, mape = eval(model, data_eval, voc_size, epoch, x_max, x_min)

        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        llprint('\tEpoch: %d, Loss: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch, avg_loss,
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


# if __name__ == '__main__':
#     main()
