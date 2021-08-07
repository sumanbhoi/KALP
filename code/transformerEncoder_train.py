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

from models import TransformerBased
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params, ddi_rate_score_analysis, ddi_rate_score_analysis_missed, missed_extra_analysis, regression_metrics

torch.manual_seed(1203)
np.random.seed(1203)

model_name = 'TransformerBased'
# resume_name = ''
# resume_name = 'Epoch_5_JA_0.4539_DDI_0.0624.model'
# resume_name = 'Epoch_10_JA_0.4831_DDI_0.0787.model'
# resume_name = 'Epoch_39_JA_0.5177_DDI_0.0791.model' # actual
# resume_name = 'Epoch_37_JA_0.5169_DDI_0.0812.model'
resume_name = 'Epoch_29_mse_1.3637.model'   # with loss 0.9 and 0.1
# resume_name = 'Epoch_39_JA_0.5029_DDI_0.0784.model'   # with loss 0.9 and 0.1 (trial)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--eval', action='store_true', default=False, help="eval mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_name, help='resume path')
parser.add_argument('--ddi', action='store_true', default=False, help="using ddi")

args = parser.parse_args()
# model_name = args.model_name
# resume_name = args.resume_path


def eval(model, data_eval, voc_size, epoch):
    # evaluate
    print('')
    model.eval()
    smm_record = []
    # smm_record_rare = []
    smm_record_gt = []
    # smm_record_primary = []
    ja, prauc, avg_p, avg_r, avg_f1, p_k, r_k = [[] for _ in range(7)]
    mse = []
    # ja_rare, prauc_rare, avg_p_rare, avg_r_rare, avg_f1_rare, p_k_rare, r_k_rare = [[] for _ in range(7)]
    # ja_pri, prauc_pri, avg_p_pri, avg_r_pri, avg_f1_pri, p_k_pri, r_k_pri = [[] for _ in range(7)]

    case_study = defaultdict(dict)
    # case_study_rare = defaultdict(dict)
    med_cnt = 0
    visit_cnt = 0
    med_cnt_gt = 0
    # med_primary_cnt = 0
    # med_primary_cnt_gt = 0
    # # To get the representation of the supplementary medication codes
    # voc_path = '../data/voc_final.pkl'
    # voc = dill.load(open(voc_path, 'rb'))
    # diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    # med_rep = med_voc.word2idx
    # diag_rep = diag_voc.idx2word
    A_count = 0
    count_2_or_more = 0
    rare_disease_patient = []
    for step, input in enumerate(data_eval):
        A_count = A_count + 1
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        y_gt_label = []
        for adm_idx, adm in enumerate(input):
            # print('adm_idx', adm_idx)
            # print('input', len(input[:adm_idx+1]))
            target_output1 = model(input[:adm_idx+1])
            y_gt_tmp = adm[2]
            # y_gt_tmp = np.zeros(voc_size[3])
            # for k in range(len(adm[3])):
            #     y_gt_tmp[adm[3][k]] = adm[4][k]
            # y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            # y_gt_label_tmp = adm[2]
            # y_gt_label.append(sorted(y_gt_label_tmp))

            # target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
            target_output1 = target_output1.detach().cpu().numpy()[0]
            y_pred.append(target_output1)
            # y_pred_tmp = target_output1.copy()
            # y_pred_tmp[y_pred_tmp>=0.5] = 1
            # y_pred_tmp[y_pred_tmp<0.5] = 0
            # y_pred.append(y_pred_tmp)

            # y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            # y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            # med_cnt += len(y_pred_label_tmp)

            # label_gt = np.where(y_gt_tmp == 1)[0]
            # med_cnt_gt += len(label_gt)

        #
        # smm_record.append(y_pred_label)
        # smm_record_gt.append(y_gt_label)
        # smm_record_primary.append(y_pred_primary_label)
        # adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1, adm_p_k, adm_r_k = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        # y_gt = np.array(y_gt).reshape(len(y_gt), 1)
        y_gt = np.reshape(y_gt, (len(y_gt), 1))
        # print('y_gt', y_gt)
        # print('y_pred', y_pred)
        print(np.shape(y_gt), np.shape(y_pred))
        adm_mse = regression_metrics(np.array(y_gt), np.array(y_pred))

        # case_study[A_count] = {'ja': adm_ja, 'patient': input, 'y_label': y_pred_label}
        #
        # ja.append(adm_ja)
        # prauc.append(adm_prauc)
        # avg_p.append(adm_avg_p)
        # avg_r.append(adm_avg_r)
        # avg_f1.append(adm_avg_f1)
        # p_k.append(adm_p_k)
        # r_k.append(adm_r_k)
        mse.append(adm_mse)
        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))

    # ddi rate
    # print(case_study)
    # ddi_rate = ddi_rate_score(smm_record)
    # ddi_rate_rare = ddi_rate_score(smm_record_rare)
    # ddi_rate_gt = ddi_rate_score(smm_record_gt)
    # print('count', A_count)
    # llprint('\tJaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
    #     np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
    # ))
    llprint('\tMSE: %.4f\n' % (np.mean(mse)))
    # print('DDI GT', ddi_rate_gt)
    #
    # print('precision at k = 1, 3, 5, 10, 20, 30', np.mean(p_k, axis=0))
    # print('recall at k = 1, 3, 5, 10, 20, 30', np.mean(r_k, axis=0))

    # # dill.dump(obj=smm_record, file=open('../data/gamenet_records.pkl', 'wb'))
    # # dill.dump(case_study, open(os.path.join('saved_original', model_name, 'case_study.pkl'), 'wb'))
    # # dill.dump(case_study_rare, open(os.path.join('saved_original', model_name, 'case_study_rare.pkl'), 'wb'))
    #
    # print('avg med', med_cnt / visit_cnt)
    # print('avg med primary', med_primary_cnt / visit_cnt)
    # print('rare_count', Number_rare)
    # return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
    return np.mean(mse)


def main():
    if not os.path.exists(os.path.join("saved_transformerbased_poly", model_name)):
        os.makedirs(os.path.join("saved_transformerbased_poly", model_name))

    data_path = '../data/records_lab_test_polyclinic_new.pkl'
    voc_path = '../data/voc_lab_test_polyclinic_new.pkl'

    # ehr_adj_path = '../data/ehr_adj_final.pkl'
    # ddi_adj_path = '../data/ddi_A_final.pkl'
    device = torch.device('cpu:0')

    # ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    # ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    data = dill.load(open(data_path, 'rb'))

    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, med_voc = voc['diag_voc'], voc['med_voc']
    # print(lab_voc.idx2word)
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]

    EPOCH = 40
    LR = 0.0002
    # TEST = args.eval
    TEST = False
    # Neg_Loss = args.ddi
    # Neg_Loss = True
    # DDI_IN_MEM = args.ddi
    # DDI_IN_MEM = True
    # TARGET_DDI = 0.05
    # T = 0.5
    # decay_weight = 0.85

    # voc_size = (len(diag_voc.idx2word), len(med_voc.idx2word))
    voc_size = (len(diag_voc.idx2word))
    # print(len(lab_voc.idx2word))
    model = TransformerBased(voc_size)
    # device = device
    if TEST:
        # model.load_state_dict(torch.load(open(resume_name, 'rb')))
        model.load_state_dict(torch.load(open(os.path.join("saved_transformerbased_poly", model_name, resume_name), 'rb')))
    model.to(device=device)
    print('parameters', get_n_params(model))
    optimizer = Adam(list(model.parameters()), lr=LR)

    if TEST:
        # med_dist = patient_distribution(data_test)
        eval(model, data_test, voc_size, 0)
        # analysis(model, data_test, voc_size, med_dist)
    else:
        history = defaultdict(list)
        best_epoch = 0
        best_ja = 0
        for epoch in range(EPOCH):
            loss_record1 = []
            start_time = time.time()
            model.train()
            prediction_loss_cnt = 0
            neg_loss_cnt = 0
            for step, input in enumerate(data_train):
                # print(input)
                for idx, adm in enumerate(input):
                    seq_input = input[:idx+1]
                    # loss_target = np.zeros((1, voc_size[3]))
                    loss_target = adm[2]

                    target_output1 = model(seq_input[0])
                    # print('target_output1', target_output1)

                    # loss1 = F.binary_cross_entropy_with_logits(target_output1, torch.FloatTensor(loss1_target).to(device))
                    # loss3 = F.multilabel_margin_loss(F.sigmoid(target_output1), torch.LongTensor(loss3_target).to(device))
                    loss = F.mse_loss(target_output1, torch.FloatTensor(loss_target).to(device))
                    # loss = 0.9 * loss1 + 0.1 * loss3

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    loss_record1.append(loss.item())

                llprint('\rTrain--Epoch: %d, Step: %d/%d, L_p cnt: %d, L_neg cnt: %d' % (epoch, step, len(data_train), prediction_loss_cnt, neg_loss_cnt))


            mse = eval(model, data_eval, voc_size, epoch)

            # history['ja'].append(ja)
            # history['ddi_rate'].append(ddi_rate)
            # history['avg_p'].append(avg_p)
            # history['avg_r'].append(avg_r)
            # history['avg_f1'].append(avg_f1)
            # history['prauc'].append(prauc)

            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            llprint('\tEpoch: %d, Loss: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                                np.mean(loss_record1),
                                                                                                elapsed_time,
                                                                                                elapsed_time * (
                                                                                                            EPOCH - epoch - 1)/60))

            torch.save(model.state_dict(), open( os.path.join('saved_transformerbased_poly', model_name, 'Epoch_%d_mse_%.4f.model' % (epoch, mse)), 'wb'))
            print('')
            if epoch != 0 and best_ja > mse:
                best_epoch = epoch
                best_ja = mse


        # dill.dump(history, open(os.path.join('saved_original', model_name, 'history.pkl'), 'wb'))

        # test
        # torch.save(model.state_dict(), open(
        #     os.path.join('saved_original', model_name, 'final.model'), 'wb'))

        print('best_epoch:', best_epoch)


if __name__ == '__main__':
    main()
