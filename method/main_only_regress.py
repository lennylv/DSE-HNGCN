import argparse
import math
from math import sqrt
import os
import pickle
import random
import time
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
import torch
from numpy import argmax
from numpy.linalg import svd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.AtomPairs import Pairs, Torsions
from scipy import io,stats
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, roc_curve, \
    precision_recall_curve
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data, Batch, DataLoader, HeteroData
from tqdm import tqdm

from utils import evaluate_others
# from getSamples import getTenFoldData
# from pubchemfp import GetPubChemFPs
# from myDataset import myDataset
from model_only_regress import myModel
from torch.utils.tensorboard import SummaryWriter   # SummaryWriter的作用是将数据以特定的格式存储到文件夹

def getSimTopK(sim_info,knn=20):
    np.fill_diagonal(sim_info, 0)
    # 对于每一行，找到最大的前10个元素，其余都置为0
    for k in range(sim_info.shape[0]):
        sorted_indices = np.argsort(sim_info[k])[::-1]  # 对每一行进行降序排序，得到元素的索引
        top_knn_indices = sorted_indices[:knn]  # 取前knn个元素的索引
        other_indices = sorted_indices[knn:]  # 其他元素的索引
        sim_info[k][other_indices] = 0  # 将其他元素置为0
    np.fill_diagonal(sim_info, 1)
    return sim_info

def getTestMasks(raw_freq,freq_masked,mask):
    # 统计训练集各频率类的位置
    train_class_count = []
    train_mask_list = []
    for i in range(0,6):
        mask_i = np.zeros((freq_masked.shape[0],freq_masked.shape[1]))
        mask_i[freq_masked == i] = 1
        train_mask_list.append(mask_i)
        train_class_count.append(np.count_nonzero(mask_i, axis=(0, 1)))

    # 统计测试集各频率类的位置
    asso_masked = np.copy(freq_masked)
    asso_masked[asso_masked!=0] = 1
    zero_posit = np.where(asso_masked - mask !=0,True,False)
    mask_0 = np.zeros((raw_freq.shape[0], raw_freq.shape[1]))
    mask_0[zero_posit] = 1
    test_class_count = [np.count_nonzero(mask_0, axis=(0, 1))]
    test_mask_list = [mask_0]
    test_posi = np.where(mask == 0, True, False)
    test_posi = test_posi.astype(int)
    test_truth = raw_freq * test_posi
    for i in range(1, 6):
        mask_i = np.zeros((raw_freq.shape[0], raw_freq.shape[1]))
        mask_i[test_truth == i] = 1
        test_mask_list.append(mask_i)
        test_class_count.append(np.count_nonzero(mask_i, axis=(0, 1)))

    return train_mask_list,test_mask_list

def getHeteNetData(obj_sim,feat_sim,obj_feat_asso,factor=1):
    np.fill_diagonal(obj_sim, 1)
    np.fill_diagonal(feat_sim, 1)
    D = np.diag(np.sum(obj_sim, axis=1))
    # 计算 D^(-1/2)
    D_inv_sqrt = np.sqrt(np.linalg.inv(D))
    if np.isnan(D_inv_sqrt).any():
        print("Obj的D^(-1/2)中存在NaN")
    # 计算 D^(-1/2) * S * D^(-1/2)
    obj_sim = np.dot(np.dot(D_inv_sqrt, obj_sim), D_inv_sqrt)
    obj_sim = obj_sim * factor
    # (b) 读取副作用语义相似性矩阵(numpy)
    # 计算 D，即每行元素之和的对角矩阵
    D = np.diag(np.sum(feat_sim, axis=1))
    # 计算 D^(-1/2)
    D_inv_sqrt = np.sqrt(np.linalg.inv(D))
    if np.isnan(D_inv_sqrt).any():
        print("Feat的D^(-1/2)中存在NaN")
    # 计算 D^(-1/2) * S * D^(-1/2)
    feat_sim = np.dot(np.dot(D_inv_sqrt, feat_sim), D_inv_sqrt)
    feat_sim = feat_sim * factor
    # (c) 读取频率矩阵,构造异构图邻接矩阵
    mat1 = np.hstack((obj_sim, obj_feat_asso))
    mat2 = np.hstack((obj_feat_asso.T, feat_sim))
    interactor_matrix = np.vstack((mat1, mat2))
    # 计算 D，即每行元素之和的对角矩阵
    D = np.diag(np.sum(interactor_matrix, axis=1))
    D_inv_sqrt = np.sqrt(np.linalg.inv(D))
    if np.isnan(D_inv_sqrt).any():
        print("Interactor的D^(-1/2)中存在NaN")
    interactor_matrix = np.dot(np.dot(D_inv_sqrt, interactor_matrix), D_inv_sqrt)
    interactor_matrix = torch.tensor(interactor_matrix, dtype=torch.float32)
    # (d) 构建零矩阵-频率矩阵
    obj_zero_matrix = np.matrix(
        np.zeros((obj_sim.shape[0], obj_sim.shape[1]), dtype=np.int8))
    feat_zero_matrix = np.matrix(
        np.zeros((feat_sim.shape[0], feat_sim.shape[1]), dtype=np.int8))
    mat1 = np.hstack((obj_zero_matrix, obj_feat_asso))
    mat2 = np.hstack((obj_feat_asso.T, feat_zero_matrix))
    hidden_init = np.vstack((mat1, mat2))
    feature_matrix = np.copy(hidden_init)
    feature_matrix = torch.tensor(feature_matrix, dtype=torch.float32)
    num_features = obj_sim.shape[0] + feat_sim.shape[0]  # 750+994=1744
    return interactor_matrix,feature_matrix,num_features


def getHeteNetData_Sim(obj_freq_sim,feat_freq_sim,obj_feat_asso,obj_sim,feat_sim,factor=1):
    np.fill_diagonal(obj_freq_sim, 1)
    np.fill_diagonal(feat_freq_sim, 1)
    D = np.diag(np.sum(obj_freq_sim, axis=1))
    # 计算 D^(-1/2)
    D_inv_sqrt = np.sqrt(np.linalg.inv(D))
    if np.isnan(D_inv_sqrt).any():
        print("Obj的D^(-1/2)中存在NaN")
    # 计算 D^(-1/2) * S * D^(-1/2)
    obj_freq_sim = np.dot(np.dot(D_inv_sqrt, obj_freq_sim), D_inv_sqrt)
    obj_freq_sim = obj_freq_sim * factor
    # (b) 读取副作用语义相似性矩阵(numpy)
    # 计算 D，即每行元素之和的对角矩阵
    D = np.diag(np.sum(feat_freq_sim, axis=1))
    # 计算 D^(-1/2)
    D_inv_sqrt = np.sqrt(np.linalg.inv(D))
    if np.isnan(D_inv_sqrt).any():
        print("Feat的D^(-1/2)中存在NaN")
    # 计算 D^(-1/2) * S * D^(-1/2)
    feat_freq_sim = np.dot(np.dot(D_inv_sqrt, feat_freq_sim), D_inv_sqrt)
    feat_freq_sim = feat_freq_sim * factor
    # (c) 读取频率矩阵,构造异构图邻接矩阵
    mat1 = np.hstack((obj_freq_sim, obj_feat_asso))
    mat2 = np.hstack((obj_feat_asso.T, feat_freq_sim))
    interactor_matrix = np.vstack((mat1, mat2))
    # 计算 D，即每行元素之和的对角矩阵
    D = np.diag(np.sum(interactor_matrix, axis=1))
    D_inv_sqrt = np.sqrt(np.linalg.inv(D))
    if np.isnan(D_inv_sqrt).any():
        print("Interactor的D^(-1/2)中存在NaN")
    interactor_matrix = np.dot(np.dot(D_inv_sqrt, interactor_matrix), D_inv_sqrt)
    interactor_matrix = torch.tensor(interactor_matrix, dtype=torch.float32)
    # (d) 构建零矩阵-频率矩阵
    obj_feat_zero_matrix = np.matrix(
        np.zeros(obj_feat_asso.shape, dtype=np.int8))
    mat1 = np.hstack((obj_sim, obj_feat_zero_matrix))
    mat2 = np.hstack((obj_feat_zero_matrix.T, feat_sim))
    hidden_init = np.vstack((mat1, mat2))
    feature_matrix = np.copy(hidden_init)
    feature_matrix = torch.tensor(feature_matrix, dtype=torch.float32)
    num_features = obj_sim.shape[0] + feat_sim.shape[0]  # 750+994=1744
    return interactor_matrix,feature_matrix,num_features


def get_data(frequency_masked,args):

    with open("../data/664_data/drug/drug_5fp_jaccard_664_664_np.pkl",'rb') as f:
        drug_5fp_jaccard = pickle.load(f)
    # with open("../data/664_data/drug/drug_morganfp_tanimoto_664_664_np.pkl",'rb') as f:
    #     drug_morganfp_tanimoto = pickle.load(f)
    with open("../data/664_data/drug/drug_go_jaccard_664_664_np.pkl",'rb') as f:
        drug_go_jaccard = pickle.load(f)
    with open("../data/664_data/drug/drug_disease_jaccard_664_664_np.pkl",'rb') as f:
        drug_disease_jaccard = pickle.load(f)
    with open("../data/664_data/drug/drug_gene_jaccard_664_664_np.pkl",'rb') as f:
        drug_gene_jaccard = pickle.load(f)
    # with open("../data/664_data/drug/sim/drug_MeSH_sim_664_664_np.pkl",'rb') as f:
    #     drug_mesh_sim = pickle.load(f)
    # with open("../data/664_data/drug/sim/drug_ATC_sim_664_664_np.pkl",'rb') as f:
    #     drug_atc_sim = pickle.load(f)
    # with open("../data/664_data/drug/sim/chem/drug_combined_score_664_664_np.pkl",'rb') as f:
    #     drug_combined_score_sim = pickle.load(f)
    # with open("../data/664_data/drug/sim/chem/drug_database_664_664_np.pkl",'rb') as f:
    #     drug_database_sim = pickle.load(f)
    # with open("../data/664_data/drug/sim/chem/drug_experimental_664_664_np.pkl",'rb') as f:
    #     drug_experimental_sim = pickle.load(f)
    # with open("../data/664_data/drug/sim/chem/drug_similarity_664_664_np.pkl",'rb') as f:
    #     drug_similarity_sim = pickle.load(f)
    # with open("../data/664_data/drug/sim/chem/drug_textmining_664_664_np.pkl",'rb') as f:
    #     drug_textmining_sim = pickle.load(f)
    with open("../data/664_data/drug/drug_3DInformax_664_256_np.pkl",'rb') as f:
        drug_3DInformax = pickle.load(f)

    with open("../data/664_data/side/side_effect_semantic_sim.pkl",'rb') as f:
        side_semantic_sim = pickle.load(f)
    with open("../data/664_data/side/side_MedDRA_994_2304_np.pkl",'rb') as f:
        side_medra_info = pickle.load(f)
    pca_side = PCA(n_components=256)
    side_medra = pca_side.fit_transform(side_medra_info)
    # print('PCA 信息保留比例： ')
    # print(pca_side.explained_variance_ratio_[:128])
    # print(sum(pca_side.explained_variance_ratio_[:128]))
    # print(pca_side.explained_variance_[:128])
    # print(sum(pca_side.explained_variance_[:128]))
    # drug_sim_info = [drug_5fp_jaccard,drug_go_jaccard,drug_disease_jaccard,drug_gene_jaccard,drug_mesh_sim,drug_combined_score_sim,drug_database_sim,drug_experimental_sim,drug_similarity_sim,drug_textmining_sim]
    drug_freq_sim = cosine_similarity(frequency_masked)
    side_freq_sim = cosine_similarity(frequency_masked.T)
    freq_sim_list = [drug_freq_sim,side_freq_sim]
    origin_info_list = [drug_3DInformax,side_medra]
    drug_5fp_jaccard[drug_5fp_jaccard<0.35] = 0
    drug_go_jaccard[drug_go_jaccard<0.2] = 0
    drug_disease_jaccard[drug_disease_jaccard<0.95] = 0
    drug_gene_jaccard[drug_gene_jaccard<0.95] = 0
    side_semantic_sim[side_semantic_sim<0.2] = 0
    drug_sim_info = [drug_5fp_jaccard,drug_go_jaccard,drug_disease_jaccard,drug_gene_jaccard]
    side_sim_info = [side_semantic_sim]
    # drug_sim_info = [getSimTopK(drug_5fp_jaccard),getSimTopK(drug_go_jaccard),getSimTopK(drug_disease_jaccard)]
    # side_sim_info = [getSimTopK(side_semantic_sim)]

    # U, S, VT = svd(frequency_masked)
    # S_matrix = np.zeros(frequency_masked.shape)
    # S_matrix[:frequency_masked.shape[0], :frequency_masked.shape[0]] = np.diag(S)
    # # 计算 S^(1/2)
    # S_sqrt = np.sqrt(S_matrix)
    # U = np.dot(U,S_sqrt)
    # V = np.dot(S_sqrt,VT)
    asso_masked = np.copy(frequency_masked)
    asso_masked[asso_masked!=0] = 1
    inm_list = []
    feat_list = []
    inm_list_asso = []
    feat_list_asso = []
    inm_list_sim = []
    feat_list_sim = []
    # train_knn = 20
    for i in range(len(drug_sim_info)):
        for j in range(len(side_sim_info)):
            # # 对于每一行，找到最大的前10个元素，其余都置为0
            # for k in range(drug_5fp_jaccard.shape[0]):
            #     sorted_indices = np.argsort(drug_sim_info[i][k])[::-1]  # 对每一行进行降序排序，得到元素的索引
            #     top_knn_indices = sorted_indices[:train_knn]  # 取前knn个元素的索引
            #     other_indices = sorted_indices[train_knn:]  # 其他元素的索引
            #     drug_sim_info[i][k][other_indices] = 0  # 将其他元素置为0

            interactor_matrix_freq, feature_matrix_freq, num_features_freq = getHeteNetData(drug_sim_info[i], side_sim_info[j],frequency_masked, factor=9)
            inm_list.append(interactor_matrix_freq)
            feat_list.append(feature_matrix_freq)


    return [drug_sim_info,side_sim_info,inm_list,feat_list,inm_list_asso,feat_list_asso,inm_list_sim,feat_list_sim,origin_info_list]

def train_test(raw_frequency,frequency_masked,mask,fold, args):
    data_list = get_data(frequency_masked,args)
    train_mask_list,test_mask_list = getTestMasks(raw_frequency,frequency_masked,mask)
    for i in range(len(train_mask_list)):
        train_mask_list[i] = torch.tensor(train_mask_list[i],dtype=torch.float32).to(args.device)
    for i in range(len(test_mask_list)):
        test_mask_list[i] = torch.tensor(test_mask_list[i],dtype=torch.float32).to(args.device)
    for i in range(len(data_list)):
        for j in range(len(data_list[i])):
            if type(data_list[i][j]) != torch.Tensor:
                data_list[i][j] = torch.tensor(data_list[i][j],dtype=torch.float32).to(args.device)
            else:
                data_list[i][j] = data_list[i][j].to(args.device)

    raw_frequency = torch.tensor(raw_frequency,dtype=torch.float32).to(args.device)
    frequency_masked = torch.tensor(frequency_masked,dtype=torch.float32).to(args.device)
    mask = torch.tensor(mask,dtype=torch.float32).to(args.device)


    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = myModel(args,drug_num=args.drug_num,side_num=args.side_num,drug_feats = len(data_list[0]), side_feats = len(data_list[1])).to(args.device)

    drug_index_train = [i for i in range(args.drug_num)]
    drug_index_test = [i for i in range(args.drug_num)]
    # drug_index = [i for i in range(args.drug_num)]
    trainset = torch.utils.data.TensorDataset(torch.LongTensor(drug_index_train))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(drug_index_test))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,num_workers=1, pin_memory=True)

    train_data = data_list[:]
    train_data.extend([train_mask_list,raw_frequency,frequency_masked,mask])
    test_data = data_list[:]
    test_data.extend([test_mask_list,raw_frequency,frequency_masked,mask])

    Regression_criterion = nn.MSELoss()
    Classification_criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0001)




    AUC_mn = 0
    AUPR_mn = 0

    rms_mn = 100000
    mae_mn = 100000
    endure_count = 0


    train_loss = []
    train_auc = []
    train_aupr = []
    train_rmse = []
    train_mae = []

    test_auc = []
    test_aupr = []
    test_rmse = []
    test_mae = []
    test_pcc = []
    best_epoch = 0

    folder = './metrics_WS/regress/fold'+str(fold)+"/runs/train/loss" # 指定存储文件夹
    writer = SummaryWriter(log_dir=folder) # 实例化SummaryWriter类，每30秒写入一次到硬盘


    folder_test = './metrics_WS/regress/fold'+str(fold)+"/runs/test/loss" # 指定存储文件夹
    writer_test = SummaryWriter(log_dir=folder_test) # 实例化SummaryWriter类，每30秒写入一次到硬盘

    folder_train_metric = './metrics_WS/regress/fold'+str(fold)+"/runs/train/metric" # 指定存储文件夹
    writer_train_metric = SummaryWriter(log_dir=folder_train_metric) # 实例化SummaryWriter类，每30秒写入一次到硬盘

    folder_test_metric = './metrics_WS/regress/fold'+str(fold)+"/runs/test/metric" # 指定存储文件夹
    writer_test_metric = SummaryWriter(log_dir=folder_test_metric) # 实例化SummaryWriter类，每30秒写入一次到硬盘

    # 使用sum()和eq()函数来统计非零元素的个数
    non_zero_count = torch.sum(frequency_masked.eq(0))


    for epoch in range(0, args.epochs):
        # if epoch % 100 == 0:
        #     _train,_test,train_batch_graph_data,test_batch_graph_data = getloader(args,drug_to_mol_graph)
        # ====================   training    ====================
        print("Epoch: %d ====================   training    ====================" %(epoch))
        # print(model.lamda_border)
        # print(model.lamda)
        # print(model.eps)
        start = time.time()
        # this_train_loss = train(model,_train,drug_side_frequency_fp,drug_side_association_fp,drug_side_frequency_atc,drug_side_association_atc,drug_fp_jaccard, drug_ATC_WHO_750_similarity, side_effect_semantic,interactor_matrix_1,interactor_matrix_2,interactor_matrix_3,features_1,features_2,features_3, optimizer, Regression_criterion, Classification_criterion,fold,epoch,args)
        this_train_loss = train(model,train_loader,train_data,optimizer,args)
        # scheduler.step()
        train_loss.append(this_train_loss)
        time_cost = time.time() - start
        print("Time: %.2f Epoch: %d <Train on Trainset>,train_loss: %.5f" % (
            time_cost, epoch,this_train_loss))
        writer.add_scalar('train_loss', this_train_loss, epoch+1)  # 把loss写入到文件夹中

        if (epoch+1)%50 == 0:
            # # # 首先，你需要获取优化器的学习率
            # # lr = optimizer.param_groups[0]['lr']
            # # print('Current learning rate:', lr)
            start = time.time()
            # t_i_auc, t_iPR_auc, t_rmse, t_mae, aupr_2 = test(model,_train,drug_side_frequency_fp,drug_side_association_fp,drug_side_frequency_atc,drug_side_association_atc,drug_fp_jaccard, drug_ATC_WHO_750_similarity, side_effect_semantic,interactor_matrix_1,interactor_matrix_2,interactor_matrix_3,features_1,features_2,features_3,args)
            t_i_auc, t_iPR_auc, t_rmse, t_mae, t_pcc = verify(model, test_loader, test_data, args)
            train_auc.append(t_i_auc)
            train_aupr.append(t_iPR_auc)
            train_rmse.append(t_rmse)
            train_mae.append(t_mae)
            time_cost = time.time() - start
            print("Time: %.2f Epoch: %d <Test on Trainset> RMSE: %.5f, MAE: %.5f, PCC: %.5f, AUC: %.5f, AUPR: %.5f " % (
                time_cost, epoch, t_rmse, t_mae, t_pcc, t_i_auc, t_iPR_auc))
            writer_train_metric.add_scalars('train_metric', {"train_auc": t_i_auc,"train_aupr": t_iPR_auc, "train_rmse": t_rmse, "train_mae": t_mae, "train_pcc": t_pcc}, epoch + 1)  # 把loss写入到文件夹中
            start = time.time()
            # i_auc, iPR_auc, rmse, mae, aupr_2 = test(model,_test,drug_side_frequency_fp,drug_side_association_fp,drug_side_frequency_atc,drug_side_association_atc,drug_fp_jaccard, drug_ATC_WHO_750_similarity, side_effect_semantic,interactor_matrix_1,interactor_matrix_2,interactor_matrix_3,features_1,features_2,features_3,args)
            test_loss_pos,test_loss_neg, drugAUC,drugAUPR, rmse, mae, pcc = test(model, test_loader, test_data, args)
            test_auc.append(drugAUC)
            test_aupr.append(drugAUPR)
            test_rmse.append(rmse)
            test_mae.append(mae)
            test_pcc.append(pcc)
            time_cost = time.time() - start
            print("Time: %.2f Epoch: %d <Test on Testset> RMSE: %.5f, MAE: %.5f, PCC: %.5f, AUC: %.5f, AUPR: %.5f " % (
                time_cost, epoch, rmse, mae, pcc, drugAUC, drugAUPR))
            writer_test.add_scalars('test_loss', {'positive':test_loss_pos, 'negtive':test_loss_neg}, epoch + 1)  # 把loss写入到文件夹中
            writer_test_metric.add_scalars('test_metric', {"test_auc": drugAUC,"test_aupr": drugAUPR, "test_rmse": rmse, "test_mae": mae, "test_pcc": pcc}, epoch + 1)  # 把loss写入到文件夹中

            if (epoch+1) == int(args.epochs*0.5):
                optimizer.param_groups[0]['lr']=0.0001

        # if endure_count > 10:
        #     break

        # break
    with open("./metrics_WS/regress/fold"+str(fold)+"/train_loss.txt", 'w') as train_loss_f:
        train_loss_f.write(str(train_loss))
    #
    with open("./metrics_WS/regress/fold"+str(fold)+"/train_auc.txt", 'w') as train_auc_f:
        train_auc_f.write(str(train_auc))
    with open("./metrics_WS/regress/fold"+str(fold)+"/train_aupr.txt", 'w') as train_aupr_f:
        train_aupr_f.write(str(train_aupr))
    with open("./metrics_WS/regress/fold"+str(fold)+"/train_rmse.txt", 'w') as train_rmse_f:
        train_rmse_f.write(str(train_rmse))
    with open("./metrics_WS/regress/fold"+str(fold)+"/train_mae.txt", 'w') as train_mae_f:
        train_mae_f.write(str(train_mae))

    with open("./metrics_WS/regress/fold"+str(fold)+"/test_auc.txt", 'w') as test_auc_f:
        test_auc_f.write(str(test_auc))
    with open("./metrics_WS/regress/fold"+str(fold)+"/test_aupr.txt", 'w') as test_aupr_f:
        test_aupr_f.write(str(test_aupr))
    with open("./metrics_WS/regress/fold"+str(fold)+"/test_rmse.txt", 'w') as test_rmse_f:
        test_rmse_f.write(str(test_rmse))
    with open("./metrics_WS/regress/fold"+str(fold)+"/test_mae.txt", 'w') as test_mae_f:
        test_mae_f.write(str(test_mae))
    with open("./metrics_WS/regress/fold"+str(fold)+"/test_pcc.txt", 'w') as test_pcc_f:
        test_pcc_f.write(str(test_pcc))

    # state_dict = torch.load("./0model.pt",map_location=torch.device('cpu'))
    # model.load_state_dict(state_dict)

    start = time.time()
    # i_auc, iPR_auc, rmse, mae, aupr_2 = test(model,_test,drug_side_frequency_fp,drug_side_association_fp,drug_side_frequency_atc,drug_side_association_atc,drug_fp_jaccard, drug_ATC_WHO_750_similarity, side_effect_semantic,interactor_matrix_1,interactor_matrix_2,interactor_matrix_3,features_1,features_2,features_3,args)
    drugAUC,drugAUPR, rmse, mae, pcc,metric = test_save(model, test_loader, test_data,fold, args)

    time_cost = time.time() - start
    print("Time: %.2f Epoch: %d <Test on Testset> RMSE: %.5f, MAE: %.5f, PCC: %.5f, AUC: %.5f, AUPR: %.5f " % (
        time_cost, best_epoch, rmse, mae, pcc, drugAUC, drugAUPR))
    # print('The best AUC/AUPR on Trainset: %.5f / %.5f' % (AUC_mn, AUPR_mn))
    # print('The RMSE/MAE on Trainset when best AUC/AUPR: %.5f / %.5f' % (rms_mn, mae_mn))
    # 保存训练好的模型
    model_name = str(fold) + 'model.pt'
    checkpointsFolder = "./metrics_WS/regress/model"
    isCheckpointExist = os.path.exists(checkpointsFolder)
    if not isCheckpointExist:
        os.makedirs(checkpointsFolder)
    torch.save(model.state_dict(), checkpointsFolder + "/" + model_name)

    return drugAUC, drugAUPR, rmse, mae, pcc,metric

# def loss_fun(output_freq,output_asso,output_avg, label,asso, lam=0.03, eps=0.5):#eps(ε) = 0.5,lam(α) = 0.03
def loss_fun(train_mask_list,batch_index,output_freq, label, lam=0.03, eps=0.2,delta=0.2,beta = 1):#eps(ε) = 0.5,lam(α) = 0.03
    x0 = torch.where(label == 0)
    x1 = torch.where(label != 0)
    loss1 = torch.sum((output_freq[x1] - label[x1]) ** 2) + lam * torch.sum((output_freq[x0] - eps) ** 2)#论文等式（10）：加权 ε（即eps） 不敏感损失函数

    # for i in range(1,len(train_mask_list)):
    #     mask_position = train_mask_list[i][batch_index]
    #     # pred_match = output_freq * mask_position
    #     # left = torch.where(pred_match != 0 and pred_match < i - delta)
    #     # right = torch.where(pred_match != 0 and pred_match > i + delta)
    #     pred_match = output_freq[mask_position.flatten()!=0]
    #     left = torch.where(pred_match < i - delta)
    #     right = torch.where(pred_match > i + delta)
    #     border_loss = torch.sum((pred_match[left] - (i - delta)) ** 2) + torch.sum((pred_match[right] - (i + delta)) ** 2)
    #     loss1 += beta * border_loss

    return loss1

def train(model,train_loader,train_data,optimizer,args):
    # train_data = [inm_list,feat_list,train_mask_list,raw_frequency,frequency_masked,mask]
    # test_data = [inm_list,feat_list,test_mask_list,raw_frequency,frequency_masked,mask]
    model.train()
    train_total_loss = []
    for i, batch_drug in enumerate(tqdm(train_loader,desc="Train......",unit="batch"), 0):
        batch_drug = batch_drug[0].to(args.device)
        optimizer.zero_grad()
        pred_freq,loss = model(train_data[:9], batch_drug,train_data[-2][batch_drug].flatten())
        loss.backward()
        optimizer.step()
        train_total_loss.append(loss.item())
    return sum(train_total_loss) / len(train_total_loss)

def verify(model, test_loader, test_data, args):
    """
    :param model:
    :param device:
    :param loader: 数据加载器
    :param sideEffectsGraph: 副作用图信息，
    :param raw: 原始数据
    :return: 所有的被mask的原始值，所有的被mask的预测值，都是1维
    """
    # 声明为张量
    total_preds = torch.FloatTensor()
    total_reals = torch.FloatTensor()
    total_preds_label = torch.FloatTensor()
    total_reals_label = torch.FloatTensor()

    singleDrug_auc = []
    singleDrug_aupr = []
    with torch.no_grad():
        model.eval()

        for i, batch_drug in enumerate(tqdm(test_loader,desc="Verify......",unit="batch"), 0):
        # for batch_idx, data in enumerate(test_loader):
        #     data = data.to(args.device)
        #     batch_drug = data.y
            train_ratings = test_data[-2][batch_drug].squeeze().cpu()
            if sum(train_ratings.flatten()) == 0:
                continue
            pred_freq = model(test_data[:9], batch_drug,None,training=False)
            this_pred_freq = pred_freq.squeeze().cpu()
            total_preds = torch.cat((total_preds, this_pred_freq[torch.where(train_ratings != 0)]), dim=0)
            total_reals = torch.cat((total_reals, train_ratings[torch.where(train_ratings != 0)]), dim=0)

            train_ratings[train_ratings != 0] = 1
            mask_position = test_data[-1][batch_drug].squeeze().cpu()
            posi = this_pred_freq[torch.where(train_ratings != 0)]  # 预测结果中，与被掩盖的位置相对应的药副对为正样本
            nege = this_pred_freq[torch.where(mask_position - train_ratings != 0)]  # 找到mask[index].flatten()和train_label对应位置元素相减不为0，即不同的元素的索引
            y = np.hstack((posi, nege))
            y_true = np.hstack((np.ones(len(posi)), np.zeros(len(nege))))
            singleDrug_auc.append(roc_auc_score(y_true, y))
            singleDrug_aupr.append(average_precision_score(y_true, y))
        drugAUC = sum(singleDrug_auc) / len(singleDrug_auc)
        drugAUPR = sum(singleDrug_aupr) / len(singleDrug_aupr)
        print('num of singleDrug_auc: ', len(singleDrug_auc))
        total_reals = total_reals.numpy()
        total_preds = total_preds.numpy()
        rmse = sqrt(((total_reals - total_preds) ** 2).mean())
        mae = mean_absolute_error(total_reals, total_preds)
        pcc = np.corrcoef(total_reals, total_preds)[0, 1]

    return drugAUC, drugAUPR, rmse, mae, pcc


def test(model, test_loader, test_data, args):
    """
    :param model:
    :param device:
    :param loader: 数据加载器
    :param sideEffectsGraph: 副作用图信息，
    :param raw: 原始数据
    :return: 所有的被mask的原始值，所有的被mask的预测值，都是1维
    """
    # 声明为张量
    total_preds = torch.FloatTensor()
    total_reals = torch.FloatTensor()

    total_preds_after_auc = []
    total_preds_after_aupr = []

    total_pred_664 = []
    total_pred_asso_664 = []

    test_total_loss_pos = []
    test_total_loss_neg = []

    singleDrug_auc = []
    singleDrug_aupr = []
    singleDrug_asso_auc = []
    singleDrug_asso_aupr = []
    singleDrug_asso_threshold_auc = []#根据AUC曲线确定的最佳阈值
    singleDrug_asso_threshold_aupr = []#根据PR曲线确定的最佳阈值
    singleDrug_auc_after_auc = []#两种预测结果根据AUC阈值结合后的单药AUC
    singleDrug_aupr_after_auc = []#两种预测结果根据AUC阈值结合后的单药AUPR
    singleDrug_auc_after_aupr = []#两种预测结果根据AUPR阈值结合后的单药AUC
    singleDrug_aupr_after_aupr = []#两种预测结果根据AUPR阈值结合后的单药AUPR

    with torch.no_grad():
        model.eval()

        for i, batch_drug in enumerate(tqdm(test_loader,desc="Test......",unit="batch"), 0):
            batch_drug = batch_drug[0].to(args.device)
            mask_position = test_data[-1][batch_drug].squeeze().cpu()
            pred_freq = model(test_data[:9], batch_drug,None, training=False)
            temp = pred_freq.tolist()
            total_pred_664.append(temp[0])
            # temp_asso = pred_asso.tolist()
            # total_pred_asso_664.append(temp_asso[0])
            if sum(mask_position.flatten()) == len(mask_position.flatten()):
                continue

            # pred_freq = model(test_batch_graph_data[i],sideEffectsGraph,interactor_matrix1,feature_matrix1,interactor_matrix2,feature_matrix2,interactor_matrix3,feature_matrix3,interactor_matrix4,feature_matrix4,[i])
            # pred_freq,pred_asso,pred_avg = model(test_batch_graph_data[i],sideEffectsGraph,interactor_matrix1,feature_matrix1,interactor_matrix2,feature_matrix2,interactor_matrix3,feature_matrix3,interactor_matrix4,feature_matrix4,[i])

            this_pred_freq = pred_freq.squeeze().cpu()
            # this_pred_asso = pred_asso.squeeze().cpu()
            this_truth_freq = test_data[-3][batch_drug].squeeze().cpu()
            train_ratings = test_data[-2][batch_drug].squeeze().cpu()
            total_preds = torch.cat((total_preds, this_pred_freq[torch.where(mask_position==0)]), dim=0)
            total_reals = torch.cat((total_reals, this_truth_freq[torch.where(mask_position==0)]), dim=0)


            train_ratings[train_ratings != 0] = 1
            this_row_pred_pos = this_pred_freq[torch.where(mask_position==0)]
            this_row_real_pos = this_truth_freq[torch.where(mask_position==0)]
            this_row_pred_neg = this_pred_freq[torch.where((mask_position - train_ratings) != 0)]
            this_row_real_neg = this_truth_freq[torch.where((mask_position - train_ratings) != 0)]
            test_loss_pos = torch.sum((this_row_pred_pos - this_row_real_pos) ** 2)
            test_loss_neg = torch.sum((this_row_pred_neg - this_row_real_neg) ** 2)
            test_total_loss_pos.append(test_loss_pos.item())
            test_total_loss_neg.append(test_loss_neg.item())

            posi = this_pred_freq[torch.where(mask_position==0)]  # 预测结果中，与被掩盖的位置相对应的药副对为正样本
            nege = this_pred_freq[torch.where((mask_position - train_ratings)!=0)]  # 找到mask[index].flatten()和train_label对应位置元素相减不为0，即不同的元素的索引
            y = np.hstack((posi.data.numpy(), nege.data.numpy()))
            y_true = np.hstack((np.ones(len(posi)), np.zeros(len(nege))))
            singleDrug_auc.append(roc_auc_score(y_true, y))
            singleDrug_aupr.append(average_precision_score(y_true, y))
            # posi = this_pred_asso[torch.where(mask_position==0)]  # 预测结果中，与被掩盖的位置相对应的药副对为正样本
            # nege = this_pred_asso[torch.where((mask_position - train_ratings)!=0)]  # 找到mask[index].flatten()和train_label对应位置元素相减不为0，即不同的元素的索引
            # y = np.hstack((posi.data.numpy(), nege.data.numpy()))
            # y_true = np.hstack((np.ones(len(posi)), np.zeros(len(nege))))
            # singleDrug_asso_auc.append(roc_auc_score(y_true, y))
            # singleDrug_asso_aupr.append(average_precision_score(y_true,y))
            # #根据关联预测结果计算分类指标
            # this_pred_asso = pred_asso.squeeze().cpu()
            # posi = this_pred_asso[torch.where(mask_position==0)]  # 预测结果中，与被掩盖的位置相对应的药副对为正样本
            # nege = this_pred_asso[torch.where((mask_position - train_ratings)!=0)]  # 找到mask[index].flatten()和train_label对应位置元素相减不为0，即不同的元素的索引
            # y_asso = np.hstack((posi.data.numpy(), nege.data.numpy()))
            # singleDrug_asso_auc.append(roc_auc_score(y_true, y_asso))
            # singleDrug_asso_aupr.append(average_precision_score(y_true, y_asso))
            # #找出AUC曲线最佳阈值
            # fpr,tpr,thresholds = roc_curve(y_true,y_asso)
            # J = tpr - fpr
            # idx = argmax(J)
            # auc_best_thrshold = thresholds[idx]
            # singleDrug_asso_threshold_auc.append(auc_best_thrshold)
            #
            # precision, recall, thresholds = precision_recall_curve(y_true, y_asso)
            # temp1 = 2 * precision * recall
            # temp2 = precision + recall
            # temp2[temp2==0] = 1e-8
            # F1 =  temp1 / temp2
            # idx = argmax(F1)
            # aupr_best_thrshold = thresholds[idx]
            # singleDrug_asso_threshold_aupr.append(aupr_best_thrshold)
            #
            # #根据阈值将关联预测结果转为01
            # pred_freq_thred_auc = np.copy(pred_freq.squeeze().cpu().numpy())
            # pred_asso_thred_auc = np.copy(pred_asso.squeeze().cpu().numpy())
            # pred_asso_thred_auc[pred_asso_thred_auc<auc_best_thrshold] = 0
            # pred_asso_thred_auc[pred_asso_thred_auc>=auc_best_thrshold] = 1
            # pred_freq_thred_auc = pred_freq_thred_auc * pred_asso_thred_auc
            # # temp = torch.where(mask_position == 0)
            # posi = pred_freq_thred_auc[torch.where(mask_position == 0)]  # 预测结果中，与被掩盖的位置相对应的药副对为正样本
            # nege = pred_freq_thred_auc[torch.where((mask_position - train_ratings) != 0)]  # 找到mask[index].flatten()和train_label对应位置元素相减不为0，即不同的元素的索引
            # y_after_auc = np.hstack((posi, nege))
            # singleDrug_auc_after_auc.append(roc_auc_score(y_true, y_after_auc))
            # singleDrug_aupr_after_auc.append(average_precision_score(y_true, y_after_auc))
            # if isinstance(posi,np.float32):
            #     total_preds_after_auc.append(posi)
            # else:
            #     total_preds_after_auc.extend(posi.tolist())
            #
            # pred_freq_thred_aupr = np.copy(pred_freq.squeeze().cpu().numpy())
            # pred_asso_thred_aupr = np.copy(pred_asso.squeeze().cpu().numpy())
            # pred_asso_thred_aupr[pred_asso_thred_aupr<aupr_best_thrshold] = 0
            # pred_asso_thred_aupr[pred_asso_thred_aupr>=aupr_best_thrshold] = 1
            # pred_freq_thred_aupr = pred_freq_thred_aupr * pred_asso_thred_aupr
            # # temp = torch.where(mask_position == 0)
            # posi = pred_freq_thred_aupr[torch.where(mask_position == 0)]  # 预测结果中，与被掩盖的位置相对应的药副对为正样本
            # nege = pred_freq_thred_aupr[torch.where((mask_position - train_ratings) != 0)]  # 找到mask[index].flatten()和train_label对应位置元素相减不为0，即不同的元素的索引
            # y_after_aupr = np.hstack((posi, nege))
            # singleDrug_auc_after_aupr.append(roc_auc_score(y_true, y_after_aupr))
            # singleDrug_aupr_after_aupr.append(average_precision_score(y_true, y_after_aupr))
            #
            # if isinstance(posi,np.float32):
            #     total_preds_after_aupr.append(posi)
            # else:
            #     total_preds_after_aupr.extend(posi.tolist())
        print('num of singleDrug: ', len(singleDrug_auc))
        # drugAUC_asso = sum(singleDrug_asso_auc) / len(singleDrug_asso_auc)
        # drugAUPR_asso = sum(singleDrug_asso_aupr) / len(singleDrug_asso_aupr)
        # print('drugAUC_asso:%.5f, drugAUPR_asso:%.5f '%(drugAUC_asso,drugAUPR_asso))

        drugAUC = sum(singleDrug_auc) / len(singleDrug_auc)
        drugAUPR = sum(singleDrug_aupr) / len(singleDrug_aupr)
        total_reals = total_reals.numpy()
        total_preds = total_preds.numpy()
        rmse = sqrt(((total_reals - total_preds) ** 2).mean())
        mae = mean_absolute_error(total_reals, total_preds)
        pcc = np.corrcoef(total_reals, total_preds)[0, 1]
        spearman = stats.spearmanr(total_reals, total_preds)[0]
        metric_1 = [rmse,mae,pcc,spearman]
        print("Before thred: RMSE:%.5f, MAE:%.5f, PCC:%.5f, Spearman:%.5f, AUC:%.5f, AUPR:%.5f"%(rmse,mae,pcc,spearman,drugAUC,drugAUPR))


        # drugAUC_after_auc = sum(singleDrug_auc_after_auc) / len(singleDrug_auc_after_auc)
        # drugAUPR_after_auc = sum(singleDrug_aupr_after_auc) / len(singleDrug_aupr_after_auc)
        # total_preds_after_auc = np.array(total_preds_after_auc)
        # rmse_after_auc = sqrt(((total_reals - total_preds_after_auc) ** 2).mean())
        # mae_after_auc = mean_absolute_error(total_reals, total_preds_after_auc)
        # pcc_after_auc = np.corrcoef(total_reals, total_preds_after_auc)[0, 1]
        # spearman_after_auc = stats.spearmanr(total_reals, total_preds_after_auc)[0]
        # print("After thred AUC: RMSE:%.5f, MAE:%.5f, PCC:%.5f, Spearman:%.5f, AUC:%.5f, AUPR:%.5f" % (rmse_after_auc, mae_after_auc, pcc_after_auc, spearman_after_auc,drugAUC_after_auc,drugAUPR_after_auc))
        #
        # drugAUC_after_aupr = sum(singleDrug_auc_after_aupr) / len(singleDrug_auc_after_aupr)
        # drugAUPR_after_aupr = sum(singleDrug_aupr_after_aupr) / len(singleDrug_aupr_after_aupr)
        # total_preds_after_aupr = np.array(total_preds_after_aupr)
        # rmse_after_aupr = sqrt(((total_reals - total_preds_after_aupr) ** 2).mean())
        # mae_after_aupr = mean_absolute_error(total_reals, total_preds_after_aupr)
        # pcc_after_aupr = np.corrcoef(total_reals, total_preds_after_aupr)[0, 1]
        # spearman_after_aupr = stats.spearmanr(total_reals, total_preds_after_aupr)[0]
        # print("After thred AUPR: RMSE:%.5f, MAE:%.5f, PCC:%.5f, Spearman:%.5f, AUC:%.5f, AUPR:%.5f" % (rmse_after_aupr, mae_after_aupr, pcc_after_aupr, spearman_after_aupr,drugAUC_after_aupr,drugAUPR_after_aupr))


        total_pred_664 = np.array(total_pred_664)
        # with open("./pred_664_994_np.pkl",'wb') as f:
        #     pickle.dump(total_pred_664,f)
        avg_pos_loss = sum(test_total_loss_pos) / len(test_total_loss_pos)
        avg_neg_loss = sum(test_total_loss_neg) / len(test_total_loss_neg)
        # print("<Test on Testset> loss of positive: %.5f, loss of negtive: %.5f" % (sum(test_total_loss_pos) / len(test_total_loss_pos),  sum(test_total_loss_neg) / len(test_total_loss_neg)))
        mask = test_data[-1].cpu().numpy()
        truth = test_data[-3].cpu().numpy()
        train_label = test_data[-2].cpu().numpy()
        pos = total_pred_664[np.where(mask == 0)]
        pos_label = np.ones(len(pos))

        neg = total_pred_664[np.where(truth == 0)]
        neg_label = np.zeros(len(neg))

        y = np.hstack((pos, neg))
        y_true = np.hstack((pos_label, neg_label))
        auc_all = roc_auc_score(y_true, y)
        aupr_all = average_precision_score(y_true, y)
        # others
        Tr_neg = {}  # key为药物id,value为出现在训练集的负样本中每种药物对应的所有副作用
        Te = {}  # key为药物id,value为出现在测试集的正样本中每种药物对应的所有副作用
        Te_pairs = np.where(mask == 0)  # 测试集的正样本
        Tr_neg_pairs = np.where(train_label == 0)  # 训练集的负样本
        Te_pairs = np.array(Te_pairs).transpose()
        Tr_neg_pairs = np.array(Tr_neg_pairs).transpose()
        for te_pair in Te_pairs:
            drug_id = te_pair[0]
            SE_id = te_pair[1]
            if drug_id not in Te:
                Te[drug_id] = [SE_id]
            else:
                Te[drug_id].append(SE_id)

        for te_pair in Tr_neg_pairs:
            drug_id = te_pair[0]
            SE_id = te_pair[1]
            if drug_id not in Tr_neg:
                Tr_neg[drug_id] = [SE_id]
            else:
                Tr_neg[drug_id].append(SE_id)

        positions = [1, 5, 10, 15]
        map_value, auc_value, ndcg, prec, rec = evaluate_others(total_pred_664, Tr_neg, Te, positions)
        p1, p5, p10, p15 = prec[0], prec[1], prec[2], prec[3]
        r1, r5, r10, r15 = rec[0], rec[1], rec[2], rec[3]
        metric_2 = [auc_all, aupr_all, drugAUC, drugAUPR, map_value, ndcg, p1, p5, p10, p15, r1, r5, r10, r15]
        metric_1.extend(metric_2)
        print(metric_1)
    return avg_pos_loss,avg_neg_loss,drugAUC, drugAUPR, rmse, mae, pcc


def test_save(model, test_loader, test_data, fold, args):
    """
    :param model:
    :param device:
    :param loader: 数据加载器
    :param sideEffectsGraph: 副作用图信息，
    :param raw: 原始数据
    :return: 所有的被mask的原始值，所有的被mask的预测值，都是1维
    """
    # 声明为张量
    total_preds = torch.FloatTensor()
    total_reals = torch.FloatTensor()
    total_preds_label = torch.FloatTensor()
    total_reals_label = torch.FloatTensor()
    total_pred_664 = []

    singleDrug_auc = []
    singleDrug_aupr = []
    with torch.no_grad():
        model.eval()

        for i, batch_drug in enumerate(tqdm(test_loader,desc="Test......",unit="batch"), 0):
        # for batch_idx, data in enumerate(test_loader):
        #     data = data.to(args.device)
        #     batch_drug = data.y
            batch_drug = batch_drug[0].to(args.device)
            mask_position = test_data[-1][batch_drug].squeeze().cpu()
            pred_freq = model(test_data[:9], batch_drug,None, training=False)
            temp = pred_freq.tolist()
            total_pred_664.append(temp[0])
            if sum(mask_position.flatten()) == len(mask_position.flatten()):
                continue
            # pred_freq = model(test_batch_graph_data[i],sideEffectsGraph,interactor_matrix1,feature_matrix1,interactor_matrix2,feature_matrix2,interactor_matrix3,feature_matrix3,interactor_matrix4,feature_matrix4,[i])
            # pred_freq,pred_asso,pred_avg = model(test_batch_graph_data[i],sideEffectsGraph,interactor_matrix1,feature_matrix1,interactor_matrix2,feature_matrix2,interactor_matrix3,feature_matrix3,interactor_matrix4,feature_matrix4,[i])

            this_pred_freq = pred_freq.squeeze().cpu()
            this_truth_freq = test_data[-3][batch_drug].squeeze().cpu()
            train_ratings = test_data[-2][batch_drug].squeeze().cpu()
            total_preds = torch.cat((total_preds, this_pred_freq[torch.where(mask_position==0)]), dim=0)
            total_reals = torch.cat((total_reals, this_truth_freq[torch.where(mask_position==0)]), dim=0)

            train_ratings[train_ratings != 0] = 1
            posi = this_pred_freq[torch.where(mask_position==0)]  # 预测结果中，与被掩盖的位置相对应的药副对为正样本
            nege = this_pred_freq[torch.where((mask_position - train_ratings)!=0)]  # 找到mask[index].flatten()和train_label对应位置元素相减不为0，即不同的元素的索引
            y = np.hstack((posi.data.numpy(), nege.data.numpy()))
            y_true = np.hstack((np.ones(len(posi)), np.zeros(len(nege))))
            singleDrug_auc.append(roc_auc_score(y_true, y))
            singleDrug_aupr.append(average_precision_score(y_true, y))
        drugAUC = sum(singleDrug_auc) / len(singleDrug_auc)
        drugAUPR = sum(singleDrug_aupr) / len(singleDrug_aupr)
        print('num of singleDrug_auc: ', len(singleDrug_auc))
        total_reals = total_reals.numpy()
        total_preds = total_preds.numpy()
        rmse = sqrt(((total_reals - total_preds) ** 2).mean())
        mae = mean_absolute_error(total_reals, total_preds)
        pcc = np.corrcoef(total_reals, total_preds)[0, 1]
        spearman = stats.spearmanr(total_reals, total_preds)[0]
        metric_1 = [rmse,mae,pcc,spearman]

        total_pred_664 = np.array(total_pred_664)
        with open("./metrics_WS/regress/pred_value/"+str(fold)+"_pred_664_994_np.pkl",'wb') as f:
            pickle.dump(total_pred_664,f)
        # print("<Test on Testset> loss of positive: %.5f, loss of negtive: %.5f" % (sum(test_total_loss_pos) / len(test_total_loss_pos),  sum(test_total_loss_neg) / len(test_total_loss_neg)))
        mask = test_data[-1].cpu().numpy()
        truth = test_data[-3].cpu().numpy()
        train_label = test_data[-2].cpu().numpy()
        pos = total_pred_664[np.where(mask == 0)]
        pos_label = np.ones(len(pos))

        neg = total_pred_664[np.where(truth == 0)]
        neg_label = np.zeros(len(neg))

        y = np.hstack((pos, neg))
        y_true = np.hstack((pos_label, neg_label))
        auc_all = roc_auc_score(y_true, y)
        aupr_all = average_precision_score(y_true, y)
        # others
        Tr_neg = {}  # key为药物id,value为出现在训练集的负样本中每种药物对应的所有副作用
        Te = {}  # key为药物id,value为出现在测试集的正样本中每种药物对应的所有副作用
        Te_pairs = np.where(mask == 0)  # 测试集的正样本
        Tr_neg_pairs = np.where(train_label == 0)  # 训练集的负样本
        Te_pairs = np.array(Te_pairs).transpose()
        Tr_neg_pairs = np.array(Tr_neg_pairs).transpose()
        for te_pair in Te_pairs:
            drug_id = te_pair[0]
            SE_id = te_pair[1]
            if drug_id not in Te:
                Te[drug_id] = [SE_id]
            else:
                Te[drug_id].append(SE_id)

        for te_pair in Tr_neg_pairs:
            drug_id = te_pair[0]
            SE_id = te_pair[1]
            if drug_id not in Tr_neg:
                Tr_neg[drug_id] = [SE_id]
            else:
                Tr_neg[drug_id].append(SE_id)

        positions = [1, 5, 10, 15]
        map_value, auc_value, ndcg, prec, rec = evaluate_others(total_pred_664, Tr_neg, Te, positions)
        p1, p5, p10, p15 = prec[0], prec[1], prec[2], prec[3]
        r1, r5, r10, r15 = rec[0], rec[1], rec[2], rec[3]
        metric_2 = [auc_all, aupr_all, drugAUC, drugAUPR, map_value, ndcg, p1, p5, p10, p15, r1, r5, r10, r15]
        metric_1.extend(metric_2)
        print(metric_1)

    return drugAUC, drugAUPR, rmse, mae, pcc,metric_1


def generateMat(raw_frequency, seed,mode='WS', k_fold=10):
    """
    将矩阵按比例mask, 将被mask的部分分为10份，生成10份mask位置矩阵，保存在./data_WS/processed/mask_mat.mat
    :return:
    """
    # 每次加载都把之前的数据删除

    mask_file_path = "../data/664_data/data_" + mode + "_seed_" + str(seed) + "/tenfold_mask.mat"
    out_dir_path = mask_file_path[:mask_file_path.rfind('/')]
    if os.path.exists(mask_file_path):
        filenames = os.listdir(out_dir_path)
        print(filenames)
        for s in filenames:
            os.remove(out_dir_path + "/" + s)
    if os.path.exists(out_dir_path):
        pass
    else:
        os.makedirs(out_dir_path)  # 使用os.makedirs()方法创建多层目录


    # with open("../dataset/frequency_593_994.pkl",'rb') as f:
    #     raw = pickle.load(f)
    if mode == "WS":
        # mask, get mask Mat
        index_pair = np.where(raw_frequency != 0)#找到药副对频率矩阵R不为0的元素的下标。包含两个tuple，第一个存行坐标，第二个存列坐标。
        index_arr = np.arange(0, index_pair[0].shape[0], 1)#0到index_pair第一维长度。左闭右开。步长为1
        np.random.seed(seed)
        np.random.shuffle(index_arr)#打乱
        x = []
        n = math.ceil(index_pair[0].shape[0] / k_fold)#math.ceil():“向上取整”， 即小数部分直接舍去，并向正数部分进1
        for i in range(k_fold):#将正样本分成10份
            if i == k_fold - 1:
                x.append(index_arr[0:].tolist())
            else:
                x.append(index_arr[0:n].tolist())
                index_arr = index_arr[n:]

        dic = {}
        for i in range(k_fold):
            mask = np.ones(raw_frequency.shape)#750 * 994的全一矩阵
            mask[index_pair[0][x[i]], index_pair[1][x[i]]] = 0#将十分之一的正样本所在位置置为0，正样本指的是raw中不为0的元素
            dic['mask' + str(i)] = mask
        io.savemat(mask_file_path, dic)#文件中存放10个矩阵 750 × 994，每个矩阵为0的地方即是被掩盖的正样本，被掩盖指的是置为0
    else:
        # mask, get mask Mat
        index = np.arange(0, len(raw_frequency), 1)
        np.random.shuffle(index)
        x = []
        n = int(np.ceil(len(index) / 10))
        for i in range(10):
            if i == 9:
                x.append(index.tolist())
            x.append(index[0:n].tolist())
            index = index[n:]

        dic = {}
        for i in range(10):
            mask = np.ones(raw_frequency.shape)
            mask[x[i]] = 0
            dic['mask' + str(i)] = mask
        io.savemat(mask_file_path, dic)


def tenfold(args,seed=42):
    with open("../data/664_data/frequency_664_994_np.pkl", 'rb') as f:
        raw_frequency = pickle.load(f)
    mask_file_path = "../data/664_data/data_WS_seed_"+str(seed)+"/tenfold_mask.mat"
    if os.path.exists(mask_file_path):
        mask_dic = io.loadmat(mask_file_path)
    else:
        generateMat(raw_frequency,seed=seed)
        mask_dic = io.loadmat(mask_file_path)
    #保存每一折的最终预测结果
    try:
        File_Path = "./metrics_WS/regress/pred_value/"
        print(File_Path)
        # 判断是否已经存在该目录
        if not os.path.exists(File_Path):
            # 目录不存在，进行创建操作
            os.makedirs(File_Path)  # 使用os.makedirs()方法创建多层目录
            print("目录新建成功：" + File_Path)
        else:
            print("目录已存在！！！")
    except BaseException as msg:
        print("新建目录失败：" + msg)
    total_auc, total_pr_auc, total_rmse, total_mae,total_pcc = [], [], [], [], []
    total_metrics = []
    for fold in range(0,10):
        start_time = datetime.now()
        print("==================================fold {} start".format(fold))
        try:
            File_Path = "./metrics_WS/regress/fold"+str(fold)+"/during_train/"
            print(File_Path)
            # 判断是否已经存在该目录
            if not os.path.exists(File_Path):
                # 目录不存在，进行创建操作
                os.makedirs(File_Path)  # 使用os.makedirs()方法创建多层目录
                print("目录新建成功：" + File_Path)
            else:
                print("目录已存在！！！")
        except BaseException as msg:
            print("新建目录失败：" + msg)
        # # 读取10折数据
        # train_data_Path = "../data/664_data/data_WS_seed_"+str(seed)+"/train_data/" + str(fold) + "train_data.pkl"
        # test_data_Path = "../data/664_data/data_WS_seed_"+str(seed)+"/test_data/" + str(fold) + "test_data.pkl"
        # with open(train_data_Path,'rb') as f:
        #     train_data = pickle.load(f)
        # with open(test_data_Path,'rb') as f:
        #     test_data = pickle.load(f)
        mask = mask_dic['mask' + str(fold)]
        frequency_masked = raw_frequency * mask
        # train_data = list_txt(os.path.join(train_data_Path, str(fold) + "train_data.txt"))
        # test_data = list_txt(os.path.join(test_data_Path, str(fold) + "test_data.txt"))
        # auc, PR_auc, rmse, mae = train_test(train_data, test_data, fold, args)

        drugAUC, drugAUPR, rmse,mae,pcc,this_fold_metric = train_test(raw_frequency, frequency_masked, mask, fold, args)
        total_metrics.append(this_fold_metric)

        total_auc.append(drugAUC)
        total_pr_auc.append(drugAUPR)
        total_rmse.append(rmse)
        total_mae.append(mae)
        total_pcc.append(pcc)
        print("==================================fold {} end".format(fold))
        fold += 1
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        # 打印运行时间
        print(f"本次程序运行时间：{elapsed_time}")
        # print('Total_AUC:')
        # print(np.mean(total_auc))
        # print('Total_AUPR:')
        # print(np.mean(total_pr_auc))
        # print('Total_RMSE:')
        # print(np.mean(total_rmse))
        # print('Total_MAE:')
        # print(np.mean(total_mae))
        #在Windows下，无论是否使用sys.stdout.flush()，其结果都是多次输出；
        # 在Linux下，可以通过三种方式实现实时将缓冲区的内容输出：
        # 1）sys.stdout.flush()
        # 2）向缓冲区输入换行符，将print()的参数end设置为’\n’（其实默认 的end=’\n’）
        # 3）将print()的参数flush设置为True（默认flush=False）
        # sys.stdout.flush()
        # break
        try:
            File_Path = "./metrics_WS/regress/total/"
            print(File_Path)
            # 判断是否已经存在该目录
            if not os.path.exists(File_Path):
                # 目录不存在，进行创建操作
                os.makedirs(File_Path)  # 使用os.makedirs()方法创建多层目录
                print("目录新建成功：" + File_Path)
            else:
                print("目录已存在！！！")
        except BaseException as msg:
            print("新建目录失败：" + msg)
        with open("./metrics_WS/regress/total/total_auc.txt", 'w') as total_auc_f:
            total_auc_f.write(str(total_auc))
        with open("./metrics_WS/regress/total/total_aupr.txt", 'w') as total_aupr_f:
            total_aupr_f.write(str(total_pr_auc))
        with open("./metrics_WS/regress/total/total_rmse.txt", 'w') as total_rmse_f:
            total_rmse_f.write(str(total_rmse))
        with open("./metrics_WS/regress/total/total_mae.txt", 'w') as total_mae_f:
            total_mae_f.write(str(total_mae))
        with open("./metrics_WS/regress/total/total_pcc.txt", 'w') as total_pcc_f:
            total_pcc_f.write(str(total_pcc))

        auc_avg = sum(total_auc)/len(total_auc)
        aupr_avg = sum(total_pr_auc)/len(total_pr_auc)
        rmse_avg = sum(total_rmse)/len(total_rmse)
        mae_avg = sum(total_mae)/len(total_mae)
        pcc_avg = sum(total_pcc)/len(total_pcc)
        temp = np.array(total_metrics)
        total_mean_metrics = np.sum(temp, axis=0) / temp.shape[0]
        print('The Average Metric on Testset in current %d folds: AUC:%.5f, AUPR:%.5f, RMSE:%.5f, MAE:%.5f, PCC:%.5f' % (fold, auc_avg, aupr_avg, rmse_avg, mae_avg, pcc_avg))
        print("rmse,mae,pcc,spearman,auc_all, aupr_all, drugAUC, drugAUPR, map_value, ndcg, p1, p5, p10, p15, r1, r5, r10, r15:")
        print(total_mean_metrics)
        print("运行结束")

def main():
    # Training settings
    parser = argparse.ArgumentParser(description = 'Model')#description在命令行显示帮助信息的时候会看到description描述的信息
    #Namespace中有两个属性（也叫成员）这里要注意个问题，当'-'和'--'同时出现的时候，系统默认后者为参数名，前者不是，但是在命令行输入的时候没有这个区分
    #对于位置参数操作，dest通常作为add_argument（）的第一个参数提供：
    parser.add_argument('--drug_num', type=int, default=664,  # 药物种类
                        metavar='N', help='drug number')
    parser.add_argument('--side_num', type=int, default=994,  # 副作用种类
                        metavar='N', help='side effect number')
    parser.add_argument('--drug_feat', type=int, default=109,  # 药物原子特征
                        metavar='N', help='drug atom feat dim')
    parser.add_argument('--side_feat', type=int, default=249,  # 副作用特征
                        metavar='N', help='side effect feat dim')
    parser.add_argument('--epochs', type = int, default = 2000,#通过对象的add_argument函数来增加参数，default参数表示我们在运行命令时若没有提供参数，程序会将此值当做参数值
                        metavar = 'N', help = 'number of epochs to train')#训练轮数
    parser.add_argument('--lr', type = float, default = 1e-3,#学习率
                        metavar = 'FLOAT', help = 'learning rate')
    parser.add_argument('--weight_decay', type = float, default = 0.01,#权重衰减
                        metavar = 'FLOAT', help = 'weight decay')
    parser.add_argument('--batch_size', type = int, default = 10,#训练时输入批的大小
                        metavar = 'N', help = 'input batch size for training')
    parser.add_argument('--test_batch_size', type = int, default = 1,#测试时输入的批的大小
                        metavar = 'N', help = 'input batch size for testing')
    parser.add_argument('--emd_dim', type=int, default=64,  # 最终的表示维度
                        metavar='N', help='final embedding dim')
    parser.add_argument('--seed', type=int, default=42,  # 随机种子
                        metavar='N', help='random seed')
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    parser.add_argument('--device', type=str, default=device,
                        help='use cuda')
    #multilayersSim
    parser.add_argument('--sim_layers', type=int, default=3,
                        metavar='N', help='The number of iterations for the similarity matrix')
    parser.add_argument('--use_residual', type=bool, default=True, help='Whether to use residuals')
    parser.add_argument('--use_stack', type=bool, default=False, help='The number of iterations for the similarity matrix')
    parser.add_argument('--use_pca', type=bool, default=True,
                        help='use pca')

    args = parser.parse_args()

    print('-------------------- Hyperparams --------------------')
    print('weight decay: ' + str(args.weight_decay))
    print('learning rate: ' + str(args.lr))
    print('Train batch size: ' + str(args.batch_size))
    print('Test batch size: ' + str(args.test_batch_size))
    tenfold(args,args.seed)

if __name__ == "__main__":
    main()
