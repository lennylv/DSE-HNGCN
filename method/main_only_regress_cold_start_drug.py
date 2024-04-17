import argparse
import math
from math import sqrt
import os
import pickle
import random
import time
from datetime import datetime
from scipy import io as sio
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
from model_only_regress_cold_start_drug import myModel
from torch.utils.tensorboard import SummaryWriter   # SummaryWriter的作用是将数据以特定的格式存储到文件夹

def WKNKN(obj_feat_asso,obj_sim,feat_sim,K,p):
    obj_count=obj_feat_asso.shape[0]
    feat_count=obj_feat_asso.shape[1]
    # 标志obj是new 还是known 如果是known ,则相应位为1
    # 如果是new ,则相应位为0
    flag_obj=np.zeros([obj_count])
    flag_feat=np.zeros([feat_count])
    for i in range(obj_count):
        for j in range(feat_count):
            if(obj_feat_asso[i][j]!=0):
                flag_obj[i]=1
                flag_feat[j]=1
    Y_obj=np.zeros([obj_count,feat_count])
    Y_feat=np.zeros([obj_count,feat_count])
    # Yd矩阵的获取
    for obj in range(obj_count):
        obj_knn=KNearestKnownNeighbors(obj,obj_sim,K,flag_obj)#返回K近邻
        w=np.zeros([K])
        Z_obj=0
        # 获取权重w和归一化因子Zd
        for i in range(K):
            w[i]=math.pow(p,i)*obj_sim[obj,obj_knn[i]]
            Z_obj+=obj_sim[obj,obj_knn[i]]
        for i in range(K):
            Y_obj[obj]=Y_obj[obj]+w[i]*obj_feat_asso[obj_knn[i],:]
        Y_obj[obj,:]=Y_obj[obj,:]/Z_obj

    # Yt矩阵的获取
    for feat in range(feat_count):
        feat_knn=KNearestKnownNeighbors(feat,feat_sim,K,flag_feat)
        w=np.zeros([K])
        Z_feat=0
        for j in range(K):
            w[j]=math.pow(p,j)*feat_sim[feat,feat_knn[j]]
            Z_feat+=feat_sim[feat,feat_knn[j]]
        for j in range(K):
            Y_feat[:,feat]=Y_feat[:,feat]+w[j]*obj_feat_asso[:,feat_knn[j]]
        Y_feat[:,feat]=Y_feat[:,feat]/Z_feat

    Y_knn=Y_obj+Y_feat
    Y_knn=Y_knn/2

    ans=np.maximum(obj_feat_asso,Y_knn)#ans的形状是[obj_count,feat_count]
    return ans
# 返回下标，node结点的K近邻（不包括new obj/new feat）
def KNearestKnownNeighbors(node,matrix,K,flagNodeArray):
    KknownNeighbors=np.array([])
    featureSimilarity=matrix[node].copy()#在相似性矩阵中取出第node行
    featureSimilarity[node]=-100   #排除自身结点,使相似度为-100
    featureSimilarity[flagNodeArray==0]=-100  #排除new obj/new feat,使其相似度为-100
    # 只考虑known node
    KknownNeighbors=featureSimilarity.argsort()[::-1]#按照相似度降序排序
    KknownNeighbors=KknownNeighbors[:K]#返回前K个结点的下标
    return KknownNeighbors

def getSimTopK(sim_info,knn=20):
    if sim_info.shape[0] == sim_info.shape[1]:
        np.fill_diagonal(sim_info, 0)
        # 对于每一行，找到最大的前10个元素，其余都置为0
        for k in range(sim_info.shape[0]):
            sorted_indices = np.argsort(sim_info[k,:])[::-1]  # 对每一行进行降序排序，得到元素的索引
            top_knn_indices = sorted_indices[:knn]  # 取前knn个元素的索引
            other_indices = sorted_indices[knn:]  # 其他元素的索引
            sim_info[k][other_indices] = 0  # 将其他元素置为0
        np.fill_diagonal(sim_info, 1)
    else:
        # 对于每一行，找到最大的前10个元素，其余都置为0
        for k in range(sim_info.shape[0]):
            sorted_indices = np.argsort(sim_info[k,:])[::-1]  # 对每一行进行降序排序，得到元素的索引
            top_knn_indices = sorted_indices[:knn]  # 取前knn个元素的索引
            other_indices = sorted_indices[knn:]  # 其他元素的索引
            sim_info[k][other_indices] = 0  # 将其他元素置为0
    return sim_info

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
    mat1 = np.hstack((obj_sim, obj_feat_asso))
    mat2 = np.hstack((obj_feat_asso.T, feat_sim))
    hidden_init = np.vstack((mat1, mat2))
    feature_matrix = np.copy(hidden_init)
    feature_matrix = torch.tensor(feature_matrix, dtype=torch.float32)
    num_features = obj_sim.shape[0] + feat_sim.shape[0]  # 750+994=1744
    return interactor_matrix,feature_matrix,num_features

def get_data(train_index, test_index, raw_freq):
    with open("../data/664_data/drug/drug_5fp_jaccard_664_664_np.pkl",'rb') as f:
        drug_5fp_jaccard = pickle.load(f)
    with open("../data/664_data/drug/drug_go_jaccard_664_664_np.pkl",'rb') as f:
        drug_go_jaccard = pickle.load(f)
    with open("../data/664_data/drug/drug_disease_jaccard_664_664_np.pkl",'rb') as f:
        drug_disease_jaccard = pickle.load(f)
    with open("../data/664_data/drug/drug_gene_jaccard_664_664_np.pkl",'rb') as f:
        drug_gene_jaccard = pickle.load(f)

    with open("../data/664_data/side/side_effect_semantic_sim.pkl",'rb') as f:
        side_semantic_sim = pickle.load(f)

    # drug_sim_info = [drug_5fp_jaccard,drug_go_jaccard,drug_disease_jaccard,drug_gene_jaccard]
    # side_sim_info = [side_semantic_sim]
    # drug_5fp_jaccard[drug_5fp_jaccard<0.4] = 0
    # drug_go_jaccard[drug_go_jaccard<0.4] = 0
    # drug_5fp_jaccard[drug_5fp_jaccard<0.35] = 0
    # drug_go_jaccard[drug_go_jaccard<0.2] = 0
    # drug_disease_jaccard[drug_disease_jaccard<0.95] = 0
    # drug_gene_jaccard[drug_gene_jaccard<0.95] = 0
    # side_semantic_sim[side_semantic_sim<0.2] = 0
    # drug_sim_info = [getSimTopK(drug_5fp_jaccard),getSimTopK(drug_go_jaccard),getSimTopK(drug_disease_jaccard)]
    # side_sim_info = [getSimTopK(side_semantic_sim)]

    train_drug_sim_5fp = drug_5fp_jaccard[train_index,:][:,train_index]
    train_drug_sim_gofs = drug_go_jaccard[train_index,:][:,train_index]
    train_drug_sim_disease = drug_disease_jaccard[train_index,:][:,train_index]
    train_drug_sim_gene = drug_gene_jaccard[train_index,:][:,train_index]
    test_drug_sim_5fp = drug_5fp_jaccard[test_index,:][:,train_index]
    test_drug_sim_gofs = drug_go_jaccard[test_index,:][:,train_index]
    test_drug_sim_disease = drug_disease_jaccard[test_index,:][:,train_index]
    test_drug_sim_gene = drug_gene_jaccard[test_index,:][:,train_index]

    train_drug_sim_info = [train_drug_sim_5fp, train_drug_sim_gofs, train_drug_sim_disease, train_drug_sim_gene]
    train_side_sim_info = [side_semantic_sim]
    test_drug_sim_info = [test_drug_sim_5fp, test_drug_sim_gofs, test_drug_sim_disease,
                           test_drug_sim_gene]
    for i in range(len(train_drug_sim_info)):
        train_drug_sim_info[i] = getSimTopK(train_drug_sim_info[i])
    for i in range(len(train_side_sim_info)):
        train_side_sim_info[i] = getSimTopK(train_side_sim_info[i])
    for i in range(len(test_drug_sim_info)):
        test_drug_sim_info[i] = getSimTopK(test_drug_sim_info[i],knn=10)
    # raw_asso = np.copy(raw_freq)
    # raw_asso[raw_asso!=0] = 1
    train_freq = raw_freq[train_index,:]
    # test_freq = raw_freq[test_index,:]

    # drug_sim = sum(drug_sim_info) / len(drug_sim_info)
    # side_sim = sum(side_sim_info) / len(side_sim_info) if len(side_sim_info) >1 else side_sim_info[0]
    #
    # asso_masked = WKNKN(asso_masked, drug_sim, side_sim, K=5, p=0.7)

    inm_list_freq = []
    feat_list_freq = []
    for i in range(len(train_drug_sim_info)):
        for j in range(len(train_side_sim_info)):
            interactor_matrix_freq, feature_matrix_freq, num_features_freq = getHeteNetData(train_drug_sim_info[i], train_side_sim_info[j],train_freq, factor=10)
            inm_list_freq.append(interactor_matrix_freq)
            feat_list_freq.append(feature_matrix_freq)


    return [train_drug_sim_info,train_side_sim_info,inm_list_freq,feat_list_freq,test_drug_sim_info]

def train_test(raw_freq,frequency_masked,mask,new_index,fold, args):
    # train_index = np.asarray(np.where(mask[:, 0].flatten() != 0)[0]).tolist()
    # test_index = np.asarray(np.where(mask[:, 0].flatten() == 0)[0]).tolist()
    train_index = np.where((mask != 0).all(axis=1))[0].tolist()
    test_index = np.where((mask == 0).all(axis=1))[0].tolist()
    nonzero_count = np.count_nonzero(raw_freq[train_index,:])
    zero_count = np.count_nonzero(raw_freq[train_index,:] == 0)

    data_list = get_data(train_index,test_index,raw_freq)
    raw_asso = np.copy(raw_freq)
    raw_asso[raw_asso!=0] = 1
    for i in range(len(data_list)):
        if type(data_list[i]) == list:
            for j in range(len(data_list[i])):
                if type(data_list[i][j]) != torch.Tensor:
                    data_list[i][j] = torch.tensor(data_list[i][j], dtype=torch.float32).to(args.device)
                else:
                    data_list[i][j] = data_list[i][j].to(args.device)
        else:
            if type(data_list[i]) != torch.Tensor:
                data_list[i] = torch.tensor(data_list[i], dtype=torch.float32).to(args.device)
            else:
                data_list[i] = data_list[i].to(args.device)
    train_freq = torch.Tensor(raw_freq[train_index]).to(args.device)
    test_freq = torch.Tensor(raw_freq[test_index]).to(args.device)
    train_index = torch.LongTensor(train_index).to(args.device)
    test_index = torch.LongTensor(test_index).to(args.device)
    train_data = data_list[:-1]
    train_data.extend([train_index, train_freq])
    test_data = [data_list[-1], data_list[1], data_list[2], data_list[3], new_index, test_freq]
    # test_data.extend([test_index,test_freq])



    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = myModel(args,drug_num=len(train_index),side_num=args.side_num,drug_feats = len(data_list[0]), side_feats = len(data_list[1]),pos_weight = zero_count/nonzero_count).to(args.device).to(args.device)

    drug_index_train = [i for i in range(len(train_index))]
    drug_index_test = [i for i in range(len(test_index))]
    # drug_index = [i for i in range(args.drug_num)]
    trainset = torch.utils.data.TensorDataset(torch.LongTensor(drug_index_train))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(drug_index_test))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,num_workers=1, pin_memory=True)
    verify_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False,num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,num_workers=1, pin_memory=True)


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loss = []
    train_auc = []
    train_aupr = []
    train_rmse = []
    train_mae = []
    train_pcc = []

    test_auc = []
    test_aupr = []
    test_rmse = []
    test_mae = []
    test_pcc = []
    best_epoch = 0

    folder = "./metrics_"+args.mode+"/regress/fold"+str(fold)+"/runs/train/loss" # 指定存储文件夹
    writer = SummaryWriter(log_dir=folder) # 实例化SummaryWriter类，每30秒写入一次到硬盘


    folder_test = "./metrics_"+args.mode+"/regress/fold"+str(fold)+"/runs/test/loss" # 指定存储文件夹
    writer_test = SummaryWriter(log_dir=folder_test) # 实例化SummaryWriter类，每30秒写入一次到硬盘

    folder_train_metric = "./metrics_"+args.mode+"/regress/fold"+str(fold)+"/runs/train/metric" # 指定存储文件夹
    writer_train_metric = SummaryWriter(log_dir=folder_train_metric) # 实例化SummaryWriter类，每30秒写入一次到硬盘

    folder_test_metric = "./metrics_"+args.mode+"/regress/fold"+str(fold)+"/runs/test/metric" # 指定存储文件夹
    writer_test_metric = SummaryWriter(log_dir=folder_test_metric) # 实例化SummaryWriter类，每30秒写入一次到硬盘



    for epoch in range(0, args.epochs):
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

        if (epoch+1) % 50 == 0:
            # # # 首先，你需要获取优化器的学习率
            # # lr = optimizer.param_groups[0]['lr']
            # # print('Current learning rate:', lr)
            start = time.time()
            # t_i_auc, t_iPR_auc, t_rmse, t_mae, aupr_2 = test(model,_train,drug_side_frequency_fp,drug_side_association_fp,drug_side_frequency_atc,drug_side_association_atc,drug_fp_jaccard, drug_ATC_WHO_750_similarity, side_effect_semantic,interactor_matrix_1,interactor_matrix_2,interactor_matrix_3,features_1,features_2,features_3,args)
            t_i_auc, t_iPR_auc, t_rmse, t_mae, t_pcc = verify(model, verify_loader, train_data, args)
            train_auc.append(t_i_auc)
            train_aupr.append(t_iPR_auc)
            train_rmse.append(t_rmse)
            train_mae.append(t_mae)
            time_cost = time.time() - start
            print("Time: %.2f Epoch: %d <Test on Trainset> RMSE: %.5f, MAE: %.5f, PCC: %.5f, AUC: %.5f, AUPR: %.5f " % (
                time_cost, epoch, t_rmse, t_mae, t_pcc, t_i_auc, t_iPR_auc))
            writer_train_metric.add_scalars('train_metric',
                                            {"train_auc": t_i_auc, "train_aupr": t_iPR_auc, "train_rmse": t_rmse,
                                             "train_mae": t_mae, "train_pcc": t_pcc}, epoch + 1)  # 把loss写入到文件夹中
            start = time.time()
            # i_auc, iPR_auc, rmse, mae, aupr_2 = test(model,_test,drug_side_frequency_fp,drug_side_association_fp,drug_side_frequency_atc,drug_side_association_atc,drug_fp_jaccard, drug_ATC_WHO_750_similarity, side_effect_semantic,interactor_matrix_1,interactor_matrix_2,interactor_matrix_3,features_1,features_2,features_3,args)
            drugAUC, drugAUPR, rmse, mae, pcc = test(model, test_loader, test_data, args)
            test_auc.append(drugAUC)
            test_aupr.append(drugAUPR)
            test_rmse.append(rmse)
            test_mae.append(mae)
            test_pcc.append(pcc)
            time_cost = time.time() - start
            print("Time: %.2f Epoch: %d <Test on Testset> RMSE: %.5f, MAE: %.5f, PCC: %.5f, AUC: %.5f, AUPR: %.5f " % (
                time_cost, epoch, rmse, mae, pcc, drugAUC, drugAUPR))
            writer_test_metric.add_scalars('test_metric',
                                           {"test_auc": drugAUC, "test_aupr": drugAUPR, "test_rmse": rmse,
                                            "test_mae": mae, "test_pcc": pcc}, epoch + 1)  # 把loss写入到文件夹中

            # if (epoch+1) == int(args.epochs*0.25):
            #     optimizer.param_groups[0]['lr']=0.001
            # if (epoch+1) == int(args.epochs*0.75):
            #     optimizer.param_groups[0]['lr']=0.0001
            if (epoch+1) == int(args.epochs*0.5):
                optimizer.param_groups[0]['lr']=0.0001


        # if endure_count > 10:
        #     break

        # break
    with open("./metrics_"+args.mode+"/regress/fold"+str(fold)+"/train_loss.txt", 'w') as train_loss_f:
        train_loss_f.write(str(train_loss))
    #
    with open("./metrics_"+args.mode+"/regress/fold"+str(fold)+"/train_auc.txt", 'w') as train_auc_f:
        train_auc_f.write(str(train_auc))
    with open("./metrics_"+args.mode+"/regress/fold"+str(fold)+"/train_aupr.txt", 'w') as train_aupr_f:
        train_aupr_f.write(str(train_aupr))
    with open("./metrics_"+args.mode+"/regress/fold"+str(fold)+"/train_rmse.txt", 'w') as train_rmse_f:
        train_rmse_f.write(str(train_rmse))
    with open("./metrics_"+args.mode+"/regress/fold"+str(fold)+"/train_mae.txt", 'w') as train_mae_f:
        train_mae_f.write(str(train_mae))

    with open("./metrics_"+args.mode+"/regress/fold"+str(fold)+"/test_auc.txt", 'w') as test_auc_f:
        test_auc_f.write(str(test_auc))
    with open("./metrics_"+args.mode+"/regress/fold"+str(fold)+"/test_aupr.txt", 'w') as test_aupr_f:
        test_aupr_f.write(str(test_aupr))
    with open("./metrics_"+args.mode+"/regress/fold"+str(fold)+"/test_rmse.txt", 'w') as test_rmse_f:
        test_rmse_f.write(str(test_rmse))
    with open("./metrics_"+args.mode+"/regress/fold"+str(fold)+"/test_mae.txt", 'w') as test_mae_f:
        test_mae_f.write(str(test_mae))
    with open("./metrics_"+args.mode+"/regress/fold"+str(fold)+"/test_pcc.txt", 'w') as test_pcc_f:
        test_pcc_f.write(str(test_pcc))

    # state_dict = torch.load("./0model.pt",map_location=torch.device('cpu'))
    # model.load_state_dict(state_dict)

    start = time.time()
    # i_auc, iPR_auc, rmse, mae, aupr_2 = test(model,_test,drug_side_frequency_fp,drug_side_association_fp,drug_side_frequency_atc,drug_side_association_atc,drug_fp_jaccard, drug_ATC_WHO_750_similarity, side_effect_semantic,interactor_matrix_1,interactor_matrix_2,interactor_matrix_3,features_1,features_2,features_3,args)
    drugAUC,drugAUPR, rmse, mae, pcc = test_save(model, test_loader, test_data,fold, args)

    time_cost = time.time() - start
    print("Time: %.2f Epoch: %d <Test on Testset> RMSE: %.5f, MAE: %.5f, PCC: %.5f, AUC: %.5f, AUPR: %.5f " % (
        time_cost, best_epoch, rmse, mae, pcc, drugAUC, drugAUPR))
    # print('The best AUC/AUPR on Trainset: %.5f / %.5f' % (AUC_mn, AUPR_mn))
    # print('The RMSE/MAE on Trainset when best AUC/AUPR: %.5f / %.5f' % (rms_mn, mae_mn))
    # 保存训练好的模型
    model_name = str(fold) + 'model.pt'
    checkpointsFolder = "./metrics_"+args.mode+"/regress/model"
    isCheckpointExist = os.path.exists(checkpointsFolder)
    if not isCheckpointExist:
        os.makedirs(checkpointsFolder)
    torch.save(model.state_dict(), checkpointsFolder + "/" + model_name)

    return drugAUC,drugAUPR, rmse, mae, pcc

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
    # train_data = [train_drug_sim_info,train_side_sim_info,inm_list_asso,feat_list_asso,train_index,train_freq]
    model.train()
    train_total_loss = []
    for i, batch_drug in enumerate(tqdm(train_loader,desc="Train......",unit="batch"), 0):
        batch_drug = batch_drug[0].to(args.device)
        optimizer.zero_grad()
        pred_freq,loss = model(train_data[:4], batch_drug,train_data[-1][batch_drug].flatten())
        loss.backward()
        optimizer.step()
        train_total_loss.append(loss.item())
    return sum(train_total_loss) / len(train_total_loss)

def verify(model, train_loader, train_data, args):
    singleDrug_auc = []
    singleDrug_aupr = []
    total_preds = torch.FloatTensor()
    total_reals = torch.FloatTensor()
    with torch.no_grad():
        model.eval()

        for i, batch_drug in enumerate(tqdm(train_loader,desc="Verify......",unit="batch"), 0):
            batch_drug = batch_drug[0].to(args.device)
            train_ratings = train_data[-1][batch_drug].squeeze().cpu()
            pred_freq,loss = model(train_data[:4], batch_drug,train_data[-1][batch_drug].flatten())
            this_pred_freq = pred_freq.squeeze().cpu()
            total_preds = torch.cat((total_preds, this_pred_freq[torch.where(train_ratings != 0)]), dim=0)
            total_reals = torch.cat((total_reals, train_ratings[torch.where(train_ratings != 0)]), dim=0)

            # train_ratings[train_ratings != 0] = 1
            posi = this_pred_freq[torch.where(train_ratings != 0)]  # 预测结果中，与被掩盖的位置相对应的药副对为正样本
            nege = this_pred_freq[torch.where(train_ratings == 0)]  # 找到mask[index].flatten()和train_label对应位置元素相减不为0，即不同的元素的索引
            y = np.hstack((posi, nege))
            y_true = np.hstack((np.ones(len(posi)), np.zeros(len(nege))))
            singleDrug_auc.append(roc_auc_score(y_true, y))
            singleDrug_aupr.append(average_precision_score(y_true, y))
        drugAUC = sum(singleDrug_auc) / len(singleDrug_auc)
        drugAUPR = sum(singleDrug_aupr) / len(singleDrug_aupr)
        print('num of singleDrug_auc: ', len(singleDrug_auc))
        print("<Test on Trainset> AUC: %.5f, AUPR: %.5f " % (drugAUC, drugAUPR))
        total_reals = total_reals.numpy()
        total_preds = total_preds.numpy()
        rmse = sqrt(((total_reals - total_preds) ** 2).mean())
        mae = mean_absolute_error(total_reals, total_preds)
        pcc = np.corrcoef(total_reals, total_preds)[0, 1]
    return drugAUC,drugAUPR,rmse,mae,pcc


def test(model, test_loader, test_data, args):
    # test_data = [test_drug_sim_info,train_side_sim_info,test_inm_list_asso,test_feat_list_asso,test_index,test_freq]
    total_pred_freq = []

    singleDrug_freq_auc = []
    singleDrug_freq_aupr = []
    total_preds = torch.FloatTensor()
    total_reals = torch.FloatTensor()
    old_drug_num = test_data[0][0].shape[1]
    new_index = test_data[-2]

    with torch.no_grad():
        model.eval()
        for i, batch_drug in enumerate(tqdm(test_loader,desc="Test......",unit="batch"), 0):
            if i not in new_index:
                continue
            batch_drug = batch_drug[0].to(args.device)
            pred_freq = model(test_data[:4], batch_drug,None, training=False)
            temp = pred_freq.tolist()
            # print(len(temp))
            total_pred_freq.append(temp)

            this_pred_freq = pred_freq.squeeze().cpu()
            this_truth_freq = test_data[-1][batch_drug].squeeze().cpu()
            total_preds = torch.cat((total_preds, this_pred_freq[torch.where(this_truth_freq != 0)]), dim=0)
            total_reals = torch.cat((total_reals, this_truth_freq[torch.where(this_truth_freq != 0)]), dim=0)

            posi = this_pred_freq[torch.where(this_truth_freq!=0)]  # 预测结果中，与被掩盖的位置相对应的药副对为正样本
            nege = this_pred_freq[torch.where(this_truth_freq==0)]  # 找到mask[index].flatten()和train_label对应位置元素相减不为0，即不同的元素的索引
            y = np.hstack((posi.data.numpy(), nege.data.numpy()))
            y_true = np.hstack((np.ones(len(posi)), np.zeros(len(nege))))
            singleDrug_freq_auc.append(roc_auc_score(y_true, y))
            singleDrug_freq_aupr.append(average_precision_score(y_true,y))
        print('num of singleDrug: ', len(singleDrug_freq_auc))
        drugAUC_freq = sum(singleDrug_freq_auc) / len(singleDrug_freq_auc)
        drugAUPR_freq = sum(singleDrug_freq_aupr) / len(singleDrug_freq_aupr)
        print('drugAUC_freq:%.5f, drugAUPR_freq:%.5f '%(drugAUC_freq,drugAUPR_freq))
        total_reals = total_reals.numpy()
        total_preds = total_preds.numpy()
        rmse = sqrt(((total_reals - total_preds) ** 2).mean())
        mae = mean_absolute_error(total_reals, total_preds)
        pcc = np.corrcoef(total_reals, total_preds)[0, 1]
    return drugAUC_freq,drugAUPR_freq,rmse,mae,pcc


def test_save(model, test_loader, test_data, fold, args):
    # test_data = [test_drug_sim_info,train_side_sim_info,test_inm_list_asso,test_feat_list_asso,test_index,test_freq]
    total_pred_freq = []

    singleDrug_freq_auc = []
    singleDrug_freq_aupr = []
    total_preds = torch.FloatTensor()
    total_reals = torch.FloatTensor()
    old_drug_num = test_data[0][0].shape[1]
    new_index = test_data[-2]

    with torch.no_grad():
        model.eval()
        for i, batch_drug in enumerate(tqdm(test_loader, desc="Test......", unit="batch"), 0):
            if i not in new_index:
                continue
            batch_drug = batch_drug[0].to(args.device)
            pred_freq = model(test_data[:4], batch_drug, None, training=False)
            temp = pred_freq.tolist()
            # print(len(temp))
            total_pred_freq.append(temp)

            this_pred_freq = pred_freq.squeeze().cpu()
            this_truth_freq = test_data[-1][batch_drug].squeeze().cpu()
            total_preds = torch.cat((total_preds, this_pred_freq[torch.where(this_truth_freq != 0)]), dim=0)
            total_reals = torch.cat((total_reals, this_truth_freq[torch.where(this_truth_freq != 0)]), dim=0)

            posi = this_pred_freq[torch.where(this_truth_freq != 0)]  # 预测结果中，与被掩盖的位置相对应的药副对为正样本
            nege = this_pred_freq[torch.where(this_truth_freq == 0)]  # 找到mask[index].flatten()和train_label对应位置元素相减不为0，即不同的元素的索引
            y = np.hstack((posi.data.numpy(), nege.data.numpy()))
            y_true = np.hstack((np.ones(len(posi)), np.zeros(len(nege))))
            singleDrug_freq_auc.append(roc_auc_score(y_true, y))
            singleDrug_freq_aupr.append(average_precision_score(y_true, y))
        print('num of singleDrug: ', len(singleDrug_freq_auc))
        drugAUC_freq = sum(singleDrug_freq_auc) / len(singleDrug_freq_auc)
        drugAUPR_freq = sum(singleDrug_freq_aupr) / len(singleDrug_freq_aupr)
        print('drugAUC_freq:%.5f, drugAUPR_freq:%.5f ' % (drugAUC_freq, drugAUPR_freq))
        total_reals = total_reals.numpy()
        total_preds = total_preds.numpy()
        rmse = sqrt(((total_reals - total_preds) ** 2).mean())
        mae = mean_absolute_error(total_reals, total_preds)
        pcc = np.corrcoef(total_reals, total_preds)[0, 1]
        with open("./metrics_" + args.mode + "/regress/pred_value/" + str(fold) + "_pred_" + str(
                len(singleDrug_freq_auc)) + "_994_np.pkl", 'wb') as f:
            pickle.dump(total_pred_freq, f)
    return drugAUC_freq, drugAUPR_freq, rmse, mae, pcc

def get_cs_ten_fold_mask(seed=42):
    mask_file_path = "../data/664_data/data_CS/ten_fold_mask_cs_seed"+str(seed)+".mat"
    new_index_file_path = "../data/664_data/data_CS/ten_fold_fliter_index_seed"+str(seed)+".mat"
    if os.path.exists(mask_file_path) and os.path.exists(new_index_file_path):
        ten_fold_mask = sio.loadmat(mask_file_path)
        ten_fold_new_index = sio.loadmat(new_index_file_path)
        return ten_fold_mask,ten_fold_new_index
    else:
        mask_file_dir_path = mask_file_path[:mask_file_path.rfind('/')]
        if os.path.exists(mask_file_dir_path):
            pass
        else:
            os.makedirs(mask_file_dir_path)  # 使用os.makedirs()方法创建多层目录
        with open("../data/664_data/frequency_664_994_np.pkl", 'rb') as f:
            raw_freq = pickle.load(f)
        index = np.arange(0, len(raw_freq), 1)
        np.random.seed(seed)
        np.random.shuffle(index)
        x = []
        n = int(np.ceil(len(index) / 10))
        for i in range(10):
            if i == 9:
                x.append(index.tolist())
                break
            x.append(index[0:n].tolist())
            index = index[n:]

        dic_mask = {}
        for i in range(10):
            mask = np.ones(raw_freq.shape)
            mask[x[i]] = 0
            dic_mask['mask' + str(i)] = mask
        sio.savemat(mask_file_path, dic_mask)
        drug_sim_data_NRFSE_file = sio.loadmat("../data/664_data/NRFSE/DrugSimMat1.mat")
        drug_sim_data_NRFSE = drug_sim_data_NRFSE_file['DrugSimMat1']
        cs_ten_fold_mask = sio.loadmat(mask_file_path)
        ten_fold_fliter_index = []
        dict_new_index = {}
        for i in range(10):
            #找到每一折的new drug索引
            old_index = []
            new_index = []
            mask_this = cs_ten_fold_mask['mask'+str(i)]
            for j in range(664):
                if np.sum(mask_this[j,:]) == 0:
                    new_index.append(j)
                else:
                    old_index.append(j)
            #过滤相似性高于0.6的new drug
            old_sim = drug_sim_data_NRFSE[old_index,:][:,old_index]
            new_sim = drug_sim_data_NRFSE[new_index,:][:,old_index]
            new_index_fliter = []
            max_values_per_row = np.amax(new_sim, axis=1)
            for k in range(len(new_index)):
                if max_values_per_row[k] < 0.6:
                    # new_index_fliter.append(new_index[k])#保存在664中新药的索引
                    new_index_fliter.append(k)#保存在新药中的相对索引
            ten_fold_fliter_index.append(new_index_fliter)
            dict_new_index["new_index"+str(i)] = np.array(new_index_fliter)
        sio.savemat(new_index_file_path,dict_new_index)
        print("mask文件新建成功，seed="+str(seed))
        return dic_mask,dict_new_index


def tenfold(args,seed=42):
    with open("../data/664_data/frequency_664_994_np.pkl", 'rb') as f:
        raw_frequency = pickle.load(f)
    dic_mask, dict_new_index = get_cs_ten_fold_mask(seed=seed)
    try:
        File_Path = "./metrics_"+args.mode+"/regress/pred_value/"
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
            File_Path = "./metrics_"+args.mode+"/regress/fold"+str(fold)+"/during_train/"
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
        mask = dic_mask['mask' + str(fold)]
        frequency_masked = raw_frequency * mask
        new_index = dict_new_index['new_index' + str(fold)].tolist()[0]
        print(new_index)
        drugAUC, drugAUPR,rmse,mae,pcc = train_test(raw_frequency, frequency_masked, mask,new_index, fold, args)

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
        # break
        try:
            File_Path = "./metrics_"+args.mode+"/regress/total"
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
        with open("./metrics_"+args.mode+"/regress/total/total_auc.txt", 'w') as total_auc_f:
            total_auc_f.write(str(total_auc))
        with open("./metrics_"+args.mode+"/regress/total/total_aupr.txt", 'w') as total_aupr_f:
            total_aupr_f.write(str(total_pr_auc))
        with open("./metrics_"+args.mode+"/regress/total/total_rmse.txt", 'w') as total_rmse_f:
            total_rmse_f.write(str(total_rmse))
        with open("./metrics_"+args.mode+"/regress/total/total_mae.txt", 'w') as total_mae_f:
            total_mae_f.write(str(total_mae))
        with open("./metrics_"+args.mode+"/regress/total/total_pcc.txt", 'w') as total_pcc_f:
            total_pcc_f.write(str(total_pcc))

        auc_avg = sum(total_auc)/len(total_auc)
        aupr_avg = sum(total_pr_auc)/len(total_pr_auc)
        rmse_avg = sum(total_rmse)/len(total_rmse)
        mae_avg = sum(total_mae)/len(total_mae)
        pcc_avg = sum(total_pcc)/len(total_pcc)
        # temp = np.array(total_metrics)
        # total_mean_metrics = np.sum(temp, axis=0) / temp.shape[0]
        print('The Average Metric on Testset in current %d folds: AUC:%.5f, AUPR:%.5f, RMSE:%.5f, MAE:%.5f, PCC:%.5f' % (fold, auc_avg, aupr_avg, rmse_avg, mae_avg, pcc_avg))
        # print("rmse,mae,pcc,spearman,auc_all, aupr_all, drugAUC, drugAUPR, map_value, ndcg, p1, p5, p10, p15, r1, r5, r10, r15:")
        # print(total_mean_metrics)
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
    parser.add_argument('--epochs', type = int, default = 1500,#通过对象的add_argument函数来增加参数，default参数表示我们在运行命令时若没有提供参数，程序会将此值当做参数值
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
    parser.add_argument('--seed', type=int, default=3,  # 随机种子
                        metavar='N', help='random seed')
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    parser.add_argument('--device', type=str, default=device,
                        help='use cuda')
    parser.add_argument('--mode', type=str, default="CS",
                        help='WS or CS')

    args = parser.parse_args()

    print('-------------------- Hyperparams --------------------')
    print('weight decay: ' + str(args.weight_decay))
    print('learning rate: ' + str(args.lr))
    print('Train batch size: ' + str(args.batch_size))
    print('Test batch size: ' + str(args.test_batch_size))
    tenfold(args,args.seed)

if __name__ == "__main__":
    main()
