import math
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from scipy import stats
import torch

def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)
def laplacian(kernel):
    d1 = sum(kernel)
    D_1 = torch.diag(d1)
    L_D_1 = D_1 - kernel
    D_5 = D_1.rsqrt()
    D_5 = torch.where(torch.isinf(D_5), torch.full_like(D_5, 0), D_5)
    L_D_11 = torch.mm(D_5, L_D_1)
    L_D_11 = torch.mm(L_D_11, D_5)
    return L_D_11


def normalized_embedding(embeddings):
    [row, col] = embeddings.size()
    ne = torch.zeros([row, col])
    for i in range(row):
        ne[i, :] = (embeddings[i, :] - min(embeddings[i, :])) / (max(embeddings[i, :]) - min(embeddings[i, :]))
    return ne


def getGipKernel(y, trans, gamma, normalized=False):
    if trans:
        y = y.T
    if normalized:
        y = normalized_embedding(y)
    krnl = torch.mm(y, y.T)
    krnl = krnl / torch.mean(torch.diag(krnl))
    krnl = torch.exp(-kernelToDistance(krnl) * gamma)
    return krnl


def kernelToDistance(k):
    di = torch.diag(k).T
    d = di.repeat(len(k)).reshape(len(k), len(k)).T + di.repeat(len(k)).reshape(len(k), len(k)) - 2 * k
    return d


def cosine_kernel(tensor_1, tensor_2):
    return torch.FloatTensor([torch.cosine_similarity(tensor_1[i], tensor_2, dim=-1).tolist() for i in
                           range(tensor_1.shape[0])])


def normalized_kernel(K):
    K = abs(K)
    k = K.flatten().sort()[0]
    min_v = k[torch.nonzero(k, as_tuple=False)[0]]
    K[torch.where(K == 0)] = min_v
    D = torch.diag(K)
    D = D.sqrt()
    S = K / (D * D.T)
    return S

def rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean())
    return rmse


def mse(y, f):
    mse = ((y - f) ** 2).mean()
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def MAE(y, f):
    rs = sklearn.metrics.mean_absolute_error(y, f)
    return rs


def ci(y, f):
    ind = np.argsort(y)  # argsort函数返回的是数组值从小到大的索引值
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


def draw_loss(train_losses, test_losses, title, result_folder):
    plt.figure()
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.ylim((0, 10))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # save image
    plt.savefig(result_folder + '/' + title + ".png")  # should before show method


def draw_pearson(pearsons, title, result_folder):
    plt.figure()
    plt.plot(pearsons, label='test pearson')
    plt.ylim((-0.1, 1))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Pearson')
    plt.legend()
    # save image
    plt.savefig(result_folder + '/' + title + ".png")  # should before show method


def my_draw_loss(train_losses, title, result_folder):
    plt.figure()
    plt.plot(train_losses, label='train loss')
    plt.ylim((0, 10))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # save image
    plt.savefig(result_folder + '/' + title + ".png")  # should before show method


def my_draw_pearson(pearsons, title, result_folder):
    plt.figure()
    plt.plot(pearsons, label='test pearson')
    plt.ylim((-0.1, 1))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Pearson')
    plt.legend()
    # save image
    plt.savefig(result_folder + '/' + title + ".png")  # should before show method


def my_draw_mse(mse, rmse, title, result_folder):
    plt.figure()
    plt.plot(mse, label='test MSE')
    plt.plot(rmse, label='test rMSE')
    plt.ylim((0, 10))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    # save image
    plt.savefig(result_folder + '/' + title + ".png")  # should before show method


def evaluate_others(M, Tr_neg, Te, positions=[1, 5, 10, 15]):
    """
    :param M: 预测值
    :param Tr_neg: dict， 包含Te
    :param Te:  dict
    :param positions:
    :return:
    """
    prec = np.zeros(len(positions))
    rec = np.zeros(len(positions))
    map_value, auc_value, ndcg = 0.0, 0.0, 0.0
    for u in Te:#遍历测试集的每种药物
        val = M[u, :]#取出每一种药物的所有预测值（994个预测值）
        inx = np.array(Tr_neg[u])#取出训练集负样本中该药物对应的所有副作用id（包括被掩盖为0的测试集正样本）
        A = set(Te[u])#取出测试集正样本中该药物对应的所有副作用id
        B = set(inx) - A#得到测试集负样本中该药物对应的所有副作用id
        # compute precision and recall
        ii = np.argsort(val[inx])[::-1][:max(positions)]#先取出预测值中对应测试集的所有样本的预测值，排序，取最大的前15个预测值的索引
        prec += precision(Te[u], inx[ii], positions)
        rec += recall(Te[u], inx[ii], positions)
        ndcg_user = nDCG(Te[u], inx[ii], 10)
        # compute map and AUC
        pos_inx = np.array(list(A))
        neg_inx = np.array(list(B))
        map_user, auc_user = map_auc(pos_inx, neg_inx, val)
        ndcg += ndcg_user
        map_value += map_user
        auc_value += auc_user
        # outf.write(" ".join([str(map_user), str(auc_user), str(ndcg_user)])+"\n")
    # outf.close()
    return map_value / len(Te.keys()), auc_value / len(Te.keys()), ndcg / len(Te.keys()), prec / len(
        Te.keys()), rec / len(Te.keys())


def precision(actual, predicted, N):#actual为测试集正样本的索引，predicted为预测值最大的前15个的索引
    if isinstance(N, int):
        inter_set = set(actual) & set(predicted[:N])
        return float(len(inter_set))/float(N)
    elif isinstance(N, list):
        return np.array([precision(actual, predicted, n) for n in N])


def recall(actual, predicted, N):
    if isinstance(N, int):
        inter_set = set(actual) & set(predicted[:N])
        return float(len(inter_set))/float(len(set(actual)))
    elif isinstance(N, list):
        return np.array([recall(actual, predicted, n) for n in N])


def nDCG(Tr, topK, num=None):
    if num is None:
        num = len(topK)
    dcg, vec = 0, []
    for i in range(num):
        if topK[i] in Tr:
            dcg += 1/math.log(i+2, 2)
            vec.append(1)
        else:
            vec.append(0)
    vec.sort(reverse=True)
    idcg = sum([vec[i]/math.log(i+2, 2) for i in range(num)])
    if idcg > 0:
        return dcg/idcg
    else:
        return idcg


def map_auc(pos_inx, neg_inx, val):
    map = 0.0
    pos_val, neg_val = val[pos_inx], val[neg_inx]
    ii = np.argsort(pos_val)[::-1]
    jj = np.argsort(neg_val)[::-1]
    pos_sort, neg_sort = pos_val[ii], neg_val[jj]
    auc_num = 0.0
    for i,pos in enumerate(pos_sort):
        num = 0.0
        for neg in neg_sort:
            if pos<=neg:
                num+=1
            else:
                auc_num+=1
        map += (i+1)/(i+num+1)
    return map/len(pos_inx), auc_num/(len(pos_inx)*len(neg_inx))