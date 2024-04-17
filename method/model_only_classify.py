import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
import torch.nn.functional as F
class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        # print(x.shape)  # print(x.shape)
        # return x
        pass
class hetenet(nn.Module):
    def __init__(self, obj_num,feat_num,out_dim):
        super(hetenet, self).__init__()
        self.obj_num = obj_num
        self.feat_num = feat_num
        self.input_dim = self.obj_num + self.feat_num
        # self.input_dim = 256
        self.out_dim = out_dim



        self.weight_final = nn.Parameter(torch.Tensor([0.5, 0.33, 0.25]))
        self.dropout_interactor_matrix = 0.4
        self.dropout_feat = 0.6
        self.dropout_layer = nn.Dropout(self.dropout_interactor_matrix)
        # gcn_1
        self.dropout_11 = nn.Dropout(self.dropout_feat)
        self.weight_11 = nn.Linear(self.input_dim, self.out_dim)
        # self.weight_11 = nn.Sequential(
        #     nn.Linear(self.input_dim, self.out_dim*4),
        #     nn.BatchNorm1d(self.out_dim*4),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(self.out_dim*4, self.out_dim),
        #     nn.BatchNorm1d(self.out_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(self.out_dim, self.out_dim),
        # )
        self.act_11 = nn.ReLU()
        self.bn_11 = nn.BatchNorm1d(num_features=self.out_dim)
        self.ln_11 = nn.LayerNorm(self.out_dim)
        # gcn_2
        self.dropout_12 = nn.Dropout(self.dropout_feat)
        self.weight_12 = nn.Linear(self.out_dim, self.out_dim)
        # self.weight_12 = nn.Sequential(
        #     nn.Linear(self.out_dim, self.out_dim),
        #     nn.BatchNorm1d(self.out_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(self.out_dim, self.out_dim)
        # )
        self.act_12 = nn.ReLU()
        self.bn_12 = nn.BatchNorm1d(num_features=self.out_dim)
        self.ln_12 = nn.LayerNorm(self.out_dim)
        # gcn_3
        self.dropout_13 = nn.Dropout(self.dropout_feat)
        self.weight_13 = nn.Linear(self.out_dim, self.out_dim)
        # self.weight_13 = nn.Sequential(
        #     nn.Linear(self.out_dim, self.out_dim),
        #     nn.BatchNorm1d(self.out_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(self.out_dim, self.out_dim)
        # )
        self.act_13 = nn.ReLU()
        self.bn_13 = nn.BatchNorm1d(num_features=self.out_dim)
        self.ln_13 = nn.LayerNorm(self.out_dim)

        self.scale = self.out_dim ** -0.5  # 公式里的根号dk
        self.q_initial = nn.Linear(self.out_dim,self.out_dim)
        self.k_initial = nn.Linear(self.out_dim,self.out_dim)
        self.v_initial = nn.Linear(self.out_dim,self.out_dim)

        self.q_self = nn.Linear(self.out_dim, self.out_dim)
        self.k_self = nn.Linear(self.out_dim, self.out_dim)
        self.v_self = nn.Linear(self.out_dim, self.out_dim)

        self.lin = nn.Sequential(
            nn.Linear(self.out_dim*3, self.out_dim),
            # nn.LayerNorm(self.out_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.out_dim, self.out_dim,bias=True),
        )

    def forward(self,interactor_matrix,features,obj_index, training=True):
        interactor_matrix = self.dropout_layer(interactor_matrix)
        #feat = self.initial_layer(features)
        x = self.weight_11(features)
        # x = self.bn_11(x)
        x = self.ln_11(x)
        x = self.act_11(x)
        x = self.dropout_11(x)
        hidden11 = torch.mm(interactor_matrix, x)

        x = self.weight_12(hidden11)
        # x = self.bn_12(x)
        x = self.ln_12(x)
        x = self.act_12(x)
        x = self.dropout_12(x)
        hidden12 = torch.mm(interactor_matrix, x)

        #hidden12 = hidden12 + feat

        x = self.weight_13(hidden12)
        # x = self.bn_13(x)
        x = self.ln_13(x)
        x = self.act_13(x)
        x = self.dropout_13(x)
        hidden13 = torch.mm(interactor_matrix, x)

        # info = 0.5*hidden11 + 0.33*hidden12 + 0.25*hidden13
        info = self.weight_final[0]*hidden11 + self.weight_final[1]*hidden12 + self.weight_final[2]*hidden13

        obj_info = info[:self.obj_num,:]
        feat_info = info[self.obj_num:,:]

        return obj_info[obj_index,:], feat_info

# class SimAttnBlock(nn.Module):
#     '''
#     网络模型
#     '''
#     def __init__(self, obj_num,feat_num,out_dim):
#         super(SimAttnBlock, self).__init__()
#     def forward(self,interactor_matrix,features,obj_index, training=True):
class myModel(nn.Module):
    '''
    网络模型
    '''

    def __init__(self, args, drug_num, side_num,drug_feats,side_feats):
        super(myModel, self).__init__()
        self.device = args.device
        # final embedding dimension
        self.drug_num = drug_num
        self.side_num = side_num
        self.emd_dim = args.emd_dim
        hete_layers = drug_feats * side_feats
        # self.hete_blocks = []
        self.hete_blocks_asso = []
        # self.hete_blocks_sim = []
        for i in range(hete_layers):
            # block = hetenet(self.drug_num, self.side_num, self.emd_dim)
            # self.add_module(f"hete_block{i}", block)
            # self.hete_blocks.append(block)
            block = hetenet(self.drug_num, self.side_num, self.emd_dim)
            self.add_module(f"hete_block_asso{i}", block)
            self.hete_blocks_asso.append(block)
            # block = hetenet(self.drug_num, self.side_num, self.emd_dim)
            # self.add_module(f"hete_block_sim{i}", block)
            # self.hete_blocks_sim.append(block)

        # self.drug_fusion_mlp = nn.Sequential(
        #     nn.Linear(self.emd_dim * hete_layers, self.emd_dim),
        #     #nn.BatchNorm1d(self.emd_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(self.emd_dim, self.emd_dim),
        #     #nn.BatchNorm1d(self.emd_dim),
        #     # nn.ReLU(),
        #     # nn.Dropout(0.5),
        # )
        # self.side_fusion_mlp = nn.Sequential(
        #     nn.Linear(self.emd_dim * hete_layers, self.emd_dim),
        #     # nn.BatchNorm1d(self.emd_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(self.emd_dim, self.emd_dim),
        #     # nn.BatchNorm1d(self.emd_dim),
        #     # nn.ReLU(),
        # )


        self.drug_fusion_mlp_asso = nn.Sequential(
            nn.Linear(self.emd_dim * hete_layers, self.emd_dim),
            # nn.BatchNorm1d(self.emd_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.emd_dim, self.emd_dim),
            # nn.BatchNorm1d(self.emd_dim),
            # nn.ReLU(),
        )
        self.side_fusion_mlp_asso = nn.Sequential(
            nn.Linear(self.emd_dim * hete_layers, self.emd_dim),
            # nn.BatchNorm1d(self.emd_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.emd_dim, self.emd_dim),
            # nn.BatchNorm1d(self.emd_dim),
            # nn.ReLU(),
        )

        self.posi_weight = torch.tensor([20]).to(self.device)
        self.weight_scores_fusion = nn.Parameter(torch.tensor([0.5]))
        self.lamda_border = torch.tensor([0.01],requires_grad=True).to(self.device)
        self.lamda = nn.Parameter(torch.Tensor([-1e10]))
        # self.eps = nn.Parameter(torch.Tensor([0.2]))
        self.eps = 0.2

    def forward(self, data_list, drug_index,true_label,training=True):
        drug_sim_info,side_sim_info,inm_list_asso,feat_list_asso,origin_info_list = data_list

        # drug_hete_emd = []
        # side_hete_emd = []
        # scores_list = []
        drug_hete_emd_asso = []
        side_hete_emd_asso = []
        scores_list_asso = []

        for i in range(len(inm_list_asso)):
            # drug_emd, side_emd = self.hete_blocks[i](inm_list[i],feat_list[i],drug_index)
            # drug_hete_emd.append(drug_emd)
            # side_hete_emd.append(side_emd)
            # scores_list.append(torch.matmul(drug_emd,side_emd.T))
            drug_emd_asso, side_emd_asso = self.hete_blocks_asso[i](inm_list_asso[i],feat_list_asso[i],drug_index)
            drug_hete_emd_asso.append(drug_emd_asso)
            side_hete_emd_asso.append(side_emd_asso)
            scores_list_asso.append(torch.matmul(drug_emd_asso,side_emd_asso.T))

        # scores_avg = sum(scores_list)/len(scores_list)
        # drug_total_emd = self.drug_fusion_mlp(torch.cat((drug_hete_emd),dim=1))
        # side_total_emd = self.side_fusion_mlp(torch.cat((side_hete_emd),dim=1))

        scores_avg_asso = sum(scores_list_asso)/len(scores_list_asso)
        drug_total_emd_asso = self.drug_fusion_mlp_asso(torch.cat((drug_hete_emd_asso),dim=1))
        side_total_emd_asso = self.side_fusion_mlp_asso(torch.cat((side_hete_emd_asso),dim=1))

        # scores_cat = torch.matmul(drug_total_emd, side_total_emd.T)
        scores_cat_asso = torch.matmul(drug_total_emd_asso, side_total_emd_asso.T)
        # scores = (scores_avg + scores_cat) / 2
        scores_asso = (scores_avg_asso + scores_cat_asso) / 2
        if training:
            # scores = scores.flatten()
            scores_asso = scores_asso.flatten()
            x0 = torch.where(true_label == 0)
            x1 = torch.where(true_label != 0)
            true_label[true_label!=0] = 1
            # loss_regress = torch.sum((scores[x1] - true_label[x1]) ** 2)
            # loss_classify = torch.sum((scores_asso[x1] - 1) ** 2) + 0.05 * torch.sum((scores_asso[x0] - 0) ** 2)
            loss_classify = F.binary_cross_entropy_with_logits(scores_asso,true_label,pos_weight=self.posi_weight)
            # loss = loss_regress + loss_classify
            return scores_asso, loss_classify
        else:
            return scores_asso