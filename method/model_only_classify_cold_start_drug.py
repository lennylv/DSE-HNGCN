import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GAT


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        # print(x.shape)  # print(x.shape)
        # return x
        pass

class multihead_gat(nn.Module):
    def __init__(self, input_dim,output_dim,dropout,alpha=-0.2,concat=True):
        super(multihead_gat, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(input_dim,output_dim)))
        nn.init.xavier_uniform_(self.W.data,gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*output_dim,1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU()
        # self.leakyrelu = nn.LeakyReLU(self.alpha)
    def forward(self,input_h,adj):
        h = torch.mm(input_h,self.W)
        N = h.size()[0]
        input_cat = torch.cat([h.repeat(1,N).view(N*N,-1),h.repeat(N,1)],dim=1).view(N,-1,2*self.output_dim)
        e = self.leakyrelu(torch.matmul(input_cat,self.a).squeeze(2))
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj>0,e,zero_vec)
        attention = F.softmax(attention,dim=1)
        attention = F.dropout(attention,self.dropout,training=self.training)
        output_h = torch.matmul(attention,h)
        return output_h

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
        self.act_12 = nn.ReLU()
        self.bn_12 = nn.BatchNorm1d(num_features=self.out_dim)
        self.ln_12 = nn.LayerNorm(self.out_dim)
        # gcn_3
        self.dropout_13 = nn.Dropout(self.dropout_feat)
        self.weight_13 = nn.Linear(self.out_dim, self.out_dim)
        self.act_13 = nn.ReLU()
        self.bn_13 = nn.BatchNorm1d(num_features=self.out_dim)
        self.ln_13 = nn.LayerNorm(self.out_dim)

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
        #info = self.weight_final[0]*hidden11 + self.weight_final[1]*hidden12
        #info = hidden11
        obj_info = info[:self.obj_num, :]
        feat_info = info[self.obj_num:, :]
        # return obj_info,feat_info
        if training:
            return obj_info[obj_index,:], feat_info
        else:
            drug_info = torch.matmul(obj_index,obj_info)/torch.sum(obj_index)
            return drug_info.view([1, -1]), feat_info
        # if training:
        #     obj_info = info[:self.obj_num, :]
        #     feat_info = info[self.obj_num:, :]
        #     return obj_info[obj_index,:], feat_info
        # else:
        #     obj_info = info[:self.obj_num + 1, :]
        #     feat_info = info[self.obj_num + 1:, :]
        #     return obj_info[obj_index, :].view([1, -1]), feat_info



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

    def __init__(self, args, drug_num, side_num,drug_feats,side_feats,pos_weight):
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
            #nn.BatchNorm1d(self.emd_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.emd_dim, self.emd_dim),
            # nn.BatchNorm1d(self.emd_dim),
            # nn.ReLU(),
        )
        self.side_fusion_mlp_asso = nn.Sequential(
            nn.Linear(self.emd_dim * hete_layers, self.emd_dim),
            #nn.BatchNorm1d(self.emd_dim),
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

        # self.drug_gat = multihead_gat(self.emd_dim,self.emd_dim,dropout=0.5)
        # self.side_gat = multihead_gat(self.emd_dim,self.emd_dim,dropout=0.5)


    def forward(self, data_list, drug_index,true_label,training=True):
        if training:
            drug_sim_info, side_sim_info, inm_list_asso, feat_list_asso, _, _ = data_list
            # drug_sim_info = drug_sim_info[0:1] + drug_sim_info[2:]
            # inm_list_asso = inm_list_asso[0:1] + inm_list_asso[2:]
            # feat_list_asso = feat_list_asso[0:1] + feat_list_asso[2:]
            drug_hete_emd_asso = []
            side_hete_emd_asso = []
            scores_list_asso = []
            for i in range(len(inm_list_asso)):
                # if i ==1:
                #     continue
                drug_emd_asso, side_emd_asso = self.hete_blocks_asso[i](inm_list_asso[i], feat_list_asso[i], drug_index,
                                                                        training=training)
                drug_hete_emd_asso.append(drug_emd_asso)
                side_hete_emd_asso.append(side_emd_asso)
                scores_list_asso.append(torch.matmul(drug_emd_asso, side_emd_asso.T))
            scores_avg_asso = sum(scores_list_asso) / len(scores_list_asso)
            drug_total_emd_asso = self.drug_fusion_mlp_asso(torch.cat((drug_hete_emd_asso), dim=1))
            side_total_emd_asso = self.side_fusion_mlp_asso(torch.cat((side_hete_emd_asso), dim=1))

            scores_cat_asso = torch.matmul(drug_total_emd_asso, side_total_emd_asso.T)
            scores_asso = (scores_avg_asso + scores_cat_asso) / 2

            scores_asso = scores_asso.flatten()
            loss_classify = F.binary_cross_entropy_with_logits(scores_asso, true_label, pos_weight=self.posi_weight)
            return F.sigmoid(scores_asso), loss_classify
        else:
            drug_sim_info, side_sim_info, inm_list_asso, feat_list_asso, _, _, new_side_index = data_list
            # drug_sim_info = drug_sim_info[0:1] + drug_sim_info[2:]
            # inm_list_asso = inm_list_asso[0:1] + inm_list_asso[2:]
            # feat_list_asso = feat_list_asso[0:1] + feat_list_asso[2:]
            drug_hete_emd_asso = []
            side_hete_emd_asso = []
            scores_list_asso = []
            for i in range(len(inm_list_asso)):
                # if i ==1:
                #     continue
                if torch.sum(drug_sim_info[i // len(side_sim_info)][drug_index]) == 0:
                    continue
                drug_emd_asso, side_emd_asso = self.hete_blocks_asso[i](inm_list_asso[i], feat_list_asso[i],
                                                                        drug_sim_info[i // len(side_sim_info)][
                                                                            drug_index], training=training)
                side_emd_asso[new_side_index] = torch.matmul(side_sim_info[0][new_side_index],side_emd_asso)/torch.sum(side_sim_info[0][new_side_index])
                drug_hete_emd_asso.append(drug_emd_asso)
                side_hete_emd_asso.append(side_emd_asso)
                scores_list_asso.append(torch.matmul(drug_emd_asso, side_emd_asso.T))
            scores_avg_asso = sum(scores_list_asso) / len(scores_list_asso)
            if len(drug_hete_emd_asso) == len(self.hete_blocks_asso):
                drug_total_emd_asso = self.drug_fusion_mlp_asso(torch.cat((drug_hete_emd_asso), dim=1))
                side_total_emd_asso = self.side_fusion_mlp_asso(torch.cat((side_hete_emd_asso), dim=1))

                scores_cat_asso = torch.matmul(drug_total_emd_asso, side_total_emd_asso.T)
                scores_asso = (scores_avg_asso + scores_cat_asso) / 2

                scores_asso = scores_asso.flatten()
            else:
                scores_asso = scores_avg_asso.flatten()
            # loss_classify = F.binary_cross_entropy_with_logits(scores_asso,true_label,pos_weight=self.posi_weight)
            return F.sigmoid(scores_asso)
        # drug_sim_info, side_sim_info, inm_list_asso, feat_list_asso,drug_autocorrelation,side_autocorrelation = data_list
        # # drug_sim_info = drug_sim_info[0:1] + drug_sim_info[2:]
        # # inm_list_asso = inm_list_asso[0:1] + inm_list_asso[2:]
        # # feat_list_asso = feat_list_asso[0:1] + feat_list_asso[2:]
        # drug_hete_emd_asso = []
        # side_hete_emd_asso = []
        # scores_list_asso = []
        # if training:
        #     for i in range(len(inm_list_asso)):
        #         # if i ==1:
        #         #     continue
        #         drug_emd_asso, side_emd_asso = self.hete_blocks_asso[i](inm_list_asso[i], feat_list_asso[i], drug_index,
        #                                                                 training=training)
        #         drug_hete_emd_asso.append(drug_emd_asso)
        #         side_hete_emd_asso.append(side_emd_asso)
        #         scores_list_asso.append(torch.matmul(drug_emd_asso[drug_index,:], side_emd_asso.T))
        #     scores_avg_asso = sum(scores_list_asso) / len(scores_list_asso)
        #     drug_total_emd_asso = self.drug_fusion_mlp_asso(torch.cat((drug_hete_emd_asso), dim=1))
        #     side_total_emd_asso = self.side_fusion_mlp_asso(torch.cat((side_hete_emd_asso), dim=1))
        #     drug_final_emd = self.drug_gat(drug_total_emd_asso,drug_autocorrelation)
        #     side_final_emd = self.side_gat(side_total_emd_asso,side_autocorrelation)
        #
        #
        #
        #     scores_cat_asso = torch.matmul(drug_final_emd[drug_index,:], side_final_emd.T)
        #     scores_asso = (scores_avg_asso + scores_cat_asso) / 2
        #
        #
        #     scores_asso = scores_asso.flatten()
        #     loss_classify = F.binary_cross_entropy_with_logits(scores_asso, true_label, pos_weight=self.posi_weight)
        #     return torch.sigmoid(scores_asso), loss_classify
        # else:
        #     test_drug_sim_info = []
        #     for i in range(len(inm_list_asso)):
        #         # if i ==1:
        #         #     continue
        #         if torch.sum(drug_sim_info[i//len(side_sim_info)][drug_index]) == 0:
        #             continue
        #         drug_emd_asso, side_emd_asso = self.hete_blocks_asso[i](inm_list_asso[i], feat_list_asso[i],
        #                                                                 drug_sim_info[i//len(side_sim_info)][drug_index], training=training)
        #         test_drug_emd_asso = torch.matmul(drug_sim_info[i//len(side_sim_info)][drug_index],drug_emd_asso)/torch.sum(drug_sim_info[i//len(side_sim_info)][drug_index])
        #         test_drug_sim_info.append(drug_sim_info[i//len(side_sim_info)][drug_index])
        # #     return drug_info.view([1, -1]), feat_info
        #         test_drug_hete_emd = torch.matmul(test_drug_sim_info[-1], drug_emd_asso)
        #         drug_hete_emd_asso.append(drug_emd_asso)
        #         side_hete_emd_asso.append(side_emd_asso)
        #         scores_list_asso.append(torch.matmul(test_drug_hete_emd, side_emd_asso.T))
        #     scores_avg_asso = sum(scores_list_asso) / len(scores_list_asso)
        #     drug_total_emd_asso = self.drug_fusion_mlp_asso(torch.cat((drug_hete_emd_asso), dim=1))
        #     side_total_emd_asso = self.side_fusion_mlp_asso(torch.cat((side_hete_emd_asso), dim=1))
        #     drug_final_emd = self.drug_gat(drug_total_emd_asso, drug_autocorrelation)
        #     side_final_emd = self.side_gat(side_total_emd_asso, side_autocorrelation)
        #     test_drug_final_emd = torch.matmul(sum(test_drug_sim_info) / len(test_drug_sim_info), drug_final_emd)
        #
        #     scores_cat_asso = torch.matmul(test_drug_final_emd, side_final_emd.T)
        #     scores_asso = (scores_avg_asso + scores_cat_asso) / 2
        #
        #     scores_asso = scores_asso.flatten()
        #     # if len(drug_hete_emd_asso) == len(self.hete_blocks_asso):
        #     #     drug_total_emd_asso = self.drug_fusion_mlp_asso(torch.cat((drug_hete_emd_asso), dim=1))
        #     #     side_total_emd_asso = self.side_fusion_mlp_asso(torch.cat((side_hete_emd_asso), dim=1))
        #     #     drug_final_emd = self.drug_gat(drug_total_emd_asso, drug_autocorrelation)
        #     #     side_final_emd = self.side_gat(side_total_emd_asso, side_autocorrelation)
        #     #     test_drug_final_emd = torch.matmul(sum(test_drug_sim_info)/len(test_drug_sim_info),drug_final_emd)
        #     #
        #     #     scores_asso = torch.matmul(test_drug_final_emd, side_final_emd.T)
        #     #     # scores_asso = (scores_avg_asso + scores_cat_asso) / 2
        #     #
        #     #     scores_asso = scores_asso.flatten()
        #     # else:
        #     #     scores_asso = scores_avg_asso.flatten()
        #     # # loss_classify = F.binary_cross_entropy_with_logits(scores_asso,true_label,pos_weight=self.posi_weight)
        #     return torch.sigmoid(scores_asso)