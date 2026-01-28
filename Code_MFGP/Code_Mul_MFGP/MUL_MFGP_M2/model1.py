from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

# 直接拼接
class MyVGG(nn.Module):
    """自主修改的resnet模型架构"""
    def __init__(self, config):
        super(MyVGG, self).__init__()
        self.step = config.step
        self.num_genes = config.genenum
        self.geneid_sx = config.geneid_sx
        self.geneid_num = config.geneid_num

        # 创建一个字典来存储每个基因的全连接层
        self.fc_layers1 = nn.ModuleList()
        for numi in self.geneid_num:
            self.fc_layers1.append(nn.Linear(numi, 1))
        self.mul_len1 = config.mul_len1
        self.mul_len2 = config.mul_len2
        self.mul_len3 = config.mul_len3
        self.fc0 = nn.Sequential(
            nn.Linear(self.num_genes, 64),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(self.mul_len1, 64),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.mul_len2, 64),
            nn.ReLU(inplace=True),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(self.mul_len3, 64),
            nn.ReLU(inplace=True),
        )


        self.fc01 = nn.Sequential(
            nn.Linear(64*2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.fc02 = nn.Sequential(
            nn.Linear(64*2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.fc03 = nn.Sequential(
            nn.Linear(64*2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.fc12 = nn.Sequential(
            nn.Linear(64*2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.fc13 = nn.Sequential(
            nn.Linear(64*2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.fc23 = nn.Sequential(
            nn.Linear(64*2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.fc012 = nn.Sequential(
            nn.Linear(64*3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.fc013 = nn.Sequential(
            nn.Linear(64*3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.fc123 = nn.Sequential(
            nn.Linear(64*3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.fc1234 = nn.Sequential(
            nn.Linear(64*4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x, z1, z2, z3):
        if self.step == "01":
            # 选择对应的特征并通过全连接层处理
            gene_outputs = []
            for gene_index in range(self.num_genes):
                start_idx = sum(self.geneid_num[:gene_index])
                end_idx = sum(self.geneid_num[:gene_index + 1])
                gene_features = x[:, start_idx:end_idx]  # 选择对应的特征
                gene_output = F.relu(self.fc_layers1[gene_index](gene_features))
                gene_outputs.append(gene_output)
            # 将所有基因的输出连接起来
            gene_outputs = torch.cat(gene_outputs, dim=1)
            data_1 = self.fc1(z1)
            data_1 = torch.cat((data_1, self.fc0(gene_outputs)), dim=1)
            x = self.fc01(data_1)
            return x
        if self.step == "02":
            # 选择对应的特征并通过全连接层处理
            gene_outputs = []
            for gene_index in range(self.num_genes):
                start_idx = sum(self.geneid_num[:gene_index])
                end_idx = sum(self.geneid_num[:gene_index + 1])
                gene_features = x[:, start_idx:end_idx]  # 选择对应的特征
                gene_output = F.relu(self.fc_layers1[gene_index](gene_features))
                gene_outputs.append(gene_output)
            # 将所有基因的输出连接起来
            gene_outputs = torch.cat(gene_outputs, dim=1)
            data_1 = self.fc2(z2)
            data_1 = torch.cat((data_1, self.fc0(gene_outputs)), dim=1)
            x = self.fc02(data_1)
            return x
        if self.step == "03":
            # 选择对应的特征并通过全连接层处理
            gene_outputs = []
            for gene_index in range(self.num_genes):
                start_idx = sum(self.geneid_num[:gene_index])
                end_idx = sum(self.geneid_num[:gene_index + 1])
                gene_features = x[:, start_idx:end_idx]  # 选择对应的特征
                gene_output = F.relu(self.fc_layers1[gene_index](gene_features))
                gene_outputs.append(gene_output)
            # 将所有基因的输出连接起来
            gene_outputs = torch.cat(gene_outputs, dim=1)
            data_1 = self.fc3(z3)
            data_1 = torch.cat((data_1, self.fc0(gene_outputs)), dim=1)
            x = self.fc03(data_1)
            return x
        if self.step == "012":
            # 选择对应的特征并通过全连接层处理
            gene_outputs = []
            for gene_index in range(self.num_genes):
                start_idx = sum(self.geneid_num[:gene_index])
                end_idx = sum(self.geneid_num[:gene_index + 1])
                gene_features = x[:, start_idx:end_idx]  # 选择对应的特征
                gene_output = F.relu(self.fc_layers1[gene_index](gene_features))
                gene_outputs.append(gene_output)
            # 将所有基因的输出连接起来
            gene_outputs = torch.cat(gene_outputs, dim=1)
            data_1 = self.fc1(z1)
            data_1 = torch.cat((data_1, self.fc0(gene_outputs)), dim=1)
            data_2 = self.fc2(z2)
            data_2 = torch.cat((data_2, data_1), dim=1)
            x = self.fc012(data_2)
            return x
        if self.step == "013":
            # 选择对应的特征并通过全连接层处理
            gene_outputs = []
            for gene_index in range(self.num_genes):
                start_idx = sum(self.geneid_num[:gene_index])
                end_idx = sum(self.geneid_num[:gene_index + 1])
                gene_features = x[:, start_idx:end_idx]  # 选择对应的特征
                gene_output = F.relu(self.fc_layers1[gene_index](gene_features))
                gene_outputs.append(gene_output)
            # 将所有基因的输出连接起来
            gene_outputs = torch.cat(gene_outputs, dim=1)
            data_1 = self.fc1(z1)
            data_1 = torch.cat((data_1, self.fc0(gene_outputs)), dim=1)
            data_2 = self.fc3(z3)
            data_2 = torch.cat((data_2, data_1), dim=1)
            x = self.fc013(data_2)
            return x
        if self.step == "123":
            # 选择对应的特征并通过全连接层处理
            # gene_outputs = []
            # for gene_index in range(self.num_genes):
            #     start_idx = sum(self.geneid_num[:gene_index])
            #     end_idx = sum(self.geneid_num[:gene_index + 1])
            #     gene_features = x[:, start_idx:end_idx]  # 选择对应的特征
            #     gene_output = F.relu(self.fc_layers1[gene_index](gene_features))
            #     gene_outputs.append(gene_output)
            # # 将所有基因的输出连接起来
            # gene_outputs = torch.cat(gene_outputs, dim=1)
            data3 = self.fc3(z3)
            data_1 = self.fc1(z1)
            data_1 = torch.cat((data_1, data3), dim=1)
            data_2 = self.fc2(z2)
            data_2 = torch.cat((data_2, data_1), dim=1)
            x = self.fc123(data_2)
            return x
        if self.step == "12":
            data3 = self.fc2(z2)
            data_1 = self.fc1(z1)
            data_1 = torch.cat((data_1, data3), dim=1)
            x = self.fc12(data_1)
            return x
        if self.step == "23":
            data3 = self.fc2(z2)
            data_1 = self.fc3(z3)
            data_1 = torch.cat((data_1, data3), dim=1)
            x = self.fc23(data_1)
            return x
        if self.step == "13":
            data3 = self.fc1(z1)
            data_1 = self.fc3(z3)
            data_1 = torch.cat((data_1, data3), dim=1)
            x = self.fc13(data_1)
            return x
        if self.step == "0123":
            # 选择对应的特征并通过全连接层处理
            gene_outputs = []
            for gene_index in range(self.num_genes):
                start_idx = sum(self.geneid_num[:gene_index])
                end_idx = sum(self.geneid_num[:gene_index + 1])
                gene_features = x[:, start_idx:end_idx]  # 选择对应的特征
                gene_output = F.relu(self.fc_layers1[gene_index](gene_features))
                gene_outputs.append(gene_output)
            # 将所有基因的输出连接起来
            gene_outputs = torch.cat(gene_outputs, dim=1)
            data_1 = self.fc1(z1)
            data_1 = torch.cat((data_1, self.fc0(gene_outputs)), dim=1)
            data_2 = self.fc2(z2)
            data_2 = torch.cat((data_2, data_1), dim=1)
            data_3 = self.fc3(z3)
            data_3 = torch.cat((data_3, data_2), dim=1)
            x = self.fc0123(data_3)
            return x

