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
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(self.mul_len1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.mul_len2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(self.mul_len3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.weight0 = nn.Parameter(torch.tensor(0.5))
        self.weight1 = nn.Parameter(torch.tensor(0.5))
        self.weight2 = nn.Parameter(torch.tensor(0.5))
        self.weight3 = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, z1, z2, z3):
        if self.step == "s01":
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
            output0 = self.fc0(gene_outputs)
            output1 = self.fc1(z1)
            return self.weight0 * output0 + self.weight1 * output1
        if self.step == "s02":
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
            output0 = self.fc0(gene_outputs)
            output2 = self.fc2(z2)
            return self.weight0 * output0 + self.weight2 * output2
        if self.step == "s03":
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
            output0 = self.fc0(gene_outputs)
            output3 = self.fc3(z3)
            return self.weight0 * output0 + self.weight3 * output3
        if self.step == "s12":
            output1 = self.fc1(z1)
            output2 = self.fc2(z2)
            return self.weight1 * output1 + self.weight2 * output2
        if self.step == "s23":
            output2 = self.fc2(z2)
            output3 = self.fc3(z3)
            return self.weight2 * output2 + self.weight3 * output3
        if self.step == "s13":
            output1 = self.fc1(z1)
            output3 = self.fc3(z3)
            return self.weight1 * output1 + self.weight3 * output3
        if self.step == "s012":
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
            output0 = self.fc0(gene_outputs)
            output1 = self.fc1(z1)
            output2 = self.fc2(z2)
            return self.weight0 * output0 + self.weight1 * output1 + self.weight2 * output2
        if self.step == "s013":
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
            output0 = self.fc0(gene_outputs)
            output1 = self.fc1(z1)
            output3 = self.fc3(z3)
            return self.weight0 * output0 + self.weight1 * output1 + self.weight3 * output3
        if self.step == "s023":
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
            output0 = self.fc0(gene_outputs)
            output1 = self.fc2(z2)
            output3 = self.fc3(z3)
            return self.weight0 * output0 + self.weight1 * output1 + self.weight3 * output3
        if self.step == "s123":
            output1 = self.fc1(z1)
            output2 = self.fc2(z2)
            output3 = self.fc3(z3)
            return self.weight1 * output1 + self.weight2 * output2 + self.weight3 * output3
        if self.step == "s0123":
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
            output0 = self.fc0(gene_outputs)
            output1 = self.fc1(z1)
            output2 = self.fc2(z2)
            output3 = self.fc3(z3)
            return self.weight0 * output0 + self.weight1 * output1 + self.weight2 * output2 + self.weight3 * output3


