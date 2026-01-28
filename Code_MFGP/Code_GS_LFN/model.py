from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

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


        self.fc1 = nn.Sequential(
            nn.Linear(self.num_genes, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        if self.step == "snp":
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

            jytk_outputs = gene_outputs
            x = self.fc1(jytk_outputs)
            return x

