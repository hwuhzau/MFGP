import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from utils import Mydataset_xy, Mydataset_xyz, standardization_mu_sigama, standardization
from scipy.stats import pearsonr


def predata(config, logger):
    data_y_b = pd.read_csv(f'../data/{config.dataset_y}')[f"{config.dataset_y_name}"]
    data_y_b = np.array(data_y_b)
    # print("data_y_数据：", data_y_b.shape)
    data_x_b = pd.read_csv(f"../data/{config.dataset_x}")
    # print("data_x_数据：", data_x_b.shape)
    data_x_b = np.array(data_x_b)

    data_x_mask = pd.read_csv(f'../data/snp_cor_{config.dataset_y_name}.csv')

    data_x_mask = np.array(data_x_mask).astype(float)
    data_x_mask = np.nan_to_num(data_x_mask, nan=0.0)  # 将 NaN 替换为 0
    data_x_mask = np.abs(data_x_mask) > float(config.x_tzgc)
    data_x_mask = np.array(data_x_mask).flatten()
    # print(data_x_mask.shape, data_x_b.shape)
    data_x_b = data_x_b[data_x_mask]
    # print("特征工程后，data_x_数据：", data_x_b.shape)
    gene_id = np.array(pd.DataFrame(data_x_b))[:, 2]
    for_z_gene_id = list(dict.fromkeys(gene_id))
    # print("设计涉及SNP个数：", gene_id.shape, "对应基因个数：", len(for_z_gene_id))
    # print("SNP层面固定的基因顺序：", gene_id[:10])
    # print("固定的基因顺序：", for_z_gene_id[:10])

    # 创建一个字典来映射类别到整数
    category_mapping = {}
    category_counter = 0
    # 生成类别列表和计数列表
    category_list = []
    count_list = []
    # 记录每个类别的计数
    count_dict = {}
    for item in gene_id:
        if item not in category_mapping:
            category_mapping[item] = category_counter
            count_dict[item] = 0  # 初始化计数
            category_counter += 1
        category_list.append(category_mapping[item])
        count_dict[item] += 1  # 增加计数
    # 生成计数列表
    for key in sorted(count_dict.keys(), key=lambda x: category_mapping[x]):
        count_list.append(count_dict[key])

    # print("SNP对应基因情况列表：（前20个）", category_list[:20])
    # print("SNP个数总和：", len(category_list))
    # print("基因对应SNP个数列表：（前10个）", count_list[:10])
    # print("基因个数：", len(count_list))
    # print("单个基因中最大的SNP个数：", max(count_list))
    # print("SNP个数总和：", sum(count_list))
    config.final_genenum = len(count_list)
    config.final_snpnum = sum(count_list)
    config.genenum = len(count_list)
    config.geneid_sx = category_list
    config.geneid_num = count_list
    data_x_b = np.array(pd.DataFrame(data_x_b))[:, 9:].T
    # print(len(config.geneid_sx))
    # print(len(config.geneid_num))

    # print("data_x_数据：", data_x_b.shape)

    index = np.random.permutation(len(data_y_b))
    data_x_b, data_y_b = np.array(data_x_b)[index], np.array(data_y_b)[index]
    s1, s2 = int(len(data_y_b) * 0.8), int(len(data_y_b) * 0.9)
    train_x, valid_x, test_x = data_x_b[:s1], data_x_b[s1:s2], data_x_b[s2:]
    train_y, valid_y, test_y = data_y_b[:s1], data_y_b[s1:s2], data_y_b[s2:]
    logger.info(f'总数据量：{len(train_y) + len(valid_y) + len(test_y)} '
                f'train_len: {len(train_y)} val_len: {len(valid_y)} test_len: {len(test_y)}')
    train_y = pd.DataFrame(train_y)
    train_y.dropna(inplace=True)
    train_x = train_x[train_y.index]
    train_y = np.array(train_y)

    valid_y = pd.DataFrame(valid_y)
    valid_y.dropna(inplace=True)
    valid_x = valid_x[valid_y.index]
    valid_y = np.array(valid_y)

    test_y = pd.DataFrame(test_y)
    test_y.dropna(inplace=True)
    test_x = test_x[test_y.index]
    test_y = np.array(test_y)
    logger.info(f'shape：{train_x.shape} ')

    logger.info(f'总数据量：{len(train_y) + len(valid_y) + len(test_y)} '
                f'train_len: {len(train_y)} val_len: {len(valid_y)} test_len: {len(test_y)}')
    # 数据批量打包
    train_dl = DataLoader(Mydataset_xy(train_x, train_y), batch_size=config.train_batch, shuffle=True)
    valid_dl = DataLoader(Mydataset_xy(valid_x, valid_y), batch_size=config.valid_batch)
    test_dl = DataLoader(Mydataset_xy(test_x, test_y), batch_size=len(test_y), shuffle=False)
    return train_dl, valid_dl, test_dl
