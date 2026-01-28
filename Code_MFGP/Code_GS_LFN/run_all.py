# coding=utf-8
import csv
import os
from train import my_train
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# x_tzgc_all = [0.04, 0.07, 0.1, 0.13, 0.15, 0.17, 0.2, 0.23, 0.25, 0.27, 0.3, 0.33, 0.35, 0.37]
# x_tzgc_all = [0.22]
# cor_or_mic_all = ["cor"]
dataset_y_name_all = ["pheno1", "pheno2", "pheno3", "pheno4", "pheno5", "pheno6", "pheno7", "pheno8", "pheno9", "pheno10"]

output_file = 'results.csv'

# 如果文件不存在，初始化 CSV 文件并写入表头
if not os.path.isfile(output_file):
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["dataset_y_namei", "got_r_mean"])

# 读取已有结果，避免重复计算
existing_results = set()
with open(output_file, mode='r', newline='') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过表头
    for row in reader:
        existing_results.add((row[1]))

# 遍历参数组合并计算结果
for dataset_y_namei in dataset_y_name_all:
        if (dataset_y_namei) in existing_results:
            continue
        got_r_list = []
        for seedi in range(1, 11):
            got_r_listi = my_train(myseed=seedi, mydataset_y_name=dataset_y_namei)
            print(dataset_y_namei, seedi, got_r_listi)
            got_r_list.append(got_r_listi)
        got_r_mean = sum(got_r_list) / len(got_r_list)

        # 将新结果追加写入 CSV 文件
        with open(output_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([dataset_y_namei, got_r_mean])
        print(dataset_y_namei, got_r_mean)
print(f"Results have been updated in {output_file}")
