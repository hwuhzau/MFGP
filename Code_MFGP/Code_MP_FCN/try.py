# coding=utf-8
import subprocess

# 总种子范围
start_seed = 1
end_seed = 20
# 每批运行的种子数
batch_size = 5

# 创建一个种子列表
seeds = [str(seedi) for seedi in range(start_seed, end_seed + 1)]

# 分批处理
for i in range(0, len(seeds), batch_size):
    # 获取当前批次的种子
    current_batch = seeds[i:i + batch_size]

    # 启动每个训练进程
    processes = []
    for seedi in current_batch:
        process = subprocess.Popen(['python', 'train.py', '--seed', seedi])
        processes.append(process)

    # 等待当前批次的所有进程完成
    for process in processes:
        process.wait()
