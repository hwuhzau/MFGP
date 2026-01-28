from torch.optim import lr_scheduler
import time
from sklearn.metrics import r2_score
from utils import *
from model import MyVGG
from config import Config
from fit import fit
from predata import *
import random

def my_train(myseed=3, mydataset_y_name="grain_weight"):
    config = Config()
    config.seed = myseed
    config.dataset_y_name = mydataset_y_name
    torch.manual_seed(2024)
    np.random.seed(config.seed)
    random.seed(2024)
    time_start = time.time()
    # 创建并实时保存日志
    device = torch.device('cuda')
    logger = bulit_logger(f"{config.path}/log.log")
    config.print_params(logger.info)
    train_dl, valid_dl, test_dl = predata(config, logger)
    # 模型训练
    net = MyVGG(config).to(device)
    # net = nn.DataParallel(net).cuda()
    # net = torch.load(r'./best_model.pth.tar').to(device)
    loss_fn = nn.SmoothL1Loss()
    opti = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    exp_lr_scheduler = lr_scheduler.StepLR(opti, step_size=10, gamma=0.1)

    # 保存在校验集上最优的模型
    best_R2_totrain,  best_R2_toepoch, best_R2 = 0., 0., 0.
    for epoch_i in range(config.epochs):
        epoch_train_R2, epoch_valid_loss, epoch_valid_R2 = fit(epoch_i, net, train_dl, valid_dl,
            device, loss_fn, opti, exp_lr_scheduler, logger, config)
        if best_R2 < epoch_valid_R2:
            best_R2, best_R2_totrain, best_R2_toepoch, is_best = epoch_valid_R2, epoch_train_R2, epoch_i, True
        else:
            is_best = False
        save_model(net, config.path, is_best)
    logger.info(f"best_epoch:{best_R2_toepoch} best_train_R2:{best_R2_totrain} best_val_R2:{best_R2}")
    time_end = time.time()
    time_sum = time_end - time_start
    logger.info('训练阶段耗时 {:.0f}分 {:.0f}秒'.format(time_sum // 60, time_sum % 60))
    logger.info("="*30)
    logger.info("模型预测阶段")
    time_start = time.time()

    try:
        net = torch.load(f'{config.path}/best_model.pth.tar')
        net.eval()
        ture_test, pre_test = [], []
        with torch.no_grad():
            for tes_j, (tes_x, tes_y) in enumerate(test_dl):
                tes_x, tes_y = tes_x.to(device), tes_y.to(device)
                tes_y_pre = net(tes_x)
                tes_loss = loss_fn(tes_y_pre, tes_y)
                ture_test.extend(tes_y.cpu().detach().numpy())
                pre_test.extend(tes_y_pre.cpu().detach().numpy())
            test_R2 = r2_score(torch.tensor(ture_test), torch.tensor(pre_test))
            config.testlog = test_R2
            logger.info(f'{test_R2}')
        y_t, y_p = np.squeeze(np.array(ture_test)), np.squeeze(np.array(pre_test))
        save_data = pd.concat([pd.DataFrame(np.expand_dims(y_t, axis=1)), pd.DataFrame(np.expand_dims(y_p, axis=1))], axis=1)
        save_data.columns = ["y_t", "y_p"]
        save_data.to_csv(f"{config.path}/test_tp.csv")
        # 绘制QQ图
        time_sum = time.time() - time_start
        logger.info('预测阶段耗时 {:.0f}分 {:.0f}秒'.format(time_sum // 60, time_sum % 60))
        config.geneid_sx = 0
        config.geneid_num = 0
        config.jytk_jz = 0
        config.data_jytk_num = 0
        log_csv(best_R2_toepoch, best_R2_totrain, best_R2, test_R2, config)
    except Exception:
        test_R2 = 0
        config.testlog = 0
        config.geneid_sx = 0
        config.geneid_num = 0
        config.jytk_jz = 0
        config.data_jytk_num = 0
        log_csv(best_R2_toepoch, best_R2_totrain, best_R2, test_R2, config)
    test_R = 0 if test_R2 < 0 else np.sqrt(test_R2)
    return test_R
if __name__ == "__main__":
    my_train()




