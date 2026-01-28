import torch
from sklearn.metrics import r2_score
from torch import nn


def fit(epoch, model, trainloader, validloader,
        device, loss_fn, opti, exp_lr_scheduler, logger, config):
    model.train()
    ture_train, pre_train = [], []
    train_running_loss = 0
    for tra_j, (tra_x, tra_y, tra_z1, tra_z2, tra_z3) in enumerate(trainloader):
        tra_x, tra_y = tra_x.to(device), tra_y.to(device)
        tra_z1, tra_z2, tra_z3 = tra_z1.to(device), tra_z2.to(device), tra_z3.to(device)
        tra_y_pre = model(tra_x, tra_z1, tra_z2, tra_z3)
        tra_loss = loss_fn(tra_y_pre, tra_y)
        opti.zero_grad()
        tra_loss.backward()
        opti.step()
        ture_train.extend(tra_y.cpu().detach().numpy())
        pre_train.extend(tra_y_pre.cpu().detach().numpy())
        with torch.no_grad():
            train_running_loss += tra_loss.item()
    # exp_lr_scheduler.step()
    train_R2 = r2_score(torch.tensor(ture_train), torch.tensor(pre_train))
    train_loss = train_running_loss / len(trainloader.dataset)

    model.eval()
    ture_valid, pre_valid = [], []
    valid_running_loss = 0
    with torch.no_grad():
        for val_j, (val_x, val_y, val_z1, val_z2, val_z3) in enumerate(validloader):
            val_x, val_y = val_x.to(device), val_y.to(device)
            val_z1, val_z2, val_z3 = val_z1.to(device), val_z2.to(device), val_z3.to(device)
            val_y_pre = model(val_x, val_z1, val_z2, val_z3)
            val_loss = loss_fn(val_y_pre, val_y)
            ture_valid.extend(val_y.cpu().detach().numpy())
            pre_valid.extend(val_y_pre.cpu().detach().numpy())
            valid_running_loss += val_loss.item()
        valid_R2 = r2_score(torch.tensor(ture_valid), torch.tensor(pre_valid))
        valid_loss = valid_running_loss / len(validloader.dataset)

    logger.info("epoch: {}/{}  train_loss: {:.5g}   "
                "train_R2: {:.5g}   val_loss: {:.5g}   "
                "val_R2: {:.5g}".format(epoch + 1, config.epochs, train_loss,
                                        train_R2, valid_loss, valid_R2))
    return train_R2, valid_loss, valid_R2
