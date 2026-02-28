from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import SVQ
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time
import warnings
import numpy as np


warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'SVQ': SVQ,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("trainable parameters:", str(trainable_num/1e6), "M")
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'mse':
            criterion = nn.MSELoss()
        elif self.args.loss == 'mae':
            criterion = nn.L1Loss()
        else:
            criterion = nn.HuberLoss(delta=self.args.delta , reduction='mean')
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_codebook_utilization = []
        total_codebook_perplexity = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                outputs, vq_loss, codebook_utilization, codebook_perplexity = self.model(batch_x)
                pred = outputs.detach().cpu()
                batch_x = batch_x.detach().cpu()
                loss = criterion(pred, batch_x)
                total_loss.append(loss)
                total_codebook_utilization.append(codebook_utilization)
                total_codebook_perplexity.append(codebook_perplexity.detach().cpu().numpy())
        total_loss = np.average(total_loss)
        avg_codebook_utilization = np.average(total_codebook_utilization)
        avg_codebook_perplexity = np.average(total_codebook_perplexity)
        self.model.train()
        return total_loss, avg_codebook_utilization, avg_codebook_perplexity

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                    steps_per_epoch = train_steps,
                                    pct_start = self.args.pct_start,
                                    epochs = self.args.train_epochs,
                                    max_lr = self.args.learning_rate)

        best_vali_loss = float('inf')

        f = open("result.txt", 'a')
        f.write(setting + "\n\n")



        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            total_codebook_utilization = []
            total_codebook_perplexity = []
            for i, (batch_x) in enumerate(train_loader):
                iter_count += 1
                
                batch_x = batch_x.float().to(self.device)
                outputs, vq_loss, codebook_utilization, codebook_perplexity = self.model(batch_x)
                total_codebook_utilization.append(codebook_utilization)
                total_codebook_perplexity.append(codebook_perplexity.detach().cpu().numpy())

                loss = criterion(outputs, batch_x)

                loss = loss + vq_loss

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    # print("iters: {0}, epoch: {1} | loss: {2:.7f} | codebook_utilization: {3:.4f}% | codebook_perplexity: {4:.4f}".format(i + 1, epoch + 1, loss.item(), codebook_utilization * 100, codebook_perplexity))
                    print("iters: {0}, epoch: {1} | loss: {2:.7f} ".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                model_optim.zero_grad()

            train_codebook_utilization = np.average(total_codebook_utilization)
            train_codebook_perplexity = np.average(total_codebook_perplexity)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))


            train_loss = np.average(train_loss)
            vali_loss, vali_codebook_utilization, vali_codebook_perplexity = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_codebook_utilization, test_codebook_perplexity = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            print("Epoch: {0} |  Train Codebook Utilization: {1:.4f}% Train Codebook_Perplexity: {2:.4f}".format(
                epoch + 1, train_codebook_utilization * 100, train_codebook_perplexity))


            print("Epoch: {0} |  vali Codebook Utilization: {1:.4f}% vali Codebook_Perplexity: {2:.4f}".format(
                epoch + 1, vali_codebook_utilization * 100, vali_codebook_perplexity))
            
            print("Epoch: {0} |  Test Codebook Utilization: {1:.4f}% Test Codebook_Perplexity: {2:.4f}".format(
                epoch + 1, test_codebook_utilization * 100, test_codebook_perplexity))

            adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

            # -------------------------------------
            # ✅ 保存验证集损失最低时的 codebook 指标
            # -------------------------------------
            if vali_loss < best_vali_loss:
                best_vali_loss = vali_loss
                best_train_codebook_utilization = train_codebook_utilization
                best_train_codebook_perplexity = train_codebook_perplexity
                best_epoch = epoch
        
                with open("result.txt", 'a') as f:
                    # f.write(setting + "  \n")
                    f.write("Best Epoch: {0} | Vali Loss: {1:.7f} | Train Codebook Utilization: {2:.4f}% | Train Codebook_Perplexity: {3:.4f}".format(
                        best_epoch + 1,
                        best_vali_loss,
                        best_train_codebook_utilization * 100,
                        best_train_codebook_perplexity))
                    f.write('\n\n')


            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            print('Updating learning rate to {}'.format(model_optim.param_groups[0]['lr']))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=1):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        inputx = []
        total_codebook_utilization = []
        total_codebook_perplexity = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                outputs, vq_loss, codebook_utilization, codebook_perplexity = self.model(batch_x)
                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                preds.append(pred)
                total_codebook_utilization.append(codebook_utilization)
                total_codebook_perplexity.append(codebook_perplexity.detach().cpu().numpy())

                inputx.append(batch_x.detach().cpu().numpy())
                # if i % 20 == 0:
                # input = batch_x.detach().cpu().numpy()
                # gt = (input[0, :, -1])
                # pd = (pred[0, :, -1])
                # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

                    # input = batch_x.detach().cpu().numpy()
                    # gt = (input[0, :, -2])
                    # pd = (pred[0, :, -2])
                    # visual(gt, pd, os.path.join(folder_path, str(i+1) + '.pdf'))

                    # input = batch_x.detach().cpu().numpy()
                    # gt = (input[0, :, -3])
                    # pd = (pred[0, :, -3])
                    # visual(gt, pd, os.path.join(folder_path, str(i+2) + '.pdf'))

        preds = np.array(preds)
        inputx = np.array(inputx)

        avg_codebook_utilization = np.average(total_codebook_utilization)
        avg_codebook_perplexity = np.average(total_codebook_perplexity)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, r2 = metric(preds.reshape(preds.shape[0],-1), inputx.reshape(inputx.shape[0],-1))
        print('mae:{0}, mse:{1}, rmse:{2}, mape:{3}, r2:{4}'.format(mae, mse, rmse, mape, r2))
        print('Codebook Utilization:{0:.4f}%, Codebook Perplexity:{1:.4f}'.format(avg_codebook_utilization * 100, avg_codebook_perplexity))
        f = open("result.txt", 'a')
        f.write(setting + "\n\n")
        f.write('mae:{0}, mse:{1}, rmse:{2}, mape:{3}, r2:{4}'.format(mae, mse, rmse, mape, r2))
        f.write('\n\n')
        f.write('Codebook Utilization:{0:.4f}%, Codebook Perplexity:{1:.4f}'.format(avg_codebook_utilization * 100, avg_codebook_perplexity))
        f.write('\n\n')
        f.close()

        # np.save(folder_path + 'pred.npy', preds)

        return

