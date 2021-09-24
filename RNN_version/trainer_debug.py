# import lib
# from evaluation_sample import *
from evaluation import *
import time
import torch
import numpy as np
import os
from dataset import *
import datetime
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, '../PyTorch_GBW_LM')
sys.path.insert(0, '../PyTorch_GBW_LM/log_uniform')

from log_uniform import LogUniformSampler

class Trainer(object):
    def __init__(self, log, model, beam_size, train_data, eval_data, optim, use_cuda, loss_func, cate_loss_func, topk, input_size, args):
        self.m_log = log
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.optim = optim
        self.m_loss_func = loss_func
        self.topk = topk
        self.evaluation = Evaluation(self.m_log, self.model, self.m_loss_func, use_cuda, self.topk, warm_start=args.warm_start)
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.args = args

        self.m_cate_loss_func = cate_loss_func

        ### early stopping
        self.m_patience = args.patience
        self.m_best_recall = 0.0
        self.m_best_mrr = 0.0
        self.m_early_stop = False
        self.m_counter = 0

        self.m_best_cate_recall = 0.0
        self.m_best_cate_mrr = 0.0

        self.m_batch_iter = 0

        #### sample negative items
        self.m_sampler = LogUniformSampler(input_size)
        self.m_nsampled = args.negative_num
        self.m_remove_match = True

        self.m_teacher_forcing_ratio = 2.0
        self.m_beam_size = beam_size
        self.m_teacher_forcing_flag = True

        self.m_logsoftmax = nn.LogSoftmax(dim=1)

    def saveModel(self, epoch, loss, recall, mrr):
        checkpoint = {
            'model': self.model.state_dict(),
            'args': self.args,
            'epoch': epoch,
            'optim': self.optim,
            'loss': loss,
            'recall': recall,
            'mrr': mrr
        }
        model_name = os.path.join(self.args.checkpoint_dir, "model_best.pt")
        torch.save(checkpoint, model_name)

    def train(self, start_epoch, end_epoch, batch_size, start_time=None):

        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time

        ### start training
        for epoch in range(start_epoch, end_epoch + 1):
            
            msg = "*"*10+str(epoch)+"*"*5
            self.m_log.addOutput2IO(msg)
            print("teaching", self.m_teacher_forcing_flag)
            st = time.time()

            ### an epoch
            train_mixture_loss, train_loss, train_cate_loss = self.train_epoch(epoch, batch_size)

            ### evaluate model on train dataset or validation dateset
            mixture_loss, loss, recall, mrr, cate_loss, cate_recall, cate_mrr = self.evaluation.eval(self.train_data, batch_size, "train")

            print("train", train_loss)
            print("mix", mixture_loss)
            print("loss", loss)
            print("recall", recall)
            print("mrr", mrr)

            msg = "train Epoch: {}, train loss: {:.4f},  mixture loss: {:.4f}, loss: {:.4f}, recall: {:.4f}, mrr: {:.4f}, cate_loss: {:.4f}, cate_recall: {:.4f}, cate_mrr: {:.4f}, time: {}".format(epoch, train_mixture_loss, mixture_loss, loss, recall, mrr, cate_loss, cate_recall, cate_mrr, time.time() - st)
            self.m_log.addOutput2IO(msg)
            self.m_log.addScalar2Tensorboard("train_mixture_loss", train_mixture_loss, epoch)
            self.m_log.addScalar2Tensorboard("mixture_loss", mixture_loss, epoch)
            self.m_log.addScalar2Tensorboard("train_loss_eval", loss, epoch)
            self.m_log.addScalar2Tensorboard("train_recall", recall, epoch)
            self.m_log.addScalar2Tensorboard("train_mrr", mrr, epoch)

            self.m_log.addScalar2Tensorboard("train_cate_loss_eval", cate_loss, epoch)
            self.m_log.addScalar2Tensorboard("train_cate_recall", cate_recall, epoch)
            self.m_log.addScalar2Tensorboard("train_cate_mrr", cate_mrr, epoch)

            if self.m_best_cate_recall == 0:
                self.m_best_cate_recall = cate_recall
            elif self.m_best_cate_recall >= cate_recall:
                self.m_teacher_forcing_flag = False
                # self.m_teacher_forcing_flag = True
            else:
                self.m_best_cate_recall = cate_recall

            ### evaluate model on test dataset
            mixture_loss, loss, recall, mrr, cate_loss, cate_recall, cate_mrr = self.evaluation.eval(self.eval_data, batch_size, "test")
            msg = "Epoch: {}, mixture loss: {:.4f}, loss: {:.4f}, recall: {:.4f}, mrr: {:.4f}, cate_loss: {:.4f}, cate_recall: {:.4f}, cate_mrr: {:.4f}, time: {}".format(epoch, mixture_loss, loss, recall, mrr, cate_loss, cate_recall, cate_mrr, time.time() - st)
            self.m_log.addOutput2IO(msg)
            self.m_log.addScalar2Tensorboard("test_mixture_loss", mixture_loss, epoch)
            self.m_log.addScalar2Tensorboard("test_loss", loss, epoch)
            self.m_log.addScalar2Tensorboard("test_recall", recall, epoch)
            self.m_log.addScalar2Tensorboard("test_mrr", mrr, epoch)

            self.m_log.addScalar2Tensorboard("test_cate_loss", cate_loss, epoch)
            self.m_log.addScalar2Tensorboard("test_cate_recall", cate_recall, epoch)
            self.m_log.addScalar2Tensorboard("test_cate_mrr", cate_mrr, epoch)
            
            if self.m_best_recall == 0:
                self.m_best_recall = recall
                self.saveModel(epoch, loss, recall, mrr)
            elif self.m_best_recall > recall:
                self.m_counter += 1
                if self.m_counter > self.m_patience:
                    break
                msg = "early stop counter "+str(self.m_counter)
                self.m_log.addOutput2IO(msg)
            else:
                self.m_best_recall = recall
                self.m_best_mrr = mrr
                self.saveModel(epoch, loss, recall, mrr)
                self.m_counter = 0

            msg = "best recall: "+str(self.m_best_recall)+"\t best mrr: \t"+str(self.m_best_mrr)
            self.m_log.addOutput2IO(msg)

    def train_epoch(self, epoch, batch_size):
        self.model.train()
        
        losses = []
        cate_losses = []
        mixture_losses = []

        def reset_hidden(hidden, mask):
            """Helper function that resets hidden state when some sessions terminate"""
            if len(mask) != 0:
                hidden[:, mask, :] = 0
            return hidden
       
        dataloader = self.train_data
        
        for x_long_iter, xnum_long_iter, c_long_iter, cnum_long_iter, x_short_iter, xnum_short_iter, c_short_iter, y_iter, z_iter, y_idx_iter in dataloader:
        
            ### negative samples
            sample_values = self.m_sampler.sample(self.m_nsampled, y_iter)
            sample_ids, true_freq, sample_freq = sample_values

            ### whether excluding current pos sample from negative samples
            if self.m_remove_match:
                acc_hits = self.m_sampler.accidental_match(y_iter, sample_ids)
                acc_hits = list(zip(*acc_hits))

            ### items of long-range actions
            x_long_iter = x_long_iter.to(self.device)

            ### num of items of long-range actions
            xnum_long_iter = xnum_long_iter.to(self.device)

             ### cates of long-range actions
            c_long_iter = c_long_iter.to(self.device)

            cnum_long_iter = cnum_long_iter.to(self.device)
           
            ### items of short-range actions
            x_short_iter = x_short_iter.to(self.device)

            ### number of items of short-range actions
            xnum_short_iter = xnum_short_iter.to(self.device)

            ### cate of short-range actions
            c_short_iter = c_short_iter.to(self.device)

            ### items of target action
            y_iter = y_iter.to(self.device)
            
            ### cates of target action
            z_iter = z_iter.to(self.device)

            ### idx of target action in seq 
            y_idx_iter = y_idx_iter.to(self.device)
            # batch_size = x_batch.size(0)

            self.optim.zero_grad()

            output_cate = self.model.m_cateNN(c_short_iter, xnum_short_iter, "train")
            logit_cate = self.model.m_cateNN.m_cate_h2o(output_cate)

            pred_item_prob = None

            if self.m_teacher_forcing_flag:
                output_action_long, output_action_short = self.model.m_itemNN(x_long_iter, xnum_long_iter, c_long_iter, cnum_long_iter, x_short_iter, xnum_short_iter, z_iter, "train")
               
                mix_output_action = torch.cat((output_action_long, output_action_short), dim=1)
                logit_action = self.model.fc(mix_output_action)

                sampled_logit_batch, sampled_target_batch = self.model.m_ss(logit_action, y_iter, sample_ids, true_freq, sample_freq, acc_hits, self.device, self.m_remove_match)

                sampled_prob_batch = self.m_logsoftmax(sampled_logit_batch)
                pred_item_prob = sampled_prob_batch

            else:
                log_prob_cate_short = self.m_logsoftmax(logit_cate)
                log_prob_cate_short, pred_cate_index = torch.topk(log_prob_cate_short, self.m_beam_size, dim=-1)

                pred_cate_index = pred_cate_index.detach()
                log_prob_cate_short = log_prob_cate_short

                for beam_index in range(self.m_beam_size):
                    pred_cate_beam = pred_cate_index[:, beam_index]
                    prob_cate_beam = log_prob_cate_short[:, beam_index]

                    output_action_long, output_action_short = self.model.m_itemNN(x_long_iter, xnum_long_iter, c_long_iter, cnum_long_iter, x_short_iter, xnum_short_iter, z_iter, "train")

                    mix_output_action = torch.cat((output_action_long, output_action_short), dim=1)
                    logit_action = self.model.fc(mix_output_action)

                    sampled_logit_batch, sampled_target_batch = self.model.m_ss(logit_action, y_iter, sample_ids, true_freq, sample_freq, acc_hits, self.device, self.m_remove_match)

                    sampled_prob_batch = self.m_logsoftmax(sampled_logit_batch)

                    if pred_item_prob is None:
                        pred_item_prob = sampled_prob_batch+prob_cate_beam.reshape(-1, 1)
                        pred_item_prob = pred_item_prob.unsqueeze(-1)
                    else:
                        pred_item_prob_beam = sampled_prob_batch+prob_cate_beam.reshape(-1, 1)
                        pred_item_prob_beam = pred_item_prob_beam.unsqueeze(-1)
                        pred_item_prob = torch.cat((pred_item_prob, pred_item_prob_beam), dim=-1)
                
                pred_item_prob = torch.logsumexp(pred_item_prob, dim=-1)

            loss_batch = self.m_loss_func(pred_item_prob, sampled_target_batch, "prob")
            losses.append(loss_batch.item())
       
            cate_loss_batch = self.m_cate_loss_func(logit_cate, z_iter, "logit")
            cate_losses.append(cate_loss_batch.item())
            
            mixture_loss_batch = loss_batch+cate_loss_batch
            
            mixture_losses.append(mixture_loss_batch.item())

            mixture_loss_batch.backward()

            max_norm = 5.0

            self.m_batch_iter += 1

            torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm)

            self.optim.step()

        mean_mixture_losses = np.mean(mixture_losses)

        mean_losses = np.mean(losses)

        mean_cate_losses = np.mean(cate_losses)

        return mean_mixture_losses, mean_losses, mean_cate_losses