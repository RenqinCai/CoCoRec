import numpy as np
import torch
import dataset
from metric import *
import datetime
import torch.nn.functional as F
import random

import sys
sys.path.insert(0, '../PyTorch_GBW_LM')
sys.path.insert(0, '../PyTorch_GBW_LM/log_uniform')

from log_uniform import LogUniformSampler

class Evaluation(object):
	def __init__(self, log, model, loss_func, use_cuda, input_size, k=20, warm_start=5):
		self.model = model
		self.m_loss_func = loss_func

		self.topk = k
		self.warm_start = warm_start
		self.device = torch.device('cuda' if use_cuda else 'cpu')
		self.m_log = log
		beam_size = 5
		self.m_beam_size = beam_size

		self.m_sampler = LogUniformSampler(input_size)
		self.m_nsampled = 100
		self.m_remove_match = True
		print("evaluation is based on sampled 100", self.m_nsampled)

	def eval(self, eval_data, batch_size, train_test_flag):
		self.model.eval()

		mixture_losses = []

		losses = []
		recalls = []
		mrrs = []
		weights = []

		cate_losses = []
		cate_recalls = []
		cate_mrrs = []
		cate_weights = []

		dataloader = eval_data

		with torch.no_grad():
			total_test_num = []
			
			for x_long_cate_action_batch, x_long_cate_batch, mask_long_cate_action_batch, mask_long_cate_batch, max_long_cate_actionNum_batch, max_long_cateNum_batch, pad_x_long_cate_actionNum_batch, x_long_cateNum_batch, x_short_action_batch, x_short_cate_batch, mask_short_action_batch, pad_x_short_actionNum_batch, y_action_batch, y_cate_batch, y_action_idx_batch in dataloader:
				
				###speed evaluation for train dataset
				if train_test_flag == "train":
					eval_flag = random.randint(1,101)
					if eval_flag != 10:
						continue

				sample_values = self.m_sampler.sample(self.m_nsampled, y_action_batch)
				sample_ids, true_freq, sample_freq = sample_values

			### whether excluding current pos sample from negative samples
				if self.m_remove_match:
					acc_hits = self.m_sampler.accidental_match(y_action_batch, sample_ids)
					acc_hits = list(zip(*acc_hits))
				
				x_long_cate_action_batch = x_long_cate_action_batch.to(self.device)

				mask_long_cate_action_batch = mask_long_cate_action_batch.to(self.device)

				x_long_cate_batch = x_long_cate_batch.to(self.device)
				mask_long_cate_batch = mask_long_cate_batch.to(self.device)
				
				x_short_action_batch = x_short_action_batch.to(self.device)
				mask_short_action_batch = mask_short_action_batch.to(self.device)

				x_short_cate_batch = x_short_cate_batch.to(self.device)

				y_action_batch = y_action_batch.to(self.device)
				y_cate_batch = y_cate_batch.to(self.device)

				warm_start_mask = (y_action_idx_batch>=self.warm_start)

				### cateNN
				seq_cate_short_input = self.model.m_cateNN(x_short_cate_batch, mask_short_action_batch, pad_x_short_actionNum_batch, "test")
				logit_cate_short = self.model.m_cateNN.m_cate_h2o(seq_cate_short_input)
				
				### retrieve top k predicted categories
				prob_cate_short = F.softmax(logit_cate_short, dim=-1)
				cate_prob_beam, cate_id_beam = prob_cate_short.topk(dim=1, k=self.m_beam_size)

				item_prob_flag = False
				
				for beam_index in range(self.m_beam_size):
					
					### for each category, predict item, then mix predictions for rec

					cate_id_beam_batch = cate_id_beam[:, beam_index]
					cate_id_beam_batch = cate_id_beam_batch.reshape(-1, 1)

					seq_cate_input, seq_short_input = self.model.m_itemNN(x_long_cate_action_batch, x_long_cate_batch, mask_long_cate_action_batch, mask_long_cate_batch, max_long_cate_actionNum_batch, max_long_cateNum_batch, pad_x_long_cate_actionNum_batch, x_long_cateNum_batch, x_short_action_batch, mask_short_action_batch, pad_x_short_actionNum_batch, cate_id_beam_batch, "test")

					mixture_output = torch.cat((seq_cate_input, seq_short_input), dim=1)
					output_batch = self.model.fc(mixture_output)

					### sampled_logit_batch
					sampled_logit_batch, sampled_target_batch = self.model.m_ss(output_batch, y_action_batch, sample_ids, true_freq, sample_freq, acc_hits, self.device, self.m_remove_match)
					
					### batch_size*voc_size
					prob_batch = F.softmax(sampled_logit_batch, dim=-1)

					## batch_size*1
					cate_prob_batch = cate_prob_beam[:, beam_index]
					
					item_prob_batch = prob_batch*cate_prob_batch.reshape(-1, 1)

					# if not item_prob:
					if not item_prob_flag:
						item_prob_flag = True
						item_prob = item_prob_batch
					else:
						item_prob += item_prob_batch

				### evaluate cate prediction
				cate_loss_batch = self.m_loss_func(logit_cate_short, y_cate_batch, "logit")
				cate_losses.append(cate_loss_batch.item())
				cate_topk = 5
				
				cate_recall_batch, cate_mrr_batch = evaluate(logit_cate_short, y_cate_batch, warm_start_mask, k=cate_topk)

				cate_weights.append(int( warm_start_mask.int().sum() ))
				cate_recalls.append(cate_recall_batch)
				cate_mrrs.append(cate_mrr_batch)

				### evaluate item prediction
				recall_batch, mrr_batch = evaluate(item_prob, y_action_batch, warm_start_mask, k=self.topk)

				weights.append( int( warm_start_mask.int().sum() ) )
				recalls.append(recall_batch)
				mrrs.append(mrr_batch)

				total_test_num.append(y_action_batch.view(-1).size(0))

		mean_mixture_losses = 0.0

		mean_losses = 0.0
		mean_recall = np.average(recalls, weights=weights)
		mean_mrr = np.average(mrrs, weights=weights)

		mean_cate_losses = np.mean(cate_losses)
		mean_cate_recall = np.average(cate_recalls, weights=cate_weights)
		mean_cate_mrr = np.average(cate_mrrs, weights=cate_weights)

		msg = "total_test_num"+str(np.sum(total_test_num))
		self.m_log.addOutput2IO(msg)

		return mean_mixture_losses, mean_losses, mean_recall, mean_mrr, mean_cate_losses, mean_cate_recall, mean_cate_mrr
