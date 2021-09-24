import pandas as pd
import numpy as np
import torch
import datetime
import pickle
import random
# import sys

class Dataset(object):

	def __init__(self, action_file, cate_file, observed_threshold, window_size, itemmap=None):
		action_f = open(action_file, "rb")

		self.m_itemmap = {}
		self.m_catemap = {}

		action_seq_arr_total = pickle.load(action_f)

		cate_f = open(cate_file, "rb")
		cate_seq_arr_total = pickle.load(cate_f)

		seq_num = len(action_seq_arr_total)
		print("seq num", seq_num)

		# self.m_seq_list = []

		### each user's sequence is composed of multiple sub sequences
		### sub sequence is composed of actions	
		
		self.m_input_seq_list = []
		self.m_input_seqLen_list = []
		# self.m_input_subseqNum_seq_list = []

		self.m_input_subseq_list_cate_list = []
		self.m_input_subseqLen_list_cate_list = []
		self.m_input_subseqNum_seq_cate_list = []

		self.m_target_action_seq_list = []
		self.m_target_cate_seq_list = []

		self.m_input_seq_idx_list = []

		print("loading item map")

		print("finish loading item map")
		print("observed_threshold", observed_threshold, window_size)
		print("loading data")
		# seq_num = 1
		for seq_index in range(seq_num):
			# print("*"*10, "seq index", seq_index, "*"*10)
			action_seq_arr = action_seq_arr_total[seq_index]
			cate_seq_arr = cate_seq_arr_total[seq_index]

			actionNum_seq = len(action_seq_arr)

			if actionNum_seq < window_size :
				window_size = actionNum_seq

			cate_action_list_map_user = {}
			for action_index in range(actionNum_seq):
				item_cur = action_seq_arr[action_index]
				if item_cur not in self.m_itemmap:
					item_id_cur = len(self.m_itemmap)
					self.m_itemmap[item_cur] = item_id_cur

				subseq_num = 0
				action_list_sub_seq = []
				actionNum_list_sub_seq = []
				if action_index < observed_threshold:

					cate_cur = cate_seq_arr[action_index]
					if cate_cur not in self.m_catemap:
						cate_id_cur = len(self.m_catemap)
						self.m_catemap[cate_cur] = cate_id_cur

					if cate_cur not in cate_action_list_map_user:
						cate_action_list_map_user[cate_cur] = []

					cate_action_list_map_user[cate_cur].append(item_cur)

					continue

				if action_index <= window_size:
					subseq = action_seq_arr[:action_index]
					
					self.m_input_seq_list.append(subseq)
					self.m_input_seqLen_list.append(action_index)

					# subseq_num += 1
					# print("cate action map", cate_action_list_map_user)
					for cate in cate_action_list_map_user:
						subseq_cate = cate_action_list_map_user[cate].copy()[-window_size:]
						actionNum_subseq_cate = len(subseq_cate)

						action_list_sub_seq.append(subseq_cate)
						actionNum_list_sub_seq.append(actionNum_subseq_cate)
						
						subseq_num += 1

					self.m_input_subseq_list_cate_list.append(action_list_sub_seq)
					self.m_input_subseqLen_list_cate_list.append(actionNum_list_sub_seq)
					self.m_input_subseqNum_seq_cate_list.append(subseq_num)
					
					target_subseq = action_seq_arr[action_index]
					self.m_target_action_seq_list.append(target_subseq)

					self.m_input_seq_idx_list.append(action_index)

				if action_index > window_size:
					subseq = action_seq_arr[action_index-window_size:action_index]
					# action_list_sub_seq.append(subseq)
					# actionNum_list_sub_seq.append(window_size)

					self.m_input_seq_list.append(subseq)
					self.m_input_seqLen_list.append(window_size)

					# print("cate action map", cate_action_list_map_user)
					for cate in cate_action_list_map_user:
						subseq_cate = cate_action_list_map_user[cate].copy()[-window_size:]
						actionNum_subseq_cate = len(subseq_cate)

						action_list_sub_seq.append(subseq_cate)
						actionNum_list_sub_seq.append(actionNum_subseq_cate)
					
						subseq_num += 1

					# print("++++++++action_list_sub_seq", action_list_sub_seq)
					self.m_input_subseq_list_cate_list.append(action_list_sub_seq)
					self.m_input_subseqLen_list_cate_list.append(actionNum_list_sub_seq)
					self.m_input_subseqNum_seq_cate_list.append(subseq_num)

					target_subseq = action_seq_arr[action_index]
					self.m_target_action_seq_list.append(target_subseq)

					self.m_input_seq_idx_list.append(action_index)
		
				cate_cur = cate_seq_arr[action_index]
				if cate_cur not in self.m_catemap:
					cate_id_cur = len(self.m_catemap)
					self.m_catemap[cate_cur] = cate_id_cur

				if cate_cur not in cate_action_list_map_user:
					cate_action_list_map_user[cate_cur] = []

				cate_action_list_map_user[cate_cur].append(item_cur)

		# print("debug", self.m_input_subseq_list_seq_list[:10])
		print("subseq num", len(self.m_input_seq_list))
		print("subseq len num", len(self.m_input_seqLen_list))
		print("seq idx num", len(self.m_input_seq_idx_list))

	@property
	def items(self):
		# print("first item", self.m_itemmap['<PAD>'])
		return self.m_itemmap

class DataLoader():
	def __init__(self, dataset, batch_size):
		self.m_dataset = dataset
		self.m_batch_size = batch_size

		self.m_itemFreq_map = dataset.m_itemFreq_map
		self.m_alpha = 1.0
		# negative_sample_num = 1000

		"""
		sort subsequences 
		"""

		sorted_data = sorted(zip(self.m_dataset.m_input_subseqNum_seq_cate_list, self.m_dataset.m_input_subseq_list_cate_list, self.m_dataset.m_input_subseqLen_list_cate_list, self.m_dataset.m_input_cate_subseq_list_cate_list, self.m_dataset.m_input_seq_list, self.m_dataset.m_input_cate_seq_list, self.m_dataset.m_input_seqLen_list , self.m_dataset.m_target_action_seq_list, self.m_dataset.m_target_cate_seq_list, self.m_dataset.m_input_seq_idx_list), reverse=True)
		
		self.m_dataset.m_input_subseqNum_seq_cate_list, self.m_dataset.m_input_subseq_list_cate_list, self.m_dataset.m_input_subseqLen_list_cate_list, self.m_dataset.m_input_cate_subseq_list_cate_list, self.m_dataset.m_input_seq_list, self.m_dataset.m_input_cate_seq_list,self.m_dataset.m_input_seqLen_list, self.m_dataset.m_target_action_seq_list, self.m_dataset.m_target_cate_seq_list, self.m_dataset.m_input_seq_idx_list = zip(*sorted_data)

		input_seq_num = len(self.m_dataset.m_input_subseqNum_seq_cate_list)
		batch_num = int(input_seq_num/batch_size)

		print("batch_num", batch_num)
		# print("subseq num", self.m_dataset.m_input_subseqNum_seq_cate_list[:1000])

		input_subseqNum_seq_cate_list_batch = [self.m_dataset.m_input_subseqNum_seq_cate_list[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]
		input_subseq_list_cate_list_batch = [self.m_dataset.m_input_subseq_list_cate_list[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]
		input_subseqLen_list_cate_list_batch = [self.m_dataset.m_input_subseqLen_list_cate_list[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]

		input_cate_subseq_list_batch = [self.m_dataset.m_input_cate_subseq_list_cate_list[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]

		input_seq_list_batch = [self.m_dataset.m_input_seq_list[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]
		input_cate_seq_list_batch = [self.m_dataset.m_input_cate_seq_list[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]

		input_seqLen_list_batch = [self.m_dataset.m_input_seqLen_list[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]
		target_action_seq_list_batch = [self.m_dataset.m_target_action_seq_list[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]
		target_cate_seq_list_batch = [self.m_dataset.m_target_cate_seq_list[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]

		input_seq_idx_list_batch = [self.m_dataset.m_input_seq_idx_list[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]

		# print("input_subseqNum_seq_cate_list_batch", input_subseqNum_seq_cate_list_batch[:100])

		temp = list(zip(input_subseqNum_seq_cate_list_batch, input_subseq_list_cate_list_batch, input_subseqLen_list_cate_list_batch, input_cate_subseq_list_batch, input_seq_list_batch, input_cate_seq_list_batch, input_seqLen_list_batch, target_action_seq_list_batch, target_cate_seq_list_batch, input_seq_idx_list_batch))

		self.m_temp = temp
		
	def __iter__(self):
		print("shuffling")
		temp = self.m_temp
		random.shuffle(temp)

		input_subseqNum_seq_cate_list_batch, input_subseq_list_cate_list_batch, input_subseqLen_list_cate_list_batch, input_cate_subseq_list_batch, input_seq_list_batch, input_cate_seq_list_batch, input_seqLen_list_batch, target_action_seq_list_batch, target_cate_seq_list_batch, input_seq_idx_list_batch = zip(*temp)

		batch_size = self.m_batch_size
		
		batch_num = len(input_subseqNum_seq_cate_list_batch)

		for batch_index in range(batch_num):
			st = datetime.datetime.now()
			input_subseqNum_seq_cate_list_subbatch = input_subseqNum_seq_cate_list_batch[batch_index]
			input_subseq_list_seq_cate_list_subbatch = input_subseq_list_cate_list_batch[batch_index]
			input_subseqLen_list_cate_list_subbatch = input_subseqLen_list_cate_list_batch[batch_index]
			input_cate_subseq_list_subbatch = input_cate_subseq_list_batch[batch_index]

			input_cate_seq_list_subbatch = input_cate_seq_list_batch[batch_index]
			input_seq_list_subbatch = input_seq_list_batch[batch_index]
			input_seqLen_list_subbatch = input_seqLen_list_batch[batch_index]
			target_action_seq_list_subbatch = target_action_seq_list_batch[batch_index]
			target_cate_seq_list_subbatch = target_cate_seq_list_batch[batch_index]

			input_seq_idx_list_subbatch = input_seq_idx_list_batch[batch_index]
			
			if batch_index %4000 == 0:
				print("batch index", batch_index)

			x_cate_batch = []

			y_batch = []

			idx_batch = []

			max_actionNum_cate_batch = 0
			max_subseqNum_cate_batch = 0

			subseqNum_cate_batch = []
			actionNum_cate_batch = []
			
			for seq_index_batch in range(batch_size):
				seq_index = seq_index_batch

				subseqNum_cate_user = input_subseqNum_seq_cate_list_subbatch[seq_index]
				subseqlen_list_cate_user = input_subseqLen_list_cate_list_subbatch[seq_index]

				subseqNum_cate_batch.append(subseqNum_cate_user)
				max_actionNum_seq_cate = max(subseqlen_list_cate_user)

				actionNum_cate_batch.append(max_actionNum_seq_cate)

			max_actionNum_cate_batch = max(actionNum_cate_batch)
			max_subseqNum_cate_batch = max(subseqNum_cate_batch)

			mask_cate_batch = None
			subseqLen_cate_batch = []
			seqLen_cate_batch = []
			mask_cate_seq_batch = None

			# x_subseq_index_batch = []

			input_cate_subseq_batch = []

			for seq_index_batch in range(batch_size):
				seq_index = seq_index_batch

				subseq_list_cate_user = input_subseq_list_seq_cate_list_subbatch[seq_index]

				pad_subseq_list_cate_user = [subseq + [0]*(max_actionNum_cate_batch-len(subseq)) for subseq in subseq_list_cate_user]
				# print("pad_subseq_list_user", pad_subseq_list_user)
				pad_subseq_list_cate_user += [[0]*max_actionNum_cate_batch for i in range(max_subseqNum_cate_batch-len(pad_subseq_list_cate_user))]	
				y = target_action_seq_list_subbatch[seq_index]

				subseqNum_cate_user = input_subseqNum_seq_cate_list_subbatch[seq_index]

				subseqLen_list_cate_user = input_subseqLen_list_cate_list_subbatch[seq_index]
			
				subseqLen_cate_batch += subseqLen_list_cate_user
			
				subseqLen_cate_batch += [0]*(max_subseqNum_cate_batch-subseqNum_cate_user)

				seqLen_cate_batch.append(subseqNum_cate_user)
				
				x_cate_batch += pad_subseq_list_cate_user
			
				y_batch.append(y)
				idx_batch.append(input_seq_idx_list_subbatch[seq_index])

				pad_cate_subseq_batch = input_cate_subseq_list_subbatch[seq_index]
				pad_cate_subseq_batch = pad_cate_subseq_batch+[0]*(max_subseqNum_cate_batch-subseqNum_cate_user)
				pad_cate_subseq_batch = np.array(pad_cate_subseq_batch)
				input_cate_subseq_batch.append(pad_cate_subseq_batch)

			subseqLen_cate_batch = np.array(subseqLen_cate_batch)
			seqLen_cate_batch = np.array(seqLen_cate_batch)

			mask_cate_batch = np.arange(max_actionNum_cate_batch)[None, :] < subseqLen_cate_batch[:, None]
			mask_cate_seq_batch = np.arange(max_subseqNum_cate_batch)[None,:] < seqLen_cate_batch[:, None]

			x_cate_batch = np.array(x_cate_batch)

			x_batch = []
			mask_batch = []
			seqLen_batch = []

			cate_batch = []
			
			target_cate_batch = []

			for seq_index_batch in range(batch_size):
				seq_index = seq_index_batch

				seqLen_user = input_seqLen_list_subbatch[seq_index]
				
				seqLen_batch.append(seqLen_user)

			max_seqLen_batch = max(seqLen_batch)

			for seq_index_batch in range(batch_size):
				seq_index = seq_index_batch

				seq_user = input_seq_list_subbatch[seq_index]

				pad_seq_user = seq_user+[0]*(max_seqLen_batch-len(seq_user))
			
				x_batch.append(pad_seq_user)

				cate_user = input_cate_seq_list_subbatch[seq_index]

				pad_cate_user = cate_user+[0]*(max_seqLen_batch-len(seq_user))

				cate_batch.append(pad_cate_user)

				target_cate_batch.append(target_cate_seq_list_subbatch[seq_index])
			
			seqLen_batch = np.array(seqLen_batch)
			# print("seqLen_batch", seqLen_batch)
			mask_batch = np.arange(max_seqLen_batch)[None,:] < seqLen_batch[:, None]
			
			x_batch = np.array(x_batch)

			y_batch = np.array(y_batch)
			idx_batch = np.array(idx_batch)
			cate_batch = np.array(cate_batch)
			target_cate_batch = np.array(target_cate_batch)
			input_cate_subseq_batch = np.array(input_cate_subseq_batch)
			# y_neg_batch = self.generateNegSample(y_batch, pop, self.m_negative_num)

			x_batch_tensor = torch.from_numpy(x_batch)
			mask_batch_tensor = torch.from_numpy(mask_batch*1).float()

			cate_batch_tensor = torch.from_numpy(cate_batch)

			x_cate_batch_tensor = torch.from_numpy(x_cate_batch)
			y_batch_tensor = torch.from_numpy(y_batch)

			target_cate_tensor = torch.from_numpy(target_cate_batch)

			input_cate_subseq_batch = torch.from_numpy(input_cate_subseq_batch)
			
			# y_neg_batch_tensor = torch.from_numpy(y_neg_batch)

			mask_cate_batch_tensor = torch.from_numpy(mask_cate_batch*1).float()
			mask_cate_seq_batch_tensor = torch.from_numpy(mask_cate_seq_batch*1).float()

			idx_batch_tensor = torch.from_numpy(idx_batch)

			pad_subseqLen_cate_batch = np.array([i-1 if i > 0 else 0 for i in subseqLen_cate_batch])
			pad_seqLen_batch = np.array([i-1 if i > 0 else 0 for i in seqLen_batch])

			et = datetime.datetime.now()
			print("batch duration", et-st)

			yield x_cate_batch_tensor, input_cate_subseq_batch, mask_cate_batch_tensor, mask_cate_seq_batch_tensor, max_actionNum_cate_batch, max_subseqNum_cate_batch, pad_subseqLen_cate_batch, seqLen_cate_batch, x_batch_tensor, cate_batch_tensor, mask_batch_tensor, pad_seqLen_batch, y_batch_tensor, target_cate_tensor, idx_batch_tensor

	def generateNegSample(self, y_batch, pop, negative_sample_num):
		neg_sample_batch = np.searchsorted(pop, np.random.rand(negative_sample_num*y_batch.shape[0]))

		neg_sample_batch = neg_sample_batch.reshape(-1, negative_sample_num)

		return neg_sample_batch