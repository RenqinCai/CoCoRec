"""
clean code
"""
import pandas as pd
import numpy as np
import torch
import datetime
import pickle
import random
# import sys
import multiprocessing

class MYDATA(object):

	def __init__(self, action_file, cate_file, time_file, valid_start_time, test_start_time, observed_thresh, window_size, cate_window_size):
		action_f = open(action_file, "rb")
		action_total = pickle.load(action_f)
		action_seq_num = len(action_total)
		print("action seq num", action_seq_num)

		self.m_itemmap = {}
		self.m_catemap = {}

		self.m_itemmap['<PAD>'] = 0
		self.m_catemap['<PAD>'] = 0

		cate_f = open(cate_file, "rb")
		cate_total = pickle.load(cate_f)
		cate_seq_num = len(cate_total)
		print("cate seq num", cate_seq_num)

		time_f = open(time_file, "rb")
		time_total = pickle.load(time_f)
		time_seq_num = len(time_total)
		print("time seq num", time_seq_num)

		### train
		## items of short-range actions
		self.m_x_short_action_list_train = []

		### num of short-range actions
		self.m_x_short_actionNum_list_train = []

		### cate of short-range actions
		self.m_x_short_cate_list_train = []

		### items of long-range actions of category
		self.m_x_long_cate_action_list_train = []

		### cates of long-range actions
		self.m_x_long_cate_list_train = []

		### num of long-range actions of category
		self.m_x_long_cate_actionNum_list_train = []

		### num of category of long-range actions
		self.m_x_long_cateNum_list_train = []

		### items of target actions
		self.m_y_action_train = []

		### cates of target actions
		self.m_y_cate_train = []

		### idx of target actions in a sequence
		self.m_y_action_idx_train = []

		### test
		self.m_x_action_list_test = []
		self.m_x_short_action_list_test = []
		self.m_x_short_actionNum_list_test = []
		self.m_x_short_cate_list_test = []

		self.m_x_long_cate_action_list_test = []
		self.m_x_long_cate_list_test = []

		self.m_x_long_cate_actionNum_list_test = []
		self.m_x_long_cateNum_list_test = []

		self.m_y_action_test = []
		self.m_y_cate_test = []

		self.m_y_action_idx_test = []

		print("loading item map")

		print("loading item map")
		print("observed_threshold", observed_thresh, window_size)
		print("loading data")

		print("valid_start_time", valid_start_time)
		print("test start time", test_start_time)

		for seq_index in range(action_seq_num):
			# print("*"*10, "seq index", seq_index, "*"*10)
			action_seq_arr = action_total[seq_index]
			cate_seq_arr = cate_total[seq_index]
			time_seq_arr = time_total[seq_index]

			actionNum_seq = len(action_seq_arr)
			actionNum_seq_cate = len(cate_seq_arr)
			actionNum_seq_time = len(time_seq_arr)

			if actionNum_seq != actionNum_seq_cate:
				assert("!= seq len action cate")
			
			if actionNum_seq_cate != actionNum_seq_time:
				assert("!= seq len cate time")

			if actionNum_seq < window_size :
				window_size = actionNum_seq
			
			###{cate:[action list]}
			cate_action_list_map_user = {}

			for action_index in range(actionNum_seq):
				item_cur = action_seq_arr[action_index]
				cate_cur = cate_seq_arr[action_index]
				time_cur = time_seq_arr[action_index]

				if time_cur <= valid_start_time:
					if item_cur not in self.m_itemmap:
						self.m_itemmap[item_cur] = len(self.m_itemmap)

				if action_index < observed_thresh:
					if cate_cur not in self.m_catemap:
						cate_id_cur = len(self.m_catemap)
						self.m_catemap[cate_cur] = cate_id_cur

					if cate_cur not in cate_action_list_map_user:
						cate_action_list_map_user[cate_cur] = []
					
					cate_action_list_map_user[cate_cur].append(item_cur)
				
					continue

				if time_cur <= valid_start_time:
					self.addItem2train(action_index, window_size, cate_window_size, cate_cur, action_seq_arr, cate_seq_arr, cate_action_list_map_user)

				if time_cur > valid_start_time:
					if time_cur <= test_start_time:
						self.addItem2test(action_index, window_size, cate_window_size, cate_cur, action_seq_arr, cate_seq_arr, cate_action_list_map_user)

				cate_cur = cate_seq_arr[action_index]
				if cate_cur not in self.m_catemap:
					cate_id_cur = len(self.m_catemap)
					self.m_catemap[cate_cur] = cate_id_cur

				if cate_cur not in cate_action_list_map_user:
					cate_action_list_map_user[cate_cur] = []

				cate_action_list_map_user[cate_cur].append(item_cur)

		print("seq num for training", len(self.m_x_short_action_list_train))
		print("seq num of actions for training", len(self.m_x_short_actionNum_list_train))

		print("seq num for testing", len(self.m_x_short_action_list_test))
		print("seq num of actions for testing", len(self.m_x_short_actionNum_list_test))

		user_num = len(self.m_x_long_cate_list_train)
		user_cate_num = 0
		total_user_cate_num = 0
		min_user_cate_num = 100
		for user_long_cate_list in self.m_x_long_cate_list_train:
			user_cate_num = max(user_cate_num, len(user_long_cate_list))
			total_user_cate_num += len(user_long_cate_list)
			min_user_cate_num= min(user_cate_num, len(user_long_cate_list))

		print("user_cate_num maximum", user_cate_num)
		print("min_user_cate_num", min_user_cate_num)
		print("avg cate num per user", total_user_cate_num/user_num)

		self.train_dataset = MYDATASET([], self.m_x_long_cate_action_list_train, self.m_x_long_cate_actionNum_list_train, self.m_x_long_cateNum_list_train, self.m_x_long_cate_list_train, self.m_x_short_action_list_train, self.m_x_short_cate_list_train, self.m_x_short_actionNum_list_train, self.m_y_action_train, self.m_y_cate_train, self.m_y_action_idx_train)

		self.test_dataset = MYDATASET([], self.m_x_long_cate_action_list_test, self.m_x_long_cate_actionNum_list_test, self.m_x_long_cateNum_list_test, self.m_x_long_cate_list_test, self.m_x_short_action_list_test, self.m_x_short_cate_list_test, self.m_x_short_actionNum_list_test, self.m_y_action_test, self.m_y_cate_test, self.m_y_action_idx_test)

	def addItem2train(self, action_index, window_size, cate_window_size, cate_cur, action_seq_arr, cate_seq_arr, cate_action_list_map_user):
		
		short_seq = None
		short_cate_seq = None

		if action_index <= window_size:
			short_seq = action_seq_arr[:action_index]
			short_cate_seq = cate_seq_arr[:action_index]
		else:
			short_seq = action_seq_arr[action_index-window_size: action_index]
			short_cate_seq = cate_seq_arr[action_index-window_size: action_index]

		self.m_x_short_action_list_train.append(short_seq)
		self.m_x_short_cate_list_train.append(short_cate_seq)

		short_actionNum_seq = len(short_seq)
		self.m_x_short_actionNum_list_train.append(short_actionNum_seq)

		long_cate_num = 0
		long_cate_action_list = []
		long_cate_actionNum_list = []
		long_cate_list = []

		for cate in cate_action_list_map_user:
			long_cate_subseq = cate_action_list_map_user[cate].copy()[-cate_window_size:]
			long_cate_actionNum_subseq = len(long_cate_subseq)

			long_cate_action_list.append(long_cate_subseq)
			long_cate_actionNum_list.append(long_cate_actionNum_subseq)

			long_cate_num += 1

			long_cate_list.append(cate)

		# self.m_x_long_cate_action_list_train.append(long_cate_action_list)
		self.m_x_long_cate_list_train.append(long_cate_list)
		self.m_y_cate_train.append(cate_cur)

		self.m_x_long_cate_action_list_train.append(long_cate_action_list)
		self.m_x_long_cate_actionNum_list_train.append(long_cate_actionNum_list)
		self.m_x_long_cateNum_list_train.append(long_cate_num)

		y_action = action_seq_arr[action_index]
		self.m_y_action_train.append(y_action)
		self.m_y_action_idx_train.append(action_index)

	def addItem2test(self, action_index, window_size, cate_window_size, cate_cur, action_seq_arr, cate_seq_arr, cate_action_list_map_user):
		short_seq = None
		short_cate_seq = None

		if action_index <= window_size:
			short_seq = action_seq_arr[:action_index]
			short_cate_seq = cate_seq_arr[:action_index]
		else:
			short_seq = action_seq_arr[action_index-window_size: action_index]
			short_cate_seq = cate_seq_arr[action_index-window_size: action_index]

		self.m_x_short_action_list_test.append(short_seq)
		self.m_x_short_cate_list_test.append(short_cate_seq)

		short_actionNum_seq = len(short_seq)
		self.m_x_short_actionNum_list_test.append(short_actionNum_seq)

		long_cate_num = 0
		long_cate_action_list = []
		long_cate_actionNum_list = []
		long_cate_list = []

		for cate in cate_action_list_map_user:
			long_cate_subseq = cate_action_list_map_user[cate].copy()[-cate_window_size:]
			long_cate_actionNum_subseq = len(long_cate_subseq)

			long_cate_action_list.append(long_cate_subseq)
			long_cate_actionNum_list.append(long_cate_actionNum_subseq)

			long_cate_num += 1

			long_cate_list.append(cate)

		# self.m_x_long_cate_action_list_train.append(long_cate_action_list)
		self.m_x_long_cate_list_test.append(long_cate_list)
		self.m_y_cate_test.append(cate_cur)

		self.m_x_long_cate_action_list_test.append(long_cate_action_list)
		self.m_x_long_cate_actionNum_list_test.append(long_cate_actionNum_list)
		self.m_x_long_cateNum_list_test.append(long_cate_num)

		y_action = action_seq_arr[action_index]
		self.m_y_action_test.append(y_action)
		self.m_y_action_idx_test.append(action_index)
		
		# self.m_x_action_list_test.append(action_seq_arr[:action_index])

	def items(self):
		print("item num", len(self.m_itemmap))
		return len(self.m_itemmap)

	def cates(self):
		print("cate num", len(self.m_catemap))
		return len(self.m_catemap)

class MYDATASET(object):
	def __init__(self, x_action_list, x_long_cate_action_list, x_long_cate_actionNum_list, x_long_cateNum_list, x_long_cate_list, x_short_action_list, x_short_cate_list, x_short_actionNum_list, y_action, y_cate, y_action_idx):
		self.m_x_long_cate_action_list = x_long_cate_action_list
		self.m_x_long_cate_actionNum_list = x_long_cate_actionNum_list
		
		self.m_x_long_cateNum_list = x_long_cateNum_list
		self.m_x_long_cate_list = x_long_cate_list

		self.m_x_short_action_list = x_short_action_list
		self.m_x_short_cate_list = x_short_cate_list
		self.m_x_short_actionNum_list = x_short_actionNum_list

		self.m_y_action = y_action
		self.m_y_cate = y_cate

		self.m_y_action_idx = y_action_idx
		self.m_x_action_list = x_action_list

class MYDATALOADER(object):
	def __init__(self, dataset, batch_size):
		self.m_dataset = dataset
		self.m_batch_size = batch_size

		print("len", len(self.m_dataset.m_x_long_cateNum_list))

		### sort data according to the number of categories of long-range actions
		sorted_data = sorted(zip(self.m_dataset.m_x_long_cateNum_list, self.m_dataset.m_x_long_cate_action_list, self.m_dataset.m_x_long_cate_actionNum_list, self.m_dataset.m_x_long_cate_list, self.m_dataset.m_x_short_action_list, self.m_dataset.m_x_short_cate_list, self.m_dataset.m_x_short_actionNum_list , self.m_dataset.m_y_action, self.m_dataset.m_y_cate, self.m_dataset.m_y_action_idx), reverse=True)

		# sorted_data = sorted(zip(self.m_dataset.m_x_long_cateNum_list, self.m_dataset.m_x_long_cate_action_list, self.m_dataset.m_x_long_cate_actionNum_list, self.m_dataset.m_x_long_cate_list, self.m_dataset.m_x_short_action_list, self.m_dataset.m_x_short_cate_list, self.m_dataset.m_x_short_actionNum_list , self.m_dataset.m_y_action, self.m_dataset.m_y_cate, self.m_dataset.m_y_action_idx, self.m_dataset.m_x_action_list), reverse=True)

		self.m_dataset.m_x_long_cateNum_list, self.m_dataset.m_x_long_cate_action_list, self.m_dataset.m_x_long_cate_actionNum_list, self.m_dataset.m_x_long_cate_list, self.m_dataset.m_x_short_action_list, self.m_dataset.m_x_short_cate_list, self.m_dataset.m_x_short_actionNum_list , self.m_dataset.m_y_action, self.m_dataset.m_y_cate, self.m_dataset.m_y_action_idx = zip(*sorted_data)

		input_seq_num = len(self.m_dataset.m_x_long_cateNum_list)
		batch_num = int(input_seq_num/batch_size)
		print("seq num", input_seq_num)
		print("batch size", self.m_batch_size)
		print("batch_num", batch_num)

		### batchify data: sequences which have close number of categories are put in a batch
		x_long_cateNum_list = [self.m_dataset.m_x_long_cateNum_list[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]
		x_long_cate_action_list = [self.m_dataset.m_x_long_cate_action_list[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]
		x_long_cate_actionNum_list = [self.m_dataset.m_x_long_cate_actionNum_list[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]

		x_long_cate_list = [self.m_dataset.m_x_long_cate_list[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]

		x_short_action_list = [self.m_dataset.m_x_short_action_list[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]
		x_short_cate_list = [self.m_dataset.m_x_short_cate_list[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]

		x_short_actionNum_list = [self.m_dataset.m_x_short_actionNum_list[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]
		y_action = [self.m_dataset.m_y_action[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]
		y_cate = [self.m_dataset.m_y_cate[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]

		y_action_idx = [self.m_dataset.m_y_action_idx[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]

		# x_action = [self.m_dataset.m_x_action_list[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]

		temp = list(zip(x_long_cateNum_list, x_long_cate_action_list, x_long_cate_actionNum_list, x_long_cate_list, x_short_action_list, x_short_cate_list, x_short_actionNum_list, y_action, y_cate, y_action_idx))

		self.m_temp = temp

	def __iter__(self):
		print("shuffling")

		temp = self.m_temp
		random.shuffle(temp)

		x_long_cateNum_list, x_long_cate_action_list, x_long_cate_actionNum_list, x_long_cate_list, x_short_action_list, x_short_cate_list, x_short_actionNum_list, y_action, y_cate, y_action_idx = zip(*temp)

		batch_size = self.m_batch_size
		
		batch_num = len(x_long_cateNum_list)

		for batch_index in range(batch_num):
		   
			x_long_cate_action_list_batch = x_long_cate_action_list[batch_index]
			x_long_cate_actionNum_list_batch = x_long_cate_actionNum_list[batch_index]
			x_long_cate_list_batch = x_long_cate_list[batch_index]
			x_long_cateNum_list_batch = x_long_cateNum_list[batch_index]

			x_short_action_list_batch = x_short_action_list[batch_index]
			x_short_cate_list_batch = x_short_cate_list[batch_index]
			x_short_actionNum_list_batch = x_short_actionNum_list[batch_index]

			y_action_batch = y_action[batch_index]
			y_cate_batch = y_cate[batch_index]
			y_action_idx_batch = y_action_idx[batch_index]

			if batch_index %4000 == 0:
				print("batch index", batch_index)

			x_long_cate_action_batch = []
			x_long_cate_actionNum_batch = []

			x_long_cate_batch = []
			x_long_cateNum_batch = []

			max_long_cate_actionNum_batch = max(max(i) for i in x_long_cate_actionNum_list_batch)
			max_long_cateNum_batch = max(x_long_cateNum_list_batch)

			mask_long_cate_action_batch = None
			mask_long_cate_batch = None

			for seq_index_batch in range(batch_size):
				x_long_cate_action_seq = x_long_cate_action_list_batch[seq_index_batch]

				pad_x_long_cate_action_seq = [subseq + [0]*(max_long_cate_actionNum_batch-len(subseq)) for subseq in x_long_cate_action_seq]

				pad_x_long_cate_action_seq += [[0]*max_long_cate_actionNum_batch for i in range(max_long_cateNum_batch-len(pad_x_long_cate_action_seq))]

				x_long_cate_action_batch += pad_x_long_cate_action_seq

				x_long_cateNum_seq = x_long_cateNum_list_batch[seq_index_batch]
				x_long_cate_actionNum_seq = x_long_cate_actionNum_list_batch[seq_index_batch]

				x_long_cate_actionNum_batch += x_long_cate_actionNum_seq
				x_long_cate_actionNum_batch += [0]*(max_long_cateNum_batch-x_long_cateNum_seq)

				x_long_cateNum_batch.append(x_long_cateNum_seq)

				x_long_cate_seq = x_long_cate_list_batch[seq_index_batch]
				pad_x_long_cate_seq = x_long_cate_seq+[0]*(max_long_cateNum_batch-x_long_cateNum_seq)

				x_long_cate_batch.append(pad_x_long_cate_seq)

			x_long_cate_actionNum_batch = np.array(x_long_cate_actionNum_batch)
			x_long_cateNum_batch = np.array(x_long_cateNum_batch)

			mask_long_cate_action_batch = np.arange(max_long_cate_actionNum_batch)[None, :] < x_long_cate_actionNum_batch[:, None]
			mask_long_cate_batch = np.arange(max_long_cateNum_batch)[None, :] < x_long_cateNum_batch[:, None]

			x_long_cate_action_batch = np.array(x_long_cate_action_batch)
			x_long_cate_batch = np.array(x_long_cate_batch)

			x_short_action_batch = []
			x_short_cate_batch = []
			x_short_actionNum_batch = []

			mask_short_action_batch = []

			max_short_actionNum_batch = max(x_short_actionNum_list_batch)
			
			for seq_index_batch in range(batch_size):
				x_short_actionNum_seq = x_short_actionNum_list_batch[seq_index_batch]
				
				pad_x_short_action_seq = x_short_action_list_batch[seq_index_batch]+[0]*(max_short_actionNum_batch-x_short_actionNum_seq)
				x_short_action_batch.append(pad_x_short_action_seq)

				pad_x_short_cate_seq = x_short_cate_list_batch[seq_index_batch]+[0]*(max_short_actionNum_batch-x_short_actionNum_seq)
				x_short_cate_batch.append(pad_x_short_cate_seq)

			x_short_action_batch = np.array(x_short_action_batch)
			x_short_cate_batch = np.array(x_short_cate_batch)

			x_short_actionNum_batch = np.array(x_short_actionNum_list_batch)
			mask_short_action_batch = np.arange(max_short_actionNum_batch)[None, :] < x_short_actionNum_batch[:, None]

			y_action_batch = np.array(y_action_batch)
			y_cate_batch = np.array(y_cate_batch)
			y_action_idx_batch = np.array(y_action_idx_batch)

			x_long_cate_action_batch_tensor = torch.from_numpy(x_long_cate_action_batch)
			mask_long_cate_action_batch_tensor = torch.from_numpy(mask_long_cate_action_batch*1).float()

			x_long_cate_batch_tensor = torch.from_numpy(x_long_cate_batch)
			mask_long_cate_batch_tensor = torch.from_numpy(mask_long_cate_batch*1).float()
			
			x_short_action_batch_tensor = torch.from_numpy(x_short_action_batch)
			x_short_cate_batch_tensor = torch.from_numpy(x_short_cate_batch)
			mask_short_action_batch_tensor = torch.from_numpy(mask_short_action_batch*1).float()

			y_action_batch_tensor = torch.from_numpy(y_action_batch)
			y_cate_batch_tensor = torch.from_numpy(y_cate_batch)

			y_action_idx_batch_tensor = torch.from_numpy(y_action_idx_batch)

			### prepare padding used in the model.py to save time of creating it each iteration
			pad_x_long_cate_actionNum_batch = np.array([i-1 if i > 0 else 0 for i in x_long_cate_actionNum_batch])
			pad_x_short_actionNum_batch = np.array([i-1 if i > 0 else 0 for i in x_short_actionNum_batch])

			yield x_long_cate_action_batch_tensor, x_long_cate_batch_tensor, mask_long_cate_action_batch_tensor, mask_long_cate_batch_tensor, max_long_cate_actionNum_batch, max_long_cateNum_batch, pad_x_long_cate_actionNum_batch, x_long_cateNum_batch, x_short_action_batch_tensor, x_short_cate_batch_tensor, mask_short_action_batch_tensor, pad_x_short_actionNum_batch, y_action_batch_tensor, y_cate_batch_tensor, y_action_idx_batch_tensor