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

        ### action
        self.m_x_short_train = []
        self.m_xnum_short_train = []

        ### category
        self.m_c_short_train = []

        ### action per category
        self.m_x_long_train = []
        self.m_c_long_train = []
        self.m_xnum_long_train = []

        self.m_cnum_long_train = []
        
        self.m_y_train = []
        self.m_z_train = []

        self.m_y_idx_train = []

        #### test
        ### action
        self.m_x_short_test = []
        self.m_xnum_short_test = []

        ### category
        self.m_c_short_test = []

        ### action per category
        self.m_x_long_test = []
        self.m_c_long_test = []
        self.m_xnum_long_test = []

        self.m_cnum_long_test = []
        
        self.m_y_test = []
        self.m_z_test = []

        self.m_y_idx_test = []

        print("xxx"*10, "loading data", "xxx"*10)
        print(" "*5, "valid start time:%d"%valid_start_time, " "*5)
        print(" "*5, "test start time:%d"%test_start_time, " "*5)
        print(" "*5, "window size:%d"%window_size, " "*5)

        for seq_index in range(action_seq_num):
            action_seq = action_total[seq_index]
            cate_seq = cate_total[seq_index]
            time_seq = time_total[seq_index]

            seq_len = len(action_seq)

            if seq_len < window_size:
                window_size = seq_len

            ### {cate: [action list]}
            c_x_user = {}

            for action_index in range(seq_len):
                item_cur = action_seq[action_index]
                cate_cur = cate_seq[action_index]
                time_cur = time_seq[action_index]

                if time_cur <= valid_start_time:
                    if item_cur not in self.m_itemmap:
                        self.m_itemmap[item_cur] = len(self.m_itemmap)
                    
                if action_index < observed_thresh:
                    if cate_cur not in self.m_catemap:
                        cate_id_cur = len(self.m_catemap)
                        self.m_catemap[cate_cur] = cate_id_cur

                    if cate_cur not in c_x_user:
                        c_x_user[cate_cur] = []

                    c_x_user[cate_cur].append(item_cur)

                    continue

                if time_cur <= valid_start_time:
                    self.add_item2train(action_index, window_size, cate_window_size, cate_cur, action_seq, cate_seq, c_x_user)

                if time_cur > valid_start_time:
                    if time_cur <= test_start_time:
                        self.add_item2test(action_index, window_size, cate_window_size, cate_cur, action_seq, cate_seq, c_x_user)

                cate_cur = cate_seq[action_index]
                if cate_cur not in self.m_catemap:
                    cate_id_cur = len(self.m_catemap)
                    self.m_catemap[cate_cur] = cate_id_cur
                
                if cate_cur not in c_x_user:
                    c_x_user[cate_cur] = []

                c_x_user[cate_cur].append(item_cur)
            
        print("seq num for training", len(self.m_x_short_train))
        print("seq num of actions for training", len(self.m_xnum_short_train))

        print("seq num for testing", len(self.m_x_short_test))
        print("seq num of actions for testing", len(self.m_xnum_short_test))

        usernum = len(self.m_x_long_train)
        max_catenum_user = 0
        total_catenum_user = 0
        min_catenum_user = 100

        for c_user in self.m_c_long_train:
            max_catenum_user = max(max_catenum_user, len(c_user))
            total_catenum_user += len(c_user)
            min_catenum_user = min(min_catenum_user, len(c_user))

        print("avg cate num per user", total_catenum_user/usernum)
        print("max_catenum_user", max_catenum_user)
        print("min_catenum_user", min_catenum_user)

        self.m_train_dataset = MYDATASET([], self.m_x_long_train, self.m_xnum_long_train, self.m_cnum_long_train, self.m_c_long_train, self.m_x_short_train, self.m_c_short_train, self.m_xnum_short_train, self.m_y_train, self.m_z_train, self.m_y_idx_train)

        self.m_test_dataset = MYDATASET([], self.m_x_long_test, self.m_xnum_long_test, self.m_cnum_long_test, self.m_c_long_test, self.m_x_short_test, self.m_c_short_test, self.m_xnum_short_test, self.m_y_test, self.m_z_test, self.m_y_idx_test)


    def add_item2train(self, action_index, window_size, cate_window_size, cate_cur, action_seq, cate_seq, c_x_user):
        x_short_seq = None
        c_short_seq = None

        if action_index <= window_size:
            x_short_seq = action_seq[:action_index]
            c_short_seq = cate_seq[:action_index]
        else:
            x_short_seq = action_seq[action_index-window_size: action_index]
            c_short_seq = cate_seq[action_index-window_size: action_index]

        self.m_x_short_train.append(x_short_seq)
        self.m_c_short_train.append(c_short_seq)

        seq_len_short = len(x_short_seq)
        self.m_xnum_short_train.append(seq_len_short)

        cnum_long = 0
        x_long = []
        xnum_long = []
        c_long = []

        for c in c_x_user:
            x_long_seq = c_x_user[c].copy()[-cate_window_size:]
            xnum_long_seq = len(x_long_seq)

            x_long.append(x_long_seq)
            xnum_long.append(xnum_long_seq)

            cnum_long += 1

            c_long.append(c)

        self.m_c_long_train.append(c_long)
        self.m_z_train.append(cate_cur)

        self.m_x_long_train.append(x_long)
        self.m_xnum_long_train.append(xnum_long)
        self.m_cnum_long_train.append(cnum_long)

        y = action_seq[action_index]
        self.m_y_train.append(y)
        self.m_y_idx_train.append(action_index)

    def add_item2test(self, action_index, window_size, cate_window_size, cate_cur, action_seq, cate_seq, c_x_user):
        x_short_seq = None
        c_short_seq = None

        if action_index <= window_size:
            x_short_seq = action_seq[:action_index]
            c_short_seq = cate_seq[:action_index]
        else:
            x_short_seq = action_seq[action_index-window_size: action_index]
            c_short_seq = cate_seq[action_index-window_size: action_index]

        self.m_x_short_test.append(x_short_seq)
        self.m_c_short_test.append(c_short_seq)

        seq_len_short = len(x_short_seq)
        self.m_xnum_short_test.append(seq_len_short)

        cnum_long = 0
        x_long = []
        xnum_long = []
        c_long = []

        for c in c_x_user:
            x_long_seq = c_x_user[c].copy()[-cate_window_size:]
            xnum_long_seq = len(x_long_seq)

            x_long.append(x_long_seq)
            xnum_long.append(xnum_long_seq)

            cnum_long += 1

            c_long.append(c)

        self.m_c_long_test.append(c_long)
        self.m_z_test.append(cate_cur)

        self.m_x_long_test.append(x_long)
        self.m_xnum_long_test.append(xnum_long)
        self.m_cnum_long_test.append(cnum_long)

        y = action_seq[action_index]
        self.m_y_test.append(y)
        self.m_y_idx_test.append(action_index)

    def items(self):
        print("item num", len(self.m_itemmap))
        return len(self.m_itemmap)

    def cates(self):
        print("cate num", len(self.m_catemap))
        return len(self.m_catemap)

class MYDATASET(object):
    def __init__(self, x_long, xnum_long, cnum_long, c_long, x_short, c_short, xnum_short, y, z, y_idx):
        self.m_x_long = x_long
        self.m_xnum_long = xnum_long

        self.m_c_long = c_long
        self.m_cnum_long = cnum_long

        self.m_x_short = x_short
        self.m_c_short = c_short
        self.m_xnum_short = xnum_short

        self.m_y = y
        self.m_z = z
        self.m_y_idx = y_idx

class MYDATALOADER(object):
    def __init__(self, dataset, batch_size):
        self.m_dataset = dataset
        self.m_batch_size = batch_size

        print("len", len(self.m_dataset.m_cnum_long))

        sorted_data = sorted(zip(self.m_dataset.m_cnum_long, self.m_dataset.m_x_long, self.m_dataset.m_xnum_long, self.m_dataset.m_c_long, self.m_dataset.m_x_short, self.m_dataset.m_c_short, self.m_dataset.m_xnum_short, self.m_dataset.m_y, self.m_dataset.m_z, self.m_dataset.m_y_idx))

        self.m_dataset.m_cnum_long, self.m_dataset.m_x_long, self.m_dataset.m_xnum_long, self.m_dataset.m_c_long, self.m_dataset.m_x_short, self.m_dataset.m_c_short, self.m_dataset.m_xnum_short, self.m_dataset.m_y, self.m_dataset.m_z, self.m_dataset.m_y_idx = zip(*sorted_data)

        seq_num = len(self.m_dataset.m_cnum_long)
        batch_num = seq_num//batch_size

        print("seq num", seq_num)
        print("batch size", self.m_batch_size)
        print("batch num", batch_num)

        cnum_long_list = [self.m_dataset.m_cum_long[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]
        x_long_list = [self.m_dataset.m_x_long[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]
        xnum_long_list = [self.m_dataset.m_xnum_long[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]

        c_long_list = [self.m_dataset.m_c_long[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]

        x_short_list = [self.m_dataset.m_x_short[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]
        c_short_list = [self.m_dataset.m_c_short[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]

        xnum_short_list = [self.m_dataset.m_xnum_short[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]
        y = [self.m_dataset.m_y[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]
        z = [self.m_dataset.m_z[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]

        y_idx = [self.m_dataset.m_y_idx[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]

        temp = list(zip(cnum_long_list, x_long_list, xnum_long_list, c_long_list, x_short_list, c_short_list, xnum_short_list, y, z, y_idx))

        self.m_temp = temp

    def __iter__(self):
        print("shuffling")
        temp = self.m_temp 

        cnum_long, x_long, xnum_long, c_long, x_short, c_short, xnum_short, y, z, y_idx = zip(*temp)

        batch_size = self.m_batch_size
        batch_num = len(cnum_long)

        for batch_index in range(batch_num):
            x_long_batch = x_long[batch_index]
            xnum_long_batch = xnum_long[batch_index]
            
            c_long_batch = c_long[batch_index]
            cnum_long_batch = cnum_long[batch_index]

            x_short_batch = x_short[batch_index]
            c_short_batch = c_short[batch_index]
            xnum_short_batch = xnum_short[batch_index]

            y_batch = y[batch_index]
            z_batch = z[batch_index]
            y_idx_batch = y_idx[batch_index]

            if batch_index%4000 == 0:
                print("batch index", batch_index)
            
            x_long_iter = []
            xnum_long_iter = []

            c_long_iter = []
            cnum_long_iter = []

            x_short_iter = []
            c_short_iter = []
            xnum_short_iter = []

            y_iter = []
            z_iter = []
            y_idx_iter = []

            max_xnum_long_iter = 0
            
            for seq_index_batch in range(batch_size):
                max_xnum_long_iter = max(max_xnum_long_iter, max(xnum_long_batch[seq_index_batch]))
            
            max_cnum_long_iter = max(cnum_long_batch)
            max_xnum_short_iter = max(xnum_short_batch)

            for seq_index_batch in range(batch_size):
                x_long_seq = x_long_batch[seq_index_batch]
                xnum_long_seq = xnum_long_batch[seq_index_batch]

                c_long_seq = c_long_batch[seq_index_batch]

                for subseq_index in range(len(x_long_seq)):
                    xnum_subseq = xnum_long_seq[subseq_index]
                    subseq = x_long_seq[subseq_index]

                    assert xnum_subseq == len(subseq)

                    pad_subseq = subseq+[0]*(max_xnum_long_iter-xnum_subseq)
                    x_long_iter.append(pad_subseq)
                    xnum_long_iter.append(xnum_subseq)

                    c_subseq = c_long_seq[subseq_index]
                    c_long_iter.append(c_subseq)
            
                cnum_long_seq = cnum_long_batch[seq_index_batch]

                # pad_c_long_seq = c_long_seq+[0]*(max_cnum_long_iter-cnum_long_seq)
                # c_long_iter.append(pad_c_long_seq)

                cnum_long_iter.append(cnum_long_seq)

                x_short_seq = x_short_batch[seq_index_batch]
                xnum_short_seq = xnum_short_batch[seq_index_batch]
                pad_x_short_seq = x_short_seq+[0]*(max_xnum_short_iter-xnum_short_seq)

                x_short_iter.append(pad_x_short_seq)
                xnum_short_iter.append(xnum_short_seq)

                c_short_seq = c_short_batch[seq_index_batch]
                pad_c_short_seq = c_short_seq+[0]*(max_xnum_short_iter-xnum_short_seq)
                c_short_iter.append(pad_c_short_seq)

                y_seq = y_batch[seq_index_batch]
                y_iter.append(y_seq)

                z_seq = z_batch[seq_index_batch]
                z_iter.append(z_seq)

                y_idx_iter.append(y_idx_batch[seq_index_batch])

            x_long_iter = np.array(x_long_iter)
            x_long_iter = torch.from_numpy(x_long_iter)

            xnum_long_iter = np.array(xnum_long_iter)
            xnum_long_iter = torch.from_numpy(xnum_long_iter)

            c_long_iter = np.array(c_long_iter)
            c_long_iter = torch.from_numpy(c_long_iter)

            cnum_long_iter = np.array(cnum_long_iter)
            cnum_long_iter = torch.from_numpy(cnum_long_iter)

            x_short_iter = np.array(x_short_iter)
            x_short_iter = torch.from_numpy(x_short_iter)

            xnum_short_iter = np.array(xnum_short_iter)
            xnum_short_iter = torch.from_numpy(xnum_short_iter)

            c_short_iter = np.array(c_short_iter)
            c_short_iter = torch.from_numpy(c_short_iter)

            y_iter = np.array(y_iter)
            y_iter = torch.from_numpy(y_iter)

            z_iter = np.array(z_iter)
            z_iter = torch.from_numpy(z_iter)

            y_idx_iter = np.array(y_idx_iter)
            y_idx_iter = torch.from_numpy(y_idx_iter)

            yield x_long_iter, xnum_long_iter, c_long_iter, cnum_long_iter, x_short_iter, xnum_short_iter, c_short_iter, y_iter, z_iter, y_idx_iter


