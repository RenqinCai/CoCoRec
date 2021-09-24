from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import datetime

class GatedLongRec(nn.Module):
    def __init__(self, log, ss, input_size, hidden_size, output_size, embedding_dim, cate_input_size, cate_output_size, cate_embedding_dim, cate_hidden_size, num_layers=1, final_act='tanh', dropout_hidden=.2, dropout_input=0, use_cuda=False, shared_embedding=True, cate_shared_embedding=True):
        super(GatedLongRec, self).__init__()
        self.m_log = log
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        ### long-term encoder and short-term encoder
        self.m_itemNN = ITEMNN(input_size, hidden_size, output_size, embedding_dim, num_layers, final_act, dropout_hidden, dropout_input, use_cuda, shared_embedding)
        
        ### gating network
        self.m_cateNN = CATENN(cate_input_size, cate_output_size, cate_embedding_dim, cate_hidden_size, num_layers, final_act, dropout_hidden, dropout_input, use_cuda, cate_shared_embedding)

        self.fc = nn.Linear(hidden_size*2, hidden_size)
        self.m_ss = ss

        if shared_embedding:
            message = "share embedding"
            self.m_log.addOutput2IO(message)
            self.m_ss.params.weight = self.m_itemNN.look_up.weight

        self = self.to(self.device)

    def forward(self, action_cate_long_batch, cate_long_batch, action_cate_mask_long_batch, cate_mask_long_batch, max_actionNum_cate_long_batch, max_cateNum_long_batch, actionNum_cate_long_batch, cateNum_long_batch, action_short_batch, cate_short_batch, action_mask_short_batch, actionNum_short_batch, y_cate_batch, train_test_flag):
    
        seq_cate_short_input = self.m_cateNN(cate_short_batch, action_mask_short_batch, actionNum_short_batch, train_test_flag)
        logit_cate_short = self.m_cateNN.m_cate_h2o(seq_cate_short_input)

        seq_cate_input, seq_short_input = self.m_itemNN(action_cate_long_batch, cate_long_batch, action_cate_mask_long_batch, cate_mask_long_batch, max_actionNum_cate_long_batch, max_cateNum_long_batch, actionNum_cate_long_batch, cateNum_long_batch, action_short_batch, action_mask_short_batch, actionNum_short_batch, y_cate_batch, train_test_flag)

        mixture_output = torch.cat((seq_cate_input, seq_short_input), dim=1)
        fc_output = self.fc(mixture_output)

        return fc_output, logit_cate_short

class ITEMNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_dim, num_layers=1, final_act='tanh', dropout_hidden=.2, dropout_input=0,use_cuda=False, shared_embedding=True):
        super(ITEMNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_hidden = dropout_hidden
        self.dropout_input = dropout_input
        self.embedding_dim = embedding_dim
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.look_up = nn.Embedding(input_size, self.embedding_dim)

        ### long-term encoder
        self.m_cate_session_gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden, batch_first=True)

        ### short-term encoder
        self.m_short_gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden, batch_first=True)

    def forward(self, action_cate_long_batch, cate_long_batch, action_cate_mask_long_batch, cate_mask_long_batch, max_actionNum_cate_long_batch, max_cateNum_long_batch, actionNum_cate_long_batch, cateNum_long_batch, action_short_batch, action_mask_short_batch, actionNum_short_batch, y_cate_batch, train_test_flag):
        
        y_cate_batch = y_cate_batch.reshape(-1, 1)

        ### retrieve cate concerning target cate
        y_cate_index = (cate_long_batch == y_cate_batch).float()
        y_cate_index = y_cate_index.unsqueeze(-1)

        ### long-term dependence
        action_cate_long_batch = action_cate_long_batch.reshape(y_cate_batch.size(0), -1, action_cate_long_batch.size(1))
        action_cate_long_batch_mask = action_cate_long_batch*y_cate_index.long()
        action_cate_long_batch_mask = torch.sum(action_cate_long_batch_mask, dim=1)

        action_cate_input = action_cate_long_batch_mask
        action_cate_embedded = self.look_up(action_cate_input)

        action_cate_batch_size = action_cate_embedded.size(0)
        action_cate_hidden = self.init_hidden(action_cate_batch_size, self.hidden_size)

        ### action_cate_embedded: batch_size*action_num_cate*hidden_size
        action_cate_output, action_cate_hidden = self.m_cate_session_gru(action_cate_embedded, action_cate_hidden) # (sequence, B, H)

        action_cate_mask_long_batch = action_cate_mask_long_batch.reshape(y_cate_batch.size(0), -1, action_cate_mask_long_batch.size(1))
        action_cate_mask_long_batch_mask = action_cate_mask_long_batch*y_cate_index
        ### action_cate_mask_long_batch_mask: batch_size*max_action_num_cate
        action_cate_mask_long_batch_mask = torch.sum(action_cate_mask_long_batch_mask, dim=1)

        # print("action_cate_mask_long_batch_mask", action_cate_mask_long_batch_mask.size())
        action_cate_mask_long_batch_mask = action_cate_mask_long_batch_mask.unsqueeze(-1)
        # output_subseq_cate 

        action_cate_output_mask = action_cate_output*action_cate_mask_long_batch_mask

        actionNum_cate_long_batch = actionNum_cate_long_batch.reshape(y_cate_batch.size(0),-1)
        actionNum_cate_long_batch = torch.from_numpy(actionNum_cate_long_batch).to(self.device)

        ### actionNum_cate_long_batch: batch_size*cate_num
        actionNum_cate_long_batch = actionNum_cate_long_batch*y_cate_index.squeeze(-1).long()
        actionNum_cate_long_batch = torch.sum(actionNum_cate_long_batch, dim=1)

        # pad_subseqLen_cate_batch = np.array([i-1 if i > 0 else 0 for i in subseqLen_cate_batch])
        first_dim_index = torch.arange(action_cate_batch_size).to(self.device)
        second_dim_index = actionNum_cate_long_batch

        ### long-term dependence: seq_cate_input
        ### batch_size*hidden_size
        seq_cate_input = action_cate_output_mask[first_dim_index, second_dim_index, :]
        
        #####################
        ### short-range actions
        action_short_input = action_short_batch.long()
        action_short_embedded = self.look_up(action_short_input)
        
        short_batch_size = action_short_embedded.size(0) 
        
        action_short_hidden = self.init_hidden(short_batch_size, self.hidden_size)
        action_short_output, action_short_hidden = self.m_short_gru(action_short_embedded, action_short_hidden)

        action_mask_short_batch = action_mask_short_batch.unsqueeze(-1).float()
        action_short_output_mask = action_short_output*action_mask_short_batch

        # pad_seqLen_batch = [i-1 if i > 0 else 0 for i in seqLen_batch]
        first_dim_index = torch.arange(short_batch_size).to(self.device)
        second_dim_index = torch.from_numpy(actionNum_short_batch).to(self.device)

        ### short-term dependence: seq_short_input
        ### batch_size*hidden_size
        seq_short_input = action_short_output_mask[first_dim_index, second_dim_index, :]

        # output_seq, hidden_seq = self.m_short_gru(input_seq, hidden_seq)

        return seq_cate_input, seq_short_input

    def init_hidden(self, batch_size, hidden_size):
        '''
        Initialize the hidden state of the GRU
        '''
        h0 = torch.zeros(self.num_layers, batch_size, hidden_size).to(self.device)
        return h0

class CATENN(nn.Module):
    def __init__(self, cate_input_size, cate_output_size, cate_embedding_dim, cate_hidden_size, num_layers=1, final_act='tanh', dropout_hidden=.2, dropout_input=0, use_cuda=False, cate_shared_embedding=True):
        super(CATENN, self).__init__()
        self.num_layers = num_layers
        self.dropout_hidden = dropout_hidden

        self.m_cate_input_size = cate_input_size
        self.m_cate_embedding_dim = cate_embedding_dim
        self.m_cate_hidden_size = cate_hidden_size
        self.m_cate_output_size = cate_output_size

        self.m_cate_embedding = nn.Embedding(self.m_cate_input_size, self.m_cate_embedding_dim)
        self.m_cate_gru = nn.GRU(self.m_cate_embedding_dim, self.m_cate_hidden_size, self.num_layers, dropout=self.dropout_hidden, batch_first=True)
        self.m_cate_h2o = nn.Linear(self.m_cate_hidden_size, self.m_cate_output_size)
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        if cate_shared_embedding:
            self.m_cate_h2o.weight = self.m_cate_embedding.weight

    def forward(self, cate_short_batch, action_mask_short_batch, actionNum_short_batch, train_test_flag):
        cate_short_embedded = self.m_cate_embedding(cate_short_batch)
        short_batch_size = cate_short_embedded.size(0)
        cate_short_hidden = self.init_hidden(short_batch_size, self.m_cate_hidden_size)
        cate_short_output, cate_short_hidden = self.m_cate_gru(cate_short_embedded, cate_short_hidden)

        action_mask_short_batch = action_mask_short_batch.unsqueeze(-1).float()
        cate_short_output_mask = cate_short_output*action_mask_short_batch

        first_dim_index = torch.arange(short_batch_size).to(self.device)
        second_dim_index = torch.from_numpy(actionNum_short_batch).to(self.device)
 
        seq_cate_short_input = cate_short_output_mask[first_dim_index, second_dim_index, :]
        
        return seq_cate_short_input
    
    def init_hidden(self, batch_size, hidden_size):
        '''
        Initialize the hidden state of the GRU
        '''
        h0 = torch.zeros(self.num_layers, batch_size, hidden_size).to(self.device)
        return h0
