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

    def forward(self, x_long, xnum_long, c_long, cnum_long, x_short, xnum_short, c_short, y, z, train_test_flag):
        output_cate = self.m_cateNN(c_short, xnum_short, train_test_flag)
        logit_cate = self.m_cateNN.m_cate_h2o(output_cate)

        output_action_long, output_action_short = self.m_itemNN(x_long, xnum_long, c_long, cnum_long, x_short, xnum_short, z, train_test_flag)
    
        mix_output_action = torch.cat([output_action_long, output_action_short], dim=1)
        logit_action = self.fc(mix_output_action)

        return logit_action, logit_cate

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

        ### first order within channel
        self.m_cate_ll = nn.Linear(self.embedding_dim, self.hidden_size)

        ### long-term encoder
        self.m_cate_session_gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden, batch_first=True)

        ### first order short-term
        self.m_short_ll = nn.Linear(self.embedding_dim, self.hidden_size)

        ### short-term encoder
        self.m_short_gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden, batch_first=True)

    def forward(self, x_long, xnum_long, c_long, cnum_long, x_short, xnum_short, z, train_test_flag):
        
        ### z: batch_size*1

        ### x_long: (batch_size*cate_num_batch)*max_action_num_batch

        ### xnum_long: (batch_size*cate_num_batch)*1

        ### c_long: (batch_size*cate_num_batch)

        ### cnum_long: batch_size

        ### mask_c: batch_size*cate_num_batch
        expand_z = z.repeat(cnum_long, 1)
        mask_c = c_long == expand_z

        ### x_masked_long: (batch_size)*max_action_num_batch
        x_masked_long = x_long[mask_c]
        xnum_masked_long = xnum_long[mask_c]

        print("x_masked_long", x_masked_long.size())
        
        x_embed_long = self.look_up(x_masked_long)
        batch_size = x_embed_long.size(0)
        x_hidden_long = self.init_hidden(batch_size, self.hidden_size)

        ### x_seq_output: batch_size*action_num*hidden_size
        x_seq_output_long, x_hidden_long = self.m_cate_session_gru(x_embed_long, x_hidden_long)

        first_dim_index = torch.arange(batch_size).to(self.device)
        second_dim_index = xnum_masked_long

        ### x_output: batch_size*hidden_size
        x_output_long = x_seq_output_long[first_dim_index, second_dim_index, :]

        ## x_short: bathc_size*action_num
        x_embed_short = self.look_up(x_short)
        batch_size = x_embed_short.size(0)

        x_hidden_short = self.init_hidden(batch_size, self.hidden_size)
        x_seq_output_short, x_hidden_short = self.m_short_gru(x_embed_short, x_hidden_short)

        first_dim_index = torch.arange(batch_size).to(self.device)
        second_dim_index = torch.from_numpy(xnum_short).to(self.device)

        x_output_short = x_seq_output_short[first_dim_index, second_dim_index, :]

        return x_output_long, x_output_short

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

    def forward(self, c_short, xnum_short, train_test_flag):
        cate_embed_short = self.m_cate_embedding(c_short)
        batch_size = cate_embed_short.size(0)

        cate_hidden_short = self.init_hidden(batch_size, self.m_cate_hidden_size)
        cate_seq_output_short, cate_hidden_short = self.m_cate_gru(cate_embed_short, cate_hidden_short)

        first_dim_index = torch.arange(batch_size).to(self.device)
        second_dim_index = torch.from_numpy(xnum_short).to(self.device)
 
        cate_output_short = cate_seq_output_short[first_dim_index, second_dim_index, :]
        
        return cate_output_short
        
    def init_hidden(self, batch_size, hidden_size):
        '''
        Initialize the hidden state of the GRU
        '''
        h0 = torch.zeros(self.num_layers, batch_size, hidden_size).to(self.device)
        return h0
