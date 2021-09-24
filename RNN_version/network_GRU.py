from torch import nn
import torch
import torch.nn.functional as F
import datetime
import numpy as np

class GRU4REC(nn.Module):
    def __init__(self, log, ss, input_size, hidden_size, output_size, num_layers=1, final_act='tanh', dropout_hidden=.8, dropout_input=0, embedding_dim=-1, use_cuda=False, shared_embedding=True):
        super(GRU4REC, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_hidden = dropout_hidden
        self.dropout_input = dropout_input
        self.embedding_dim = embedding_dim
    
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        print("self device", self.device)
        self.m_log = log
        
        self.look_up = nn.Embedding(input_size, self.embedding_dim)
        self.m_short_gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden, batch_first=True)

        self.m_ss = ss

        if shared_embedding:
            message = "share embedding"
            self.m_log.addOutput2IO(message)

            self.m_ss.params.weight = self.look_up.weight

        if self.embedding_dim != self.hidden_size:
            self.m_fc = nn.Linear(self.hidden_size, self.embedding_dim)
            self.m_fc_relu = nn.ReLU()
            
        self = self.to(self.device)

    def create_final_activation(self, final_act):
        if final_act == 'tanh':
            self.final_activation = nn.Tanh()
        elif final_act == 'relu':
            self.final_activation = nn.ReLU()
        elif final_act == 'softmax':
            self.final_activation = nn.Softmax()
        elif final_act == 'softmax_logit':
            self.final_activation = nn.LogSoftmax()
        elif final_act.startswith('elu-'):
            self.final_activation = nn.ELU(alpha=float(final_act.split('-')[1]))
        elif final_act.startswith('leaky-'):
            self.final_activation = nn.LeakyReLU(negative_slope=float(final_act.split('-')[1]))

    def forward(self, action_short_batch, action_mask_short_batch, actionNum_short_batch):
        action_short_input = action_short_batch.long()
        action_short_embedded = self.look_up(action_short_input)

        short_batch_size = action_short_embedded.size(0) 
        
        action_short_hidden = self.init_hidden(short_batch_size, self.hidden_size)
        action_short_output, action_short_hidden = self.m_short_gru(action_short_embedded, action_short_hidden)

        action_mask_short_batch = action_mask_short_batch.unsqueeze(-1).float()
        action_short_output_mask = action_short_output*action_mask_short_batch

        first_dim_index = torch.arange(short_batch_size).to(self.device)
        second_dim_index = torch.from_numpy(actionNum_short_batch).to(self.device)

        ### batch_size*hidden_size
        seq_short_input = action_short_output_mask[first_dim_index, second_dim_index, :]

        last_output = seq_short_input
        if self.embedding_dim != self.hidden_size:
            last_output = self.m_fc(seq_short_input)
            last_output = self.m_fc_relu(last_output)

        return last_output

    def onehot_encode(self, input):

        self.onehot_buffer.zero_()

        index = input.unsqueeze(2)
        # index = input.view(-1, 1)
        one_hot = self.onehot_buffer.scatter_(2, index, 1)

        return one_hot

    def embedding_dropout(self, input):
        p_drop = torch.Tensor(input.size(0), input.size(1), 1).fill_(1 - self.dropout_input)  # (B,1)
        mask = torch.bernoulli(p_drop).expand_as(input) / (1 - self.dropout_input)  # (B,C)
        mask = mask.to(self.device)
        input = input * mask  # (B,C)

        return input

    def init_hidden(self, batch_size, hidden_size):
        '''
        Initialize the hidden state of the GRU
        '''
        # print(self.num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, batch_size, hidden_size).to(self.device)

        return h0