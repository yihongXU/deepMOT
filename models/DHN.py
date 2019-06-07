# ==========================================================================
#
# This file is a part of implementation for paper:
# DeepMOT: A Differentiable Framework for Training Multiple Object Trackers.
# This contribution is headed by Perception research team, INRIA.
#
# Contributor(s) : Yihong Xu
# INRIA contact  : yihong.xu@inria.fr
#
# ===========================================================================

import torch
import torch.nn as nn

# Deep Hungarian Net #


class Munkrs(nn.Module):
    def __init__(self, element_dim, hidden_dim, target_size, biDirenction, minibatch, is_cuda, is_train=True):
        super(Munkrs, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirect = biDirenction
        self.minibatch = minibatch
        self.is_cuda = is_cuda

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm_row = nn.GRU(element_dim, hidden_dim, bidirectional=biDirenction, num_layers=2)
        self.lstm_col = nn.GRU(512, hidden_dim, bidirectional=biDirenction, num_layers=2)

        # The linear layer that maps from hidden state space to tag space
        if biDirenction:
            # *2 directions * 2 ways concat
            self.hidden2tag_1 = nn.Linear(hidden_dim * 2, 256)
            self.hidden2tag_2 = nn.Linear(256, 64)
            self.hidden2tag_3 = nn.Linear(64, target_size)
        else:
            # * 2 ways concat
            self.hidden2tag_1 = nn.Linear(hidden_dim, target_size)

        self.hidden_row = self.init_hidden(1)
        self.hidden_col = self.init_hidden(1)

        # init layers
        if is_train:
            for m in self.modules():
                if isinstance(m, nn.GRU):
                    print("weight initialization")
                    torch.nn.init.orthogonal_(m.weight_ih_l0.data)
                    torch.nn.init.orthogonal_(m.weight_hh_l0.data)
                    torch.nn.init.orthogonal_(m.weight_ih_l0_reverse.data)
                    torch.nn.init.orthogonal_(m.weight_hh_l0_reverse.data)

                    # initial gate bias as one
                    m.bias_ih_l0.data[0:self.hidden_dim].fill_(-1)
                    m.bias_hh_l0.data[0:self.hidden_dim].fill_(-1)
                    m.bias_ih_l0_reverse.data[0:self.hidden_dim].fill_(-1)
                    m.bias_hh_l0_reverse.data[0:self.hidden_dim].fill_(-1)

                    torch.nn.init.orthogonal_(m.weight_ih_l1.data)
                    torch.nn.init.orthogonal_(m.weight_hh_l1.data)
                    torch.nn.init.orthogonal_(m.weight_ih_l1_reverse.data)
                    torch.nn.init.orthogonal_(m.weight_hh_l1_reverse.data)

                    # initial gate bias as one
                    m.bias_ih_l1.data[0:self.hidden_dim].fill_(-1)
                    m.bias_hh_l1.data[0:self.hidden_dim].fill_(-1)
                    m.bias_ih_l1_reverse.data[0:self.hidden_dim].fill_(-1)
                    m.bias_hh_l1_reverse.data[0:self.hidden_dim].fill_(-1)

    def init_hidden(self, batch):
        # The axes semantics are (num_layers * num_directions, minibatch_size, hidden_dim),
        # one for hidden, others for memory cell

        if self.bidirect:
            if self.is_cuda:
                hidden = torch.zeros(2*2, batch, self.hidden_dim).cuda()
            else:
                hidden = (torch.zeros(2, batch, self.hidden_dim),
                          torch.zeros(2, batch, self.hidden_dim))

        else:
            if self.is_cuda:
                hidden = (torch.zeros(1, batch, self.hidden_dim).cuda(),
                          torch.zeros(1, batch, self.hidden_dim).cuda())
            else:
                hidden = (torch.zeros(1, batch, self.hidden_dim),
                          torch.zeros(1, batch, self.hidden_dim))
        return hidden

    def forward(self, Dt):

        # Dt is of shape [batch, h, w]
        # input_row is of shape [h*w, batch, 1], [time steps, mini batch, element dimension]
        # row lstm #

        input_row = Dt.view(Dt.size(0), -1, 1).permute(1, 0, 2).contiguous()
        lstm_R_out, self.hidden_row = self.lstm_row(input_row, self.hidden_row)

        # column lstm #
        # lstm_R_out is of shape [seq_len=h*w, batch, hidden_size * num_directions]

        # [h * w*batch, hidden_size * num_directions]
        lstm_R_out = lstm_R_out.view(-1, lstm_R_out.size(2))

        # [h * w*batch, 1]
        # lstm_R_out = self.hidden2tag_1(lstm_R_out).view(-1, Dt.size(0))

        # [h,  w, batch, hidden_size * num_directions]
        lstm_R_out = lstm_R_out.view(Dt.size(1), Dt.size(2), Dt.size(0), -1)

        # col wise vector
        # [w,  h, batch, hidden_size * num_directions]
        input_col = lstm_R_out.permute(1, 0, 2, 3).contiguous()
        # [w*h, batch, hidden_size * num_directions]
        input_col = input_col.view(-1, input_col.size(2), input_col.size(3)).contiguous()
        lstm_C_out, self.hidden_col = self.lstm_col(input_col, self.hidden_col)

        # undo col wise vector
        # lstm_out is of shape [seq_len=time steps=w*h, batch, hidden_size * num_directions]

        # [h, w, batch, hidden_size * num_directions]
        lstm_C_out = lstm_C_out.view(Dt.size(2), Dt.size(1), Dt.size(0), -1).permute(1, 0, 2, 3).contiguous()

        # [h*w*batch, hidden_size * num_directions]
        lstm_C_out = lstm_C_out.view(-1, lstm_C_out.size(3))

        # [h*w, batch, 1]
        tag_space = self.hidden2tag_1(lstm_C_out)
        tag_space = self.hidden2tag_2(tag_space)
        tag_space = self.hidden2tag_3(tag_space).view(-1, Dt.size(0))
        tag_scores = torch.sigmoid(tag_space)
        # tag_scores is of shape [batch, h, w] as Dt
        return tag_scores.view(Dt.size(1), Dt.size(2), -1).permute(2, 0, 1).contiguous()