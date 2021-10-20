import torch
import torch.nn as nn
import torch.nn.functional as F


"""
LSTNet: https://github.com/laiguokun/LSTNet/blob/master/models/LSTNet.py
"""


class LSTNet(nn.Module):
    def __init__(
        self,
        window: int,
        m: int,
        hidRNN: int,
        hidCNN: int,
        hidSkip: int,
        CNN_kernel: int,
        skip: int,
        highway_window: int,
        dropout: float,
        output_fun: str,
    ):
        super(LSTNet, self).__init__()
        self.P = window
        self.m = m
        self.hidR = hidRNN
        self.hidC = hidCNN
        self.hidS = hidSkip
        self.Ck = CNN_kernel
        self.skip = skip

        # cast to int based on github issue
        self.pt = int((self.P - self.Ck) / self.skip)
        self.hw = highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=dropout)
        if self.skip > 0:
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if self.hw > 0:
            self.highway = nn.Linear(self.hw, 1)
        self.output = None
        if output_fun == "sigmoid":
            self.output = torch.sigmoid
        if output_fun == "tanh":
            self.output = torch.tanh

    def forward(self, y_c):
        batch_size = y_c.size(0)

        # CNN
        c = y_c.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))

        # skip-rnn

        if self.skip > 0:
            s = c[:, :, int(-self.pt * self.skip) :].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        res = self.linear1(r)

        # highway
        if self.hw > 0:
            z = y_c[:, -self.hw :, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            res = res + z

        if self.output:
            res = self.output(res)

        return res
