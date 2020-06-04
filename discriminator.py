import torch
import torch.autograd as autograd
import torch.nn as nn
import pdb

class Discriminator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, dropout=0.2):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru2hidden = nn.Linear(2*2*hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2*2*1, batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, input, hidden):
        # input dim                                                # batch_size x seq_len
        emb = self.embeddings(input)                               # batch_size x seq_len x embedding_dim
        emb = emb.permute(1, 0, 2)                                 # seq_len x batch_size x embedding_dim
        _, hidden = self.gru(emb, hidden)                          # 4 x batch_size x hidden_dim
        hidden = hidden.permute(1, 0, 2).contiguous()              # batch_size x 4 x hidden_dim
        feature = self.gru2hidden(hidden.view(-1, 4*self.hidden_dim))  # batch_size x 4*hidden_dim
        feature = torch.tanh(feature)
        feature = self.dropout_linear(feature)
        out = self.hidden2out(feature)                                 # batch_size x 1
        out = torch.sigmoid(out)
        return out, feature

    def batchClassify(self, inp):
        """
        Classifies a batch of sequences.

        Inputs: inp
            - inp: batch_size x seq_len

        Returns: out
            - out: batch_size ([0,1] score)
        """

        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        return out[0].view(-1), out[1]

    def batchBCELoss(self, inp, target):
        """
        Returns Binary Cross Entropy Loss for discriminator.

         Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size (binary 1/0)
        """

        loss_fn = nn.BCELoss()
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)[0]
        return loss_fn(out, target)


class Classifier(nn.Module):
    def __init__(self, feature_dim, class_num, gpu=False):
        super(Classifier, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.gpu = gpu

        self.feature2class = nn.Linear(feature_dim, class_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature):
        c = self.feature2class(feature)
        c = self.softmax(c)
        return c