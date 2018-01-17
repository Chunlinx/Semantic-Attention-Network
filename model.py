import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import Constants


# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, dropout1):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3*self.mem_dim)
        self.iouh =  nn.Linear(self.mem_dim, 3*self.mem_dim)
        self.fx =  nn.Linear(self.in_dim,self.mem_dim)
        self.fh = nn.Linear(self.mem_dim,self.mem_dim)
        self.H = []
        self.drop = nn.Dropout(dropout1)

    def node_forward(self, inputs, child_c, child_h):
        inputs =  torch.unsqueeze(inputs, 0)
        child_h_sum = torch.sum(child_h, dim=0)
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1)//3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)
        f = F.sigmoid(self.fh(child_h) + self.fx(inputs).repeat(len(child_h), 1))
        fc = torch.mul(f, child_c)
        c = torch.mul(i, u) + torch.sum(fc, dim=0)
        h = torch.mul(o, F.tanh(c))
        self.H.append(h)
        return c, h

    def forward(self, tree, inputs):
        _ = [self.forward(tree.children[idx], inputs) for idx in range(tree.num_children)]

        if tree.num_children==0:
            child_c = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            child_h = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        # self.H.append(tree.state[0])
        return tree.state


class SelfAttentiveEncoder(nn.Module):
    def __init__(self, cuda, mem_dim, att_units, att_hops, maxlen, dropout2):
        super(SelfAttentiveEncoder, self).__init__()

        self.att_units = att_units
        self.hops = att_hops
        self.len = maxlen
        self.attention_att_hops = att_hops
        self.cudaFlag = cuda
        self.pad_idx = 0
        self.drop = nn.Dropout(dropout2)
        self.ws1 =  nn.Linear(mem_dim, att_units, bias=False)
        self.ws2 =  nn.Linear(att_units, att_hops, bias=False)

        self.init_weights()

        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, outp, inputs=None,penalize=True):

        outp = torch.unsqueeze(outp, 0) # Expects input of the form [btch, len, nhid]

        compressed_embeddings = outp.view(outp.size(1), -1)  # [btch*len, nhid]
        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        alphas = self.ws2(hbar).view(1, outp.size(1), -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]

        if penalize and inputs:
            top = Var(torch.zeros(inputs.size(0), self.hops))
            bottom = Var(torch.ones(outp.size(1) - inputs.size(0), self.hops))

            total = torch.cat((top, bottom), 0)
            total = torch.unsqueeze(torch.transpose(total, 0, 1), 0)
            penalized_term = torch.unsqueeze(total, 0)
            if self.cudaFlag:
                penalized_term = penalized_term.cuda()
            penalized_alphas = torch.add(alphas, -10000 * penalized_term)
        else:
            assert penalize == False and inputs == None
            penalized_alphas = alphas

        # [bsz, hop, len] + [bsz, hop, len]
        alphas = self.softmax(penalized_alphas.view(-1, outp.size(1)))  # [bsz*hop, len]
        alphas = alphas.view(outp.size(0), self.hops, outp.size(1))  # [hop, len]
        M = torch.bmm(alphas, outp)  # [bsz, hop, mem_dim]
        return M, alphas


class Similarity(nn.Module):
    def __init__(self, mem_dim, hidden_dim1, hidden_dim2, hidden3, num_classes, att_hops, dropout3):
        super(Similarity, self).__init__()
        self.mem_dim = mem_dim
        self.num_classes = num_classes
        self.att_hops = att_hops

        self.drop = nn.Dropout(dropout3)

        self.lat_att = nn.Linear(self.att_hops, 1, bias=True)
        self.fc = nn.Linear(3*hidden_dim1, hidden_dim2)
        self.out = nn.Linear(hidden_dim2, hidden3)
        self.out1 = nn.Linear(hidden3, num_classes)

    def forward(self, lvec, rvec):
        lvec = self.lat_att(lvec.t()).t()
        rvec = self.lat_att(rvec.t()).t()

        mult_dist = torch.mul(lvec, rvec)
        abs_dist = torch.abs(torch.add(lvec, -rvec))
        mean_dist = 0.5*(torch.add(lvec, rvec))

        fr = torch.cat((abs_dist, mult_dist, mean_dist), 1)

        fc = F.leaky_relu(self.fc(self.drop(fr)))
        fc2 = F.sigmoid(self.out(self.drop(fc)))
        out = F.log_softmax(self.out1(self.drop(fc2)))

        return out


def pad(H, pad, maxlen):
    if H.size(0) > maxlen:
        return H[0:maxlen]
    elif H.size(0) < maxlen:
        pad = torch.cat([pad] * (maxlen - H.size(0)), 0)
        return torch.cat((H, pad), 0)
    else:
        return H


# putting the whole model together
class SimilarityTreeLSTM(nn.Module):
    def __init__(self, cuda, vocab_size, in_dim, mem_dim, hidden_dim1, hidden_dim2,hidden3, num_classes,
                 sparsity, att_hops, att_units, maxlen, dropout1, dropout2, dropout3, freeze_emb):
        super(SimilarityTreeLSTM, self).__init__()

        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
        self.maxlen = maxlen

        self.childsumtreelstm = ChildSumTreeLSTM(in_dim, mem_dim, dropout1)
        self.attention = SelfAttentiveEncoder(cuda, mem_dim, att_units, att_hops, maxlen, dropout2)
        self.similarity = Similarity(mem_dim, hidden_dim1, hidden_dim2, hidden3, num_classes, att_hops, dropout3)

        self.pad_hidden = nn.Parameter(torch.zeros(1, mem_dim))

        self.wf = nn.Parameter(torch.zeros(1, mem_dim, hidden_dim1).uniform_(-1, 1))

        if freeze_emb:
            self.emb.weight.requires_grad = False

    def forward(self, ltree, linputs, rtree, rinputs):
        linputs = self.emb(linputs)
        rinputs = self.emb(rinputs)

        self.childsumtreelstm.H = []
        lstate, lhidden = self.childsumtreelstm.forward(ltree, linputs)
        Hl = torch.cat(self.childsumtreelstm.H, 0)

        Hl = pad(Hl, lhidden.view(1, -1), self.maxlen)

        self.childsumtreelstm.H = []
        rstate, rhidden = self.childsumtreelstm.forward(rtree, rinputs)
        Hr = torch.cat(self.childsumtreelstm.H, 0)
        Hr = pad(Hr, self.pad_hidden, self.maxlen) # [btch, len, mem_dim]

        Ml, attl = self.attention.forward(Hl, linputs)
        Mr, attr = self.attention.forward(Hr, rinputs)  # [btc, hops, mem_dim]

        Ml = F.relu(torch.bmm(Ml, self.wf))
        Mr = F.relu(torch.bmm(Mr, self.wf))

        lstate = torch.squeeze(Ml)
        rstate = torch.squeeze(Mr)

        output = self.similarity.forward(lstate, rstate)
        return output, attl, attr
