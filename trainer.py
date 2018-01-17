from tqdm import tqdm
import torch
from torch.autograd import Variable as Var
from utils import map_label_to_target


def Frobenius(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix
        ret = (torch.sum(torch.sum((mat ** 2), 1), 2).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')


class Trainer(object):
    def __init__(self, args, model, criterion, optimizer):
        super(Trainer, self).__init__()
        self.args       = args
        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.epoch      = 0
        att_hops = args.att_hops

        I = Var(torch.zeros(1, att_hops, att_hops))
        for i in range(1):
            for j in range(att_hops):
                I.data[i][j][j] = 1

        if self.args.cuda == True:
            I = I.cuda()
        self.I = I

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        loss, k = 0.0, 0
        indices = torch.randperm(len(dataset))
        for idx in tqdm(range(len(dataset)),desc='Training epoch '+str(self.epoch+1)+''):
            ltree,lsent,rtree,rsent,label = dataset[indices[idx]]

            linput, rinput = Var(lsent), Var(rsent)
            target = Var(map_label_to_target(label,dataset.num_classes))
            if self.args.cuda:
                linput, rinput = linput.cuda(), rinput.cuda()
                target = target.cuda()
            output, attl, attr = self.model(ltree,linput,rtree,rinput)
            err = self.criterion(output, target)

            loss += err.data[0]

            if attl:  # add penalization term
                attentionT = torch.transpose(attl, 1, 2).contiguous()
                extra_loss = Frobenius(torch.bmm(attl, attentionT) - self.I[0])
                loss += 1 * extra_loss

                attentionT = torch.transpose(attr, 1, 2).contiguous()
                extra_loss = Frobenius(torch.bmm(attr, attentionT) - self.I[0])
                loss += 1 * extra_loss

            err.backward()
            k += 1
            if k%self.args.batchsize==0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        self.epoch += 1
        return loss/len(dataset)

    # helper function for testing
    def test(self, dataset, mode='test'):
        self.model.eval()

        if mode == 'eval':
            inp1 = []
            lattention = []
            rattention = []
            inp2 = []
            sims = []

        loss = 0
        predictions = torch.zeros(len(dataset))
        indices = torch.arange(1,dataset.num_classes+1)
        for idx in tqdm(range(len(dataset)),desc='Testing epoch  ' + str(self.epoch)+''):
            ltree,lsent,rtree,rsent,label = dataset[idx]

            linput, rinput = Var(lsent, volatile=True), Var(rsent, volatile=True)

            if mode == 'test':
                target = Var(map_label_to_target(label, dataset.num_classes), volatile=True)

            if self.args.cuda:
                linput, rinput = linput.cuda(), rinput.cuda()
                if mode == 'test':
                    target = target.cuda()
            output, attl, attr = self.model(ltree, linput, rtree, rinput)

            if mode == 'eval':
                inp1.append(linput)
                inp2.append(rinput)
                cpu_output = output.data.squeeze().cpu()
                sim = torch.dot(indices, torch.exp(cpu_output))
                sims.append(sim)
                lattention.append(attl)
                rattention.append(attr)
            elif mode == 'test':
                err = self.criterion(output, target)
                loss += err.data[0]
                output = output.data.squeeze().cpu()
                predictions[idx] = torch.dot(indices, torch.exp(output))

        if mode == 'test':
            return loss/len(dataset), predictions
        elif mode == 'eval':
            return inp1, inp2, sims, lattention, rattention