from __future__ import division
from __future__ import print_function

import logging
import os
import random

import numpy as np
import torch.optim as optim
from config import parse_args
from dataset import SICKDataset
from metrics import Metrics
from model import *
from trainer import Trainer
from utils import load_word_vectors, build_vocab
from vocab import Vocab
import torch
import Constants
import time
from copy import deepcopy

def main():
    args = parse_args()
    print(args)
    args.cuda = args.cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    train_dir = os.path.join(args.data,'train/')
    dev_dir = os.path.join(args.data,'dev/')
    test_dir = os.path.join(args.data,'test/')

    # write unique words from all token files
    sick_vocab_file = os.path.join(args.data,'sick.vocab')
    if not os.path.isfile(sick_vocab_file):
        token_files_a = [os.path.join(split,'a.toks') for split in [train_dir,dev_dir,test_dir]]
        token_files_b = [os.path.join(split,'b.toks') for split in [train_dir,dev_dir,test_dir]]
        token_files = token_files_a+token_files_b
        sick_vocab_file = os.path.join(args.data,'sick.vocab')
        build_vocab(token_files, sick_vocab_file)

    # get vocab object from vocab file previously written
    vocab = Vocab(filename=sick_vocab_file, data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])

    # load SICK dataset splits
    train_file = os.path.join(args.data,'sick_train.pth')
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = SICKDataset(train_dir, vocab, args.num_classes)
        torch.save(train_dataset, train_file)

    dev_file = os.path.join(args.data,'sick_dev.pth')
    if os.path.isfile(dev_file):
        dev_dataset = torch.load(dev_file)
    else:
        dev_dataset = SICKDataset(dev_dir, vocab, args.num_classes)
        torch.save(dev_dataset, dev_file)

    test_file = os.path.join(args.data,'sick_test.pth')
    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
        test_dataset = SICKDataset(test_dir, vocab, args.num_classes)
        torch.save(test_dataset, test_file)

    # initialize model, criterion/loss_function, optimizer
    model = SimilarityTreeLSTM(
                args.cuda,
                vocab.size(),
                args.input_dim,
                args.mem_dim,
                args.hidden_dim1,
                args.hidden_dim2,
                args.hidden_dim3,
                args.num_classes,
                args.sparse,
                args.att_hops,
                args.att_units,
                args.maxlen,
                args.dropout1,
                args.dropout2,
                args.dropout3,
                freeze_emb=True)

    criterion = nn.KLDivLoss()

    if args.cuda:
        model.cuda(), criterion.cuda()
    if args.optim=='adam':
        optimizer   = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim=='adagrad':
        optimizer   = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim=='sgd':
        optimizer   = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adadelta':
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'asgd':
        optimizer = optim.ASGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)

    metrics = Metrics(args.num_classes)
    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(args.data, 'sick_embed.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = load_word_vectors(os.path.join(args.glove,'glove.840B.300d'))
        emb = torch.Tensor(vocab.size(),glove_emb.size(1)).normal_(-0.05,0.05)
        # zero out the embeddings for padding and other special words if they are absent in vocab
        for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD]):
            emb[idx].zero_()
        for word in vocab.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
        torch.save(emb, emb_file)
    # plug these into embedding matrix inside model
    if args.cuda:
        emb = emb.cuda()

    model.emb.weight.data.copy_(emb)

    # create trainer object for training and testing
    trainer = Trainer(args, model, criterion, optimizer)

    best = -float('inf')

    def adjust_learning_rate(optimizer, epoch):
        # Decay learning rate after 15 epochs
        lr = args.lr * (0.01 ** (epoch // 15))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)

        train_loss = trainer.train(train_dataset)
        dev_loss, dev_pred = trainer.test(dev_dataset, mode='test')

        test_pearson = metrics.pearson(dev_pred,dev_dataset.labels)
        test_mse = metrics.mse(dev_pred, dev_dataset.labels)

        if best < test_pearson:
            best = test_pearson
            checkpoint = {'model': trainer.model.state_dict(), 'optim': trainer.optimizer,
                          'pearson': test_pearson, 'mse': test_mse,
                          'args': args, 'epoch': epoch, 'vocab':vocab}

            torch.save(checkpoint, '%s.pt' % os.path.join(args.save, args.expname + '_' + str(test_pearson)))

    # Evaluate
    trainer.model.load_state_dict(checkpoint['model'])
    # trainer.train(train_dataset)
    test_loss, test_pred = trainer.test(test_dataset, mode='test')
    test_pearson = metrics.pearson(test_pred, test_dataset.labels)
    test_mse = metrics.mse(test_pred, test_dataset.labels)
    # Final read out
    checkpoint = {'model': trainer.model.state_dict(), 'optim': trainer.optimizer,
                  'pearson': test_pearson, 'mse': test_mse,
                  'args': args, 'vocab': vocab}
    torch.save(checkpoint, '%s.pt' % os.path.join(args.save, 'end_model_test'+ str(test_pearson)+'.pt'))


if __name__ == "__main__":
    main()
