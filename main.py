from __future__ import division
from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import *
import numpy
from sklearn.metrics import f1_score
import os
import torch.nn as nn
import argparse
from config import Config
from model.model import CGCN
from tensorboardX import SummaryWriter
import math
import time


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--dataset", help="dataset", type=str, required=True)
    parse.add_argument("-l", "--labelrate", help="labeled data for train per class", type = int, required = True)
    parse.add_argument("-it", "--niter", type=int, default=10, help="num of iteration")
    parse.add_argument("-a", "--alpha", type=float, default=1, help="weight of contrastive loss")
    parse.add_argument("-ld", "--log_dir", type=str, default="experiments", help="log path")
    parse.add_argument("-r", "--record", type=str, default="no record", help = "record data using tensorboardX" )
    args = parse.parse_args()
    
    config_file = "./config/" + str(args.labelrate) + str(args.dataset) + ".ini"
    config = Config(config_file)

    def process_output(output1, output2, softmax, nit):
        epsilon = 0.3
        temprature = config.temperature1
        n_ite = nit

        q1 = output1 / epsilon
        q2 = output2 / epsilon
        q1 = torch.exp(q1).t()
        q2 = torch.exp(q2).t()
        q1 = sinkhorn(q1, n_ite)
        q2 = sinkhorn(q2, n_ite)
        
        p1 = softmax(output1 / temprature)
        p2 = softmax(output2 / temprature)

        return p1, p2, q1, q2
       
    def prototype(p1, p2, idx_train, labels, label_rate):
        p1 = p1.cpu().detach().numpy()
        p2 = p2.cpu().detach().numpy()
        pro_1 = np.zeros((num_labels, p1.shape[1]), dtype=float)
        pro_2 = np.zeros((num_labels, p1.shape[1]), dtype=float)
        train_idx = idx_train.numpy()
        labels = labels.numpy()
        
        for i in train_idx:
            row = int(labels[i])
            prob = pro_1[row] + p1[i]
            pro_1[row] = prob
            prob_1 = pro_2[row] + p2[i]
            pro_2[row] = prob_1
        pro_1 = np.array(pro_1/label_rate).T
        target_pro_1 = np.argmax(pro_1, axis=1)
        target_pro_1 = torch.LongTensor(target_pro_1).cuda()
        pro_1 = torch.FloatTensor(pro_1).cuda()
        pro_2 = np.array(pro_2/label_rate).T
        target_pro_2 = np.argmax(pro_2, axis=1)
        pro_2 = torch.FloatTensor(pro_2).cuda()
        target_pro_2 = torch.LongTensor(target_pro_2).cuda()
        return pro_1, pro_2, target_pro_1, target_pro_2
       
    def pro_loss(arr1, arr2, arr3, softmax):
        arr1 = softmax(arr1)
        arr2 = softmax(arr2)
        arr3 = softmax(arr3)
        return arr1, arr2, arr3

    cuda = torch.cuda.is_available()

    use_seed = not config.no_seed
    if use_seed:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if cuda:
            torch.cuda.manual_seed(config.seed)

   
    sadj, fadj = load_graph(args.labelrate, config)
    features, labels, idx_train, idx_test, num_labels = load_data(config)
    # idx_train, idx_val, index_test = load_data_scarce(args.dataset, args.labelrate)
    

    model = CGCN(features.shape[1], config.nhid1, config.nhid2, num_labels, 0.5, 3 * num_labels)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.1)

    if cuda:
        model = model.cuda()
        features = features.cuda()
        sadj = sadj.cuda()
        fadj = fadj.cuda()
        label = labels.cuda()
        softmax = nn.Softmax(dim=1).cuda()
        logsoftmax = nn.LogSoftmax(dim=1).cuda()
    acc_max = 0
    f1_max=0
    epoch_max=0

    temperature1 = config.temperature1
    temperature2 = config.temperature2

    if(args.record == "record"):
        log_dir = os.path.join(args.log_dir, args.dataset)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)

    T1 = time.time()
    for epoch in range(config.epochs):
        model.train()
        output1, output2, out = model(features, sadj, fadj)
        # p1, p2, q1, q2 = process_output(output1[idx_train], output2[idx_train], softmax, config.it)
        p1, p2, q1, q2 = process_output(output1, output2, softmax, config.it)
        
        # pro_1, pro_2, target_pro_1, target_pro_2 = prototype(p1, p2, idx_train, labels, args.labelrate)
        # prototype1, prototype2, prototype3 = pro_loss(pro_1, pro_2, p_out, softmax)
        # pro_pout = p_out
        
        # prototype_loss = -1 * (F.nll_loss(pro_pout, target_pro_1) + F.nll_loss(pro_pout, target_pro_2))
        # prototype_loss = -0 * (torch.mean(torch.sum(pro_1 * torch.log(pro_pout))) + torch.mean(torch.sum(pro_2 * torch.log(pro_pout))))
        # prototype_loss = F.nll_loss(pro_pout, target_pro_1) + F.nll_loss(pro_pout, target_pro_2)
        # prototype_loss = -0.3 * (torch.mean(torch.sum(prototype3 * torch.log(prototype1), dim=1)) + torch.mean(torch.sum(prototype3 * torch.log(prototype2), dim=1)))
        prototype_loss = 0
        loss_con = -1* config.alpha * (torch.mean(torch.sum(q1 * torch.log(p2), dim=1))+torch.mean(torch.sum(q2 * torch.log(p1), dim=1))) / 2
        loss_class = F.nll_loss(out[idx_train], label[idx_train])
        loss = loss_con + loss_class 
        acc_train = accuracy(out[idx_train], label[idx_train])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        output1, output2, out = model(features, sadj, fadj)
        # p1, p2, q1, q2 = process_output(output1[idx_test], output2[idx_test], softmax, config.it)
        p1, p2, q1, q2 = process_output(output1, output2, softmax, config.it)
        # o_l = 0.5*(logsoftmax(output1 / temperature2) + logsoftmax(output2 / temperature2))
        # o_l = 1*logsoftmax(output1/temperature2) + 0 * logsoftmax(output2/temperature2)
        acc_test = accuracy(out[idx_test], label[idx_test])
        loss_class_eval = F.nll_loss(out[idx_test], label[idx_test])
        loss_test = loss_class_eval + loss_con
        label_max = []
        for idx in idx_test:
            label_max.append(torch.argmax(out[idx]).item())
        labelcpu = label[idx_test].data.cpu()
        macro_f1 = f1_score(labelcpu, label_max, average='macro')

        if acc_test > acc_max:
            acc_max = acc_test.item()
            f1_max = macro_f1
            epoch_max = epoch
        if(args.record == "record"):
            writer.add_scalars('loss', {'training loss': loss,  'val loss': loss_test}, epoch)
            writer.add_scalars('accuracy', {'trainging acc': acc_train, 'test loss': acc_test}, epoch)
        if(epoch % 10 == 0):
            print('e:{}'.format(epoch),
                    'l_pred:{:.4f}'.format(loss_class),
                    'l_con:{:.4f}'.format(loss_con),
                    # 'l_pro:{:.4f}'.format(prototype_loss),
                    'ltr: {:.4f}'.format(loss.item()),
                    'atr: {:.4f}'.format(acc_train.item()),
                    'ate: {:.4f}'.format(acc_test.item()),
                    'f1te:{:.4f}'.format(macro_f1.item()))
    print( 'epoch:{}'.format(epoch_max),
            'acc_max: {:.4f}'.format(acc_max),
            'f1_max: {:.4f}'.format(f1_max))
    T2 = time.time()
    print((T2-T1)*1000/config.epochs)
    if(args.record == "record"):
        writer.close()
    
