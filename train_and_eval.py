#coding:utf-8
import torch 
import sys 
from dataset import ZhWikipediaDataSet,collate_fn_wiki
from torch.utils.data import DataLoader
from transformers import AdamW
device = torch.device("cuda")
from seq_model import SeqModel
import torch.nn as nn 
import os 
import time 
now_time = time.strftime("%Y%m%d%H", time.localtime())
import torch
import random
import numpy as np
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
from config import Config 
config = Config()
model = SeqModel(config)
model.to(device)
from sklearn.metrics import classification_report
optimizer = AdamW(model.parameters(),lr = config.learning_rate)
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""
    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = torch.nn.functional.nll_loss(log_pt, target, self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss
        
def train(model,train_data_loader):
    model.train()
    total_loss,total_accuracy = 0,0 
    for step,batch in enumerate(train_data_loader):
        sent_id,mask,labels = batch[0].to(device),batch[1].to(device),batch[2].to(device)
        model.zero_grad()
        logits = model(sent_id,mask)
        loss_fct =  nn.CrossEntropyLoss()
        loss_fct  = FocalLoss()
        #loss = loss_fct(logits.view(-1,config.class_num),labels.view(-1))
        loss= F.binary_cross_entropy_with_logits(logits.view(-1,config.class_num),labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        #update parameters 
        optimizer.step()
        loss_item = loss.item()
        # if step % 10 == 0 and not step == 0:
        #     print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_data_loader)))
        #     print("loss",loss_item)
        total_loss+=loss_item
    avg_loss = total_loss/ len(train_data_loader)
    return avg_loss 

def evaluate(model,dev_data_loader):
    model.eval()
    total_loss,total_accuracy = 0,0 
    count = 0 
    golds= []
    predicts = [] 
    for step,batch in enumerate(dev_data_loader):
        sent_id,mask,labels = batch[0].to(device),batch[1].to(device),batch[2].to(device)
        logits = model(sent_id,mask)
        loss_fct =  nn.CrossEntropyLoss()
        loss_fct  = FocalLoss()
        #loss = loss_fct(logits.view(-1,config.class_num),labels.view(-1))
        loss= F.binary_cross_entropy_with_logits(logits.view(-1,config.class_num),labels)
        loss_item = loss.item()
        preds =torch.sigmoid(logits).view(-1,config.class_num).detach().cpu().numpy()
        gold = batch[2].detach().cpu().numpy()
        # preds = preds.reshape(np.shape(gold)[0],config.seq_len,config.class_num)
        print(np.shape(gold))
        print(np.shape(preds))
        for i in range(len(gold)):
            gold_list = gold[i].tolist()
            pred_list = preds[i].tolist()
            pred_list =[int(p>0.5) for p in pred_list]
            # print(len(gold_list))
            # print(len(gold))
            # print(gold_list)
            #pred_list = [int(p>0.5) for p in preds[i].tolist()]
            # print(gold_list[0])
            # print(pred_list[0])
            # print(len(gold_list[0]))
            # sys.exit(1)
            tmp_gold = []
            tmp_preds =[] 
            for idx,(g,p) in enumerate(zip(gold_list,pred_list)):
                if g==1:
                    tmp_gold.append(idx)
                else:
                    tmp_gold.append(0)
                if p==1:
                    tmp_preds.append(idx)
                else:
                    tmp_preds.append(0)
            #     if g==-100:continue 
            #     tmp_gold.append(g)
            #     tmp_preds.append(p)
                #golds.extend(g)
                # for t_idx,tg in enumerate(g):
                #     if tg==1:
                #         tmp_gold.append(t_idx)
                #     else:
                #         tmp_gold.append(0)
                # for t_idx,tp in enumerate(p):
                #     if tp>=0.5:
                #         tmp_preds.append(t_idx)
                #     else:
                #         tmp_preds.append(0)
                    
            golds.extend(tmp_gold)
            predicts.extend(tmp_preds)
            if gold_list == pred_list:
                count+=1
        total_loss+=loss_item
    print(classification_report(golds,predicts))
    avg_loss = total_loss/ len(dev_data_loader)
    return avg_loss,count/len(dev_data_loader)


dataset = ZhWikipediaDataSet(filepath=config.train_file,mini_test=True)
print("train data siz {}".format(len(dataset)))
train_data_loader = DataLoader(dataset, batch_size=2, collate_fn = collate_fn_wiki, shuffle=True)
dev_dataset = ZhWikipediaDataSet(filepath=config.dev_file)
print("train data siz {}".format(len(dev_dataset)))

dev_data_loader =  DataLoader(dev_dataset, batch_size=2, collate_fn = collate_fn_wiki, shuffle=False)
best_valid_loss = float('inf')
for epoch in range(config.epoch):
    print('\n Epoch {:} / {:}'.format(epoch+1 ,config.epoch ))
    train_loss = train(model,train_data_loader)
    dev_loss,dev_acc = evaluate(model,dev_data_loader)
    if dev_loss<best_valid_loss:
        best_valid_loss = dev_loss 
        torch.save(model.state_dict(), 'model_weights.pth')
    print('train loss {}'.format(train_loss))
    print('val loss {} val acc {}'.format(dev_loss,dev_acc))

