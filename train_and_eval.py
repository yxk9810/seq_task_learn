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


def train(model,train_data_loader):
    model.train()
    total_loss,total_accuracy = 0,0 
    for step,batch in enumerate(train_data_loader):
        sent_id,mask,labels = batch[0].to(device),batch[1].to(device),batch[2].to(device)
        model.zero_grad()
        logits = model(sent_id,mask)
        loss_fct =  nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1,config.class_num),labels.view(-1))
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
        loss = loss_fct(logits.view(-1,config.class_num),labels.view(-1))
        loss_item = loss.item()
        preds =torch.argmax(torch.softmax(logits,dim=-1),dim=-1).detach().cpu().numpy()
        gold = batch[2].detach().cpu().numpy()
        # print(gold.shape)
        # print(preds.shape)
        for i in range(len(gold)):
            gold_list = gold[i].tolist()
            pred_list = preds[i].tolist()
            tmp_gold = []
            tmp_preds =[] 
            for g,p in zip(gold_list,pred_list):
                if g==-100:continue 
                tmp_gold.append(g)
                tmp_preds.append(p)
            golds.extend(tmp_gold)
            predicts.extend(tmp_preds)
            # print(gold[i].tolist())
            # print(preds[i].tolist())
            # print(golds[-1])
            # print(predicts[-1])
            # sys.exit(1)
            if tmp_gold==tmp_preds:
                count+=1
        total_loss+=loss_item
    print(classification_report(golds,predicts))
    avg_loss = total_loss/ len(dev_data_loader)
    return avg_loss,count/len(dev_data_loader)


dataset = ZhWikipediaDataSet(filepath=config.train_file)
print('train data siz %d'.format(len(dataset)))
train_data_loader = DataLoader(dataset, batch_size=8, collate_fn = collate_fn_wiki, shuffle=True)
dev_dataset = ZhWikipediaDataSet(filepath=config.dev_file)
dev_data_loader =  DataLoader(dev_dataset, batch_size=4, collate_fn = collate_fn_wiki, shuffle=False)
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

