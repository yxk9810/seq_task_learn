#coding:utf-8
import sys 
import json 
def save_data(data,filename=''):
    writer = open(filename,'a+',encoding='utf-8')
    for t in data:
        writer.write(json.dumps(t,ensure_ascii=False)+'\n')
    writer.close()
cnt = 0 
track1_data = json.load(open('./data/track1_train.json','r',encoding='utf-8'))
print(len(track1_data))
max_seq_len = 0 
dataset = []
uniq_labels = set()
for d in track1_data:
    max_seq_len = max(max_seq_len,len(d['sentences']))
    labels = []
    for sent in d['sentences']:
        labels.append(d['sentence_quality'][sent][0])
    json_data = {'sentences':d['sentences'],'labels':labels}
    dataset.append(json_data)
    assert len(json_data['sentences']) == len(labels)
    for l in labels:
        uniq_labels.add(l)
print(uniq_labels)
import random 
random.shuffle(dataset)
train_size =int(len(dataset)*0.9)
train_data = dataset[:train_size]
dev_data = dataset[train_size:]
dev = dev_data[:20]
test = dev_data[20:]
save_data(train_data,'./data/track1_train.txt')
save_data(dev,'./data/track1_dev.txt')
save_data(test,'./data/track1_test.txt')
