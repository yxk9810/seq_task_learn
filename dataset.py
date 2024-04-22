#coding:utf-8
from torch.utils.data import Dataset 
from tqdm import tqdm 
import json 
from transformers import BertTokenizer  
import torch 
from config import checkpoint_name
tokenizer = BertTokenizer.from_pretrained(checkpoint_name)
from config import Config 
config = Config()
import json 
sent2title = {}
track1_data = json.load(open('./data/track1_train.json','r',encoding='utf-8'))
for d in track1_data:
    for sent in d['sentences']:
        sent2title[sent] = d['title']
class ZhWikipediaDataSet(Dataset):
    def __init__(self, filepath='',is_train = True,mini_test = False):
        self.mini_test = mini_test
        self.dataset = self.load_json_data(filepath)
    
    def load_json_data(self,filename):
        error_cnt = 0 
        tmp_dataset = [] 
        with open(filename,'r',encoding='utf-8') as lines:
            for idx,line in enumerate(lines):
                if self.mini_test and idx>100:
                    break 
                try:
                    data = json.loads(line.strip())
                    tmp_dataset.append(data)
                
                except Exception as e:
                    error_cnt+=1
        return tmp_dataset



    def __getitem__(self, index):
        
        return self.dataset[index]


    def __len__(self):
        return len(self.dataset)

def collate_fn_wiki(batch):
    max_sentences_num = config.seq_len
    max_sequence_len = 256 
    batch_data = [] 
    batch_targets = [] 
    for d in batch:
        sentence = d['sentences'][:max_sentences_num]
        labels = d['labels'][:max_sentences_num]
        while len(sentence)<max_sentences_num:
            sentence.append('[PAD]')
            labels.append([0]*5)
        title = sent2title[sentence[0]]
        batch_data.extend([title+'[SEP'+s for s in sentence])
        batch_targets.extend(labels)
    tokens = tokenizer(
                    batch_data,
                    padding = True,
                    max_length = max_sequence_len,
                    truncation=True)
    seq = torch.tensor(tokens['input_ids'])
    mask = torch.tensor(tokens['attention_mask'])
    #y = torch.tensor(batched_targets,dtype=torch.float32).unsqueeze(axis=1)
    y = torch.tensor(batch_targets,dtype=torch.long)    
    return seq, mask, y