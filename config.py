#coding:utf-8
#define config file 
checkpoint_name = 'bert-base-chinese'
checkpoint_name = 'hfl/chinese-roberta-wwm-ext'
class Config:
    pretrain_model_path = checkpoint_name
    hidden_size = 768
    learning_rate = 2e-5 
    class_num = 5 
    epoch = 5
    train_file = './data/track1_train_0422.txt'
    dev_file = './data/track1_dev_0422.txt'
    test_file = './data/track1_test_0422.txt'
    target_dir = './models/'
    seq_len = 16
    use_bilstm=True
