#coding:utf-8
#define config file 
checkpoint_name = 'bert-base-chinese'
class Config:
    pretrain_model_path = checkpoint_name
    hidden_size = 768
    learning_rate = 2e-5 
    class_num = 5 
    epoch = 5
    train_file = './data/track1_train.txt'
    dev_file = './data/track1_dev.txt'
    test_file = './data/track1_test.txt'
    target_dir = './models/'
    seq_len = 16
    use_bilstm=True
