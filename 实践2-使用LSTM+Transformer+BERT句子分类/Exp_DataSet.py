import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset
import jieba
from gensim.models import KeyedVectors

class Dictionary(object):
    def __init__(self, path):

        self.word2tkn = {"[PAD]": 0}
        self.tkn2word = ["[PAD]"]

        self.label2idx = {} #内容为 label->idx 的映射
        self.idx2label = [] #内容为 [label, label_desc] 的列表

        # 获取 label 的 映射
        with open(os.path.join(path, 'labels.json'), 'r', encoding='utf-8') as f:
            for line in f:
                one_data = json.loads(line)
                label, label_desc = one_data['label'], one_data['label_desc']
                self.idx2label.append([label, label_desc])
                self.label2idx[label] = len(self.idx2label) - 1

    def add_word(self, word):
        if word not in self.word2tkn:
            self.tkn2word.append(word)
            self.word2tkn[word] = len(self.tkn2word) - 1
        return self.word2tkn[word]


class Corpus(object):
    '''
    完成对数据集的读取和预处理，处理后得到所有文本数据的对应的 token 表示及相应的标签。
    
    该类适用于任务一、任务二，若要完成任务三，需对整个类进行调整，例如，可直接调用预训练模型提供的 tokenizer 将文本转为对应的 token 序列。
    '''
    def __init__(self, path, max_token_per_sent):
        self.dictionary = Dictionary(path)

        self.max_token_per_sent = max_token_per_sent

        self.train = self.tokenize(os.path.join(path, 'train.json'))
        self.valid = self.tokenize(os.path.join(path, 'dev.json'))
        self.test = self.tokenize(os.path.join(path, 'test.json'), True)
       

        #-----------------------------------------------------begin-----------------------------------------------------#
        # 若要采用预训练的 embedding, 需处理得到 token->embedding 的映射矩阵 embedding_weight。矩阵的格式参考 nn.Embedding() 中的参数 _weight
        # 注意，需考虑 [PAD] 和 [UNK] 两个特殊词向量的设置
        word2vector = KeyedVectors.load_word2vec_format(os.path.join(path,'sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'))
        embedding_dim = word2vector.vector_size
        embedding_weight = np.zeros((len(self.dictionary.tkn2word), embedding_dim))
        for word, token in self.dictionary.word2tkn.items():
            if word in word2vector:
                embedding_weight[token] = word2vector[word]
            else:
                embedding_weight[token] = np.random.uniform(-0.01, 0.01, embedding_dim).astype("float32")
        self.embedding_weight = torch.tensor(embedding_weight, dtype=torch.float32)
        

        #把train, valid, test 中的label换乘词向量
        #不需要这一步，在LSTM_Model中会自动转换
        # for i in range(len(self.train)):
        #     self.train[i][1] = self.embedding_weight[self.train[i][1]]
        # for i in range(len(self.valid)):
        #     self.valid[i][1] = self.embedding_weight[self.valid[i][1]]
        # for i in range(len(self.test)):
        #     self.test[i][1] = self.embedding_weight[self.test[i][1]]
        #------------------------------------------------------end------------------------------------------------------#

    def pad(self, origin_token_seq):
        '''
        padding: 将原始的 token 序列补 0 至预设的最大长度 self.max_token_per_sent
        '''
        if len(origin_token_seq) > self.max_token_per_sent:
            return origin_token_seq[:self.max_token_per_sent]
        else:
            return origin_token_seq + [0 for _ in range(self.max_token_per_sent-len(origin_token_seq))]

    def tokenize(self, path, test_mode=False):
        '''
        处理指定的数据集分割，处理后每条数据中的 sentence 都将转化成对应的 token 序列。
        '''
        idss = []
        labels = []
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                one_data = json.loads(line)  # 读取一条数据
                sent = one_data['sentence']
                #-----------------------------------------------------begin-----------------------------------------------------#
                # 若要采用预训练的 embedding, 需在此处对 sent 进行分词

                sent = jieba.lcut(sent)

                #------------------------------------------------------end------------------------------------------------------#
                # 向词典中添加词
                for word in sent:
                    self.dictionary.add_word(word)

                ids = []
                for word in sent:
                    ids.append(self.dictionary.word2tkn[word])
                idss.append(self.pad(ids))
                
                # 测试集无标签，在 label 中存测试数据的 id，便于最终预测文件的打印
                if test_mode:
                    label = json.loads(line)['id']      
                    labels.append(label)
                else:
                    label = json.loads(line)['label']
                    labels.append(self.dictionary.label2idx[label])

            idss = torch.tensor(np.array(idss))
            labels = torch.tensor(np.array(labels)).long()
             #idss的内容格式：[ ids1,id2....    ]
             #labels的内容格式：[label1,label2....]
             #TensorDataset是一个pytorch的包装数据的类，可以把数据包装成TensorDataset的形式，然后再放入DataLoader中
             #格式是 (ids1, label1)元素组成的数据集
        return TensorDataset(idss, labels)