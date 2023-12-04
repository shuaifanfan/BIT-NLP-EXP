import torch.nn as nn
import torch as torch
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer_model(nn.Module):
    def __init__(self, vocab_size, ntoken, d_emb=512, d_hid=2048, nhead=8, nlayers=6, dropout=0.2, embedding_weight=None):
        super(Transformer_model, self).__init__()
        # 将"预训练的词向量"整理成 token->embedding 的二维映射矩阵 emdedding_weight 的形式，初始化 _weight
        # 当 emdedding_weight == None 时，表示随机初始化
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_emb, _weight=embedding_weight)

        self.pos_encoder = PositionalEncoding(d_model=d_emb, max_len=ntoken)
        self.encode_layer = nn.TransformerEncoderLayer(d_model=d_emb, nhead=nhead, dim_feedforward=d_hid)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encode_layer, num_layers=nlayers)
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 请自行设计对 transformer 隐藏层数据的处理和选择方法
        self.dropout = nn.Dropout(dropout)  # 可选
        self.ntoken = ntoken
        self.d_hid = d_hid
        self.d_emb = d_emb
        # 请自行设计分类器
        self.fc = nn.Sequential(
            nn.Linear(self.ntoken*self.d_emb, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 15),
        )

        #------------------------------------------------------end------------------------------------------------------#

    def forward(self, x):
       
        x = self.embed(x)     
        x = x.permute(1, 0, 2)          
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)      # [batch_size, ntoken, d_emb]
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 对 transformer_encoder 的隐藏层输出进行处理和选择，并完成分类
        x = self.dropout(x) # 可选
        x = x.reshape(-1, self.ntoken*self.d_emb) 
       # x = F.avg_pool1d(x.permute(0, 2, 1), x.size(1)).squeeze()   # 池化并挤压后[batch_size, d_emb]
        x = self.fc(x)
        #------------------------------------------------------end------------------------------------------------------#
        return x
    
    
class BiLSTM_model(nn.Module):
    """
    vocab_size: 词表大小,不是词向量表中词的个数，而是数据集中切出来的所有词的个数
    ntoken: 一个句子中的词的最大个数
    d_emb: 词向量的维度
    d_hid: 隐藏层的维度
    nlayers: lstm层数
    dropout: dropout的比例
    embedding_weight: 预训练的词向量，格式为 token->embedding 的二维映射矩阵
    """
    def __init__(self, vocab_size, ntoken, d_emb=100, d_hid=80, nlayers=1, dropout=0.2, embedding_weight=None):
        super(BiLSTM_model, self).__init__()
        # 将"预训练的词向量"整理成 token->embedding 的二维映射矩阵 emdedding_weight 的形式，初始化 _weight
        # 当 emdedding_weight == None 时，表示随机初始化
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_emb, _weight=embedding_weight)

        self.lstm = nn.LSTM(input_size=d_emb, hidden_size=d_hid, num_layers=nlayers, bidirectional=True, batch_first=True)
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 请自行设计对 bilstm 隐藏层数据的处理和选择方法
        self.dropout = nn.Dropout(dropout)  # 可选

        # 请自行设计分类器
        self.classifier = nn.Sequential(
            nn.Linear(ntoken*d_hid*2, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 15),
        )
        self.ntoken = ntoken
        self.d_hid = d_hid
        #------------------------------------------------------end------------------------------------------------------#

    def forward(self, x:torch.Tensor):
        
        # x = x.long()
        #print("输入embed的x:",type(x), x.shape)
        x = self.embed(x)
        x = self.lstm(x)[0]
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 对 bilstm 的隐藏层输出进行处理和选择，并完成分类
        x = self.dropout(x).reshape(-1, self.ntoken*self.d_hid*2)   # ntoken*nhid*2 (2 means bidirectional)
        x = self.classifier(x)
        #------------------------------------------------------end------------------------------------------------------#
        return x