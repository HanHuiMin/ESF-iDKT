import torch
import datetime
import os
import torch.utils.data as Data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from typing import List

#网络定义
num_input = 1
num_output = 1
class RNN(torch.nn.Module):
    def __init__(self,input_size,concept_num,num_skills,hidden_size,Num_layers):
        super().__init__()
        self.rnn=torch.nn.LSTM(
            input_size=concept_num*2,
            hidden_size=hidden_size,
            num_layers=Num_layers,
            batch_first=True
        ).cuda()
        
#         self.input = torch.nn.Linear(in_features=num_skills,out_features=concept_num).cuda()
        self.kc = torch.nn.Linear(in_features=num_skills,out_features=concept_num).cuda()
        self.state=torch.nn.Linear(in_features=hidden_size,out_features=concept_num).cuda()
#         self.predict = torch.nn.Sigmoid().cuda()
        self.out = torch.nn.Linear(in_features=concept_num,out_features=1).cuda()

#     def forward(self,x,target):
#         # 一下关于shape的注释只针对单项
#         # output: [batch_size, time_step, hidden_size]
#         # h_n: [num_layers,batch_size, hidden_size] # 虽然LSTM的batch_first为True,但是h_n/c_n的第一维还是num_layers
#         # c_n: 同h_n
#         #indata = self.input(x)
        
#         #输入数据进过线性层
#         indata = self.input(x)
#         #用lstm进行训练，output是最后一层lstm每个时间步的输出
#         #h_n是每一层在最后一个时间步的输出，c_n是每一层在最后一个时间步的细胞状态
#         h,(h_n,c_n) = self.rnn(indata)
# #         print("ht:",h[0,0,:5])
#         #将每个时间步的隐含变量通过线性层得到学生能力
#         stu_state = self.state(h)
#         #学生能力和题目知识向量做差，（乘个参数是为了左右压缩sigmoid函数）
# #         print("stu_state:",stu_state[0,0,:5])
# #         print("target:",target[0,0,:5])
#         knowledge_div = stu_state-target
#         vector = torch.mul(knowledge_div,target)
#         result = self.out(vector)
#         return (result,stu_state)
    def forward(self,q_hot,result,next_q_hot,concept_num):
        """
        q_hot是当前时刻题目的onehot编码
        result是当前时刻答题结果序列
        next_q_hot是下一时刻题目的onehot编码
        concept_num是知识点个数
        kc_vec=[]
        """
        #将当前时刻题目和下一时刻题目
        fill_hot = torch.zeros(concept_num).cuda()
        kc_hot = self.kc(q_hot)
        next_kc_hot = self.kc(next_q_hot)
        kc_vec=[]
        for i in range(kc_hot.shape[0]):
            batch_vec=[]
            for j in range(kc_hot.shape[1]):
                if result[i][j]==1:
                    single_vec = torch.cat((kc_hot[i][j],fill_hot))
                else:
                    single_vec = torch.cat((fill_hot,kc_hot[i][j]))
                batch_vec.append(single_vec.tolist())
            kc_vec.append(batch_vec)
        kc_vec = torch.tensor(kc_vec).cuda()
        h,(h_n,c_n) = self.rnn(kc_vec)
        stu_state = self.state(h)
        knowledge_div = stu_state-next_kc_hot
        vector = torch.mul(knowledge_div, next_kc_hot)
        result = self.out(vector)
        return (result, stu_state)
        
        
