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

kc_number = 124
#网络定义
num_input = 1
num_output = 1
Num_layers = 2
class RNN(torch.nn.Module):
    def __init__(self,input_size,hidden_size,num_skills,Num_layers):
        super().__init__()
        self.rnn=torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=Num_layers,
            batch_first=True
        ).cuda()
        
        self.input = torch.nn.Linear(in_features=input_size,out_features=input_size).cuda()
        # self.input1 = torch.nn.Linear(in_features=2*kc_number,out_features=kc_number).cuda()
        self.out=torch.nn.Linear(in_features=hidden_size,out_features=num_skills).cuda()

    def forward(self,x):
        # 一下关于shape的注释只针对单项
        # output: [batch_size, time_step, hidden_size]
        # h_n: [num_layers,batch_size, hidden_size] # 虽然LSTM的batch_first为True,但是h_n/c_n的第一维还是num_layers
        # c_n: 同h_n
        #indata = self.input(x)
#         print("x.shape",x.shape)
        if num_input == 0 :
            output,(h_n,h_c)=self.rnn(x)
        elif num_input == 1:
            indata = self.input(x)
            output,(h_n,h_c)=self.rnn(indata)
        elif num_input == 2:
            indata = self.input(x)
            in1 = self.input1(indata)
            output,(h_n,h_c)=self.rnn(in1)
        if num_output == 0:
            out=(output,h_n,h_c)
#             out = output
        else:   
            out = self.out(output)
#         self.rnn.flatten_parameters()
        #print(output.size())
        # output_in_last_timestep=output[:,-1,:] # 也是可以的
        #output_in_last_timestep=h_n[-1,:,:]
        # print(output_in_last_timestep.equal(output[:,-1,:])) #ture
        #x=self.out(output_in_last_timestep)
#         print("out.shape",out.shape)
        out1 = (out, output)
#         return out1
        return out