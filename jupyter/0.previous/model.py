import torch.nn as nn
import torch.optim
import os
import numpy as np
import numpy
from tqdm import tqdm
'''
定义模型
'''
from lib import  ws,max_len,hidden_size
from  torch import  optim
import torch.nn.functional as F
from dataset import get_dataloader
import lib
class my_model(nn.Module):
    def __init__(self):
        super(my_model, self).__init__()
        self.embedding=nn.Embedding(len(ws),100)#句子最大长度
        #加入LSTM
        self.lstm=nn.LSTM(input_size=100,hidden_size=hidden_size,num_layers=lib.num_layer,
                batch_first=True,bidirectional=lib.bidriectional,dropout=lib.dropout)
        self.fc1=nn.Linear(lib.hidden_size*2,2)
    def forward(self,input):
        x=self.embedding(input)
        x,(h_n,c_n)=self.lstm(x)
        '''
        x:[batch_size,max_len,2*hidden_size]
        h_n:[num_layers*num_directions,batch_size,hidden_size]
        '''
        output=torch.cat([h_n[-2,:,:],h_n[-1,:,:]],dim=-1)#[batch_size,hidden_size*2]
        out=self.fc1(output)
        return F.log_softmax(out,dim=-1)
model1=my_model().to(lib.device)
optimizer=optim.Adam(model1.parameters(),0.001)
if os.path.exists("./model/model.pkl"):
    model1.load_state_dict(torch.load("E:/python eye/project2/NLP/model/model.pkl"))
    optimizer.load_state_dict(torch.load("E:/python eye/project2/NLP/model/optimizer.pkl"))
def train():
    for idx,(input,target) in enumerate(get_dataloader()):
        #梯度为0
        input=input.to(lib.device)
        target=target.to(lib.device)
 
        optimizer.zero_grad()
        output=model1(input)
        loss=F.nll_loss(output,target)
        loss.backward()#梯度返回
        optimizer.step()
        print(idx,loss.item())
        if idx%100==0:
            torch.save(model1.state_dict(), "E:/python eye/project2/NLP/model/model.pkl")
            torch.save(optimizer.state_dict(),"E:/python eye/project2/NLP/model/optimizer.pkl")
def eval():#魔性的评估
    loss_list=[]#所有损失
    acc_list=[]#准确率
    for idx,(input,target) in tqdm(enumerate(get_dataloader(False,batch_size=lib.test_batch_size))):
        input=input.to(lib.device)#到gpu上面
        target=target.to(lib.device)
        with torch.no_grad():
            output=model1(input)
            cur_loss=F.nll_loss(output,target)
            loss_list.append(cur_loss.cpu().item())
             #计算准确率
            pred=output.max(1)[1]#不是很理解
            curacc=pred.eq(target).float().mean()
            acc_list.append(curacc.cpu().item())#转成cpu类型
    print("total loss,acc:",np.mean(loss_list),np.mean(acc_list))
if __name__=='__main__':
    eval()
 
 